"""Selective checkpointing of MiniMax-H3's native MUSA FlashAttention results."""

import math
import threading
import warnings
import weakref

import torch
from torch._prims_common import is_non_overlapping_and_dense
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.checkpoint import checkpoint


class _HostResult:
    def __init__(self, tensors, streams):
        output, lse, mask = tensors
        self.mask = mask.detach()
        self.layout = [(t.shape, t.stride()) for t in (output, lse)]
        self.device = output.device
        self.streams = streams
        self.prefetched = None
        self.lock = threading.Lock()
        # Use the MUSA pinned caching allocator. torch_musa's async copy records
        # the host allocation on the copy stream, including if the graph is dropped.
        self.host = [torch.empty(t.numel(), device="cpu", dtype=t.dtype, pin_memory=True)
                     for t in (output, lse)]
        d2h, _ = streams
        d2h.wait_stream(torch.musa.current_stream(self.device))
        with torch.no_grad(), torch.musa.stream(d2h):
            for src, dst in zip((output, lse), self.host):
                # BHSD can be dense BSHD physical storage. Both DMA endpoints
                # must be contiguous and same dtype to avoid a layout kernel.
                dst.copy_(src.as_strided((src.numel(),), (1,)), non_blocking=True)
                src.record_stream(d2h)
            self.downloaded = d2h.record_event()

    def _prefetch(self):
        if self.prefetched is not None:
            return
        _, h2d = self.streams
        h2d.wait_event(self.downloaded)
        restored = []
        with torch.no_grad(), torch.musa.stream(h2d):
            for src, (shape, stride) in zip(self.host, self.layout):
                flat = torch.empty_like(src, device=self.device)
                flat.copy_(src, non_blocking=True)
                restored.append(flat.as_strided(shape, stride))
            self.prefetched = (restored, h2d.record_event())

    def prefetch(self):
        with self.lock:
            self._prefetch()

    def restore(self):
        with self.lock:
            self._prefetch()
            tensors, event = self.prefetched
            self.prefetched = None
        consumer = torch.musa.current_stream(self.device)
        consumer.wait_event(event)
        for tensor in tensors:
            tensor.record_stream(consumer)
        return (*tensors, self.mask)


class _ReplayFlash(TorchDispatchMode):
    """Replace only the native forward below autograd; keep its backward formula."""

    def __init__(self, op, result):
        super().__init__()
        self.op = op
        self.result = result

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func == self.op:
            return self.result
        return func(*args, **(kwargs or {}))


class _LayerCache:
    def __init__(self, policy, on_host):
        self.policy = policy
        self.on_host = on_host
        self.entries = []
        self.started = False

    def checkpoint(self, block, hidden, kwargs):
        op = torch.ops.aten._scaled_dot_product_attention_flash_musa.default

        def run(x, **block_kwargs):
            # Non-reentrant checkpoint calls run once in forward. Every later
            # invocation, including retain_graph replay, gets its own cursor.
            replay = iter(self.entries) if self.started else None
            self.started = True

            def segment_attention(q, k, v, *, scale):
                if replay is None:
                    result = op(q, k, v, scale=scale)
                    entry = (self.policy.save_to_host(result) if self.on_host else
                             tuple(t.detach() for t in result))
                    self.entries.append(entry)
                    return result[0]
                entry = next(replay)
                if entry is None:  # Budget or layout selected ordinary recomputation.
                    return op(q, k, v, scale=scale)[0]
                result = entry.restore() if self.on_host else entry
                with _ReplayFlash(op, result):
                    return op(q, k, v, scale=scale)[0]

            return block(x, segment_attention=segment_attention, **block_kwargs)

        return checkpoint(run, hidden, use_reentrant=False, **kwargs)

    def prefetch_at_backward(self, output):
        owner = weakref.ref(self)

        def prefetch(grad):
            cache = owner()
            if cache is not None:
                for entry in cache.entries:
                    if entry is not None:
                        entry.prefetch()
            return grad

        if output.requires_grad:
            output.register_hook(prefetch)


class H3ActivationOffload:
    """One policy per DiT. Pinned payload accounting spans all live graphs."""

    def __init__(self, depth, cpu_layers, gpu_layers, host_gib):
        if min(cpu_layers, gpu_layers) < 0 or cpu_layers + gpu_layers > depth:
            raise ValueError(f"H3 CPU/GPU activation layer counts must be nonnegative and sum to at most {depth}.")
        if not math.isfinite(host_gib) or host_gib < 0 or (cpu_layers and host_gib == 0):
            raise ValueError("H3 activation host budget must be finite and positive when CPU layers are enabled.")
        self.cpu_layers = cpu_layers
        self.gpu_start = depth - gpu_layers
        self.host_limit = int(host_gib * 2**30)
        self.host_bytes = 0
        self.streams = {}
        self.lock = threading.RLock()

    def _release_bytes(self, size):
        with self.lock:
            self.host_bytes -= size

    def save_to_host(self, tensors):
        output, lse, mask = tensors
        if mask.numel() or not all(is_non_overlapping_and_dense(t) for t in (output, lse)):
            warnings.warn("H3 FA returned a nonempty mask or non-dense layout; recomputing this segment.")
            return None
        size = output.nbytes + lse.nbytes
        with self.lock:
            if self.host_bytes + size > self.host_limit:
                warnings.warn("H3 activation pinned payload limit reached; recomputing the excess FA segments.")
                return None
            device = output.device
            if device not in self.streams:
                self.streams[device] = (torch.musa.Stream(device=device), torch.musa.Stream(device=device))
            self.host_bytes += size
        try:
            entry = _HostResult(tensors, self.streams[device])
        except BaseException:
            self._release_bytes(size)
            raise
        weakref.finalize(entry, self._release_bytes, size)
        return entry

    def begin_forward(self, device):
        if device.type != "musa" or not hasattr(torch.ops.aten, "_scaled_dot_product_attention_flash_musa"):
            raise RuntimeError("H3 activation caching requires native MUSA Flash SDPA.")
        previous = None

        def run(index, block, hidden, **kwargs):
            nonlocal previous
            on_host = index < self.cpu_layers
            cache = None
            if on_host or index >= self.gpu_start:
                cache = _LayerCache(self, on_host)
                output = cache.checkpoint(block, hidden, kwargs)
            else:
                output = checkpoint(block, hidden, use_reentrant=False, **kwargs)
            if previous is not None:
                previous.prefetch_at_backward(output)
            previous = cache if on_host else None
            return output

        return run
