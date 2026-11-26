"""Selective checkpointing of MiniMax-H3's native MUSA FlashAttention results."""

import math
import threading
import warnings
import weakref

import torch
from torch._prims_common import is_non_overlapping_and_dense
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.checkpoint import checkpoint


class _Transfer:
    """Hold storage until its allocation stream has joined an async copy."""

    def __init__(self, tensors, owner, done):
        self.tensors = list(tensors)
        self.owner = owner
        self.done = done
        # Also join before releasing storage if a prefetched graph is dropped.
        self.finish = weakref.finalize(self, self._finish, self.tensors, owner, done)

    @staticmethod
    def _finish(tensors, owner, done):
        owner.wait_event(done)
        tensors.clear()

    def take(self):
        tensors = tuple(self.tensors)
        self.finish()
        return tensors


class _PrefetchCleanup:
    """Discard unused prefetches when a backward task completes or is aborted."""

    def __init__(self, entries):
        self.finish = weakref.finalize(self, self._finish, [weakref.ref(e) for e in entries if e is not None])

    @staticmethod
    def _finish(entries):
        for ref in entries:
            entry = ref()
            if entry is not None:
                entry.discard_prefetch()


class _HostResult:
    def __init__(self, tensors, streams):
        output, lse, mask = tensors
        self.mask = mask.detach()
        self.layout = [(t.shape, t.stride()) for t in (output, lse)]
        self.device = output.device
        self.streams = streams
        self.producer = torch.musa.current_stream(self.device)
        self.prefetched = None
        self.lock = threading.Lock()
        # Use the MUSA pinned caching allocator. torch_musa's async copy records
        # the host allocation on the copy stream, including if the graph is dropped.
        self.host = [torch.empty(t.numel(), device="cpu", dtype=t.dtype, pin_memory=True)
                     for t in (output, lse)]
        d2h, _ = streams
        d2h.wait_stream(self.producer)
        try:
            with torch.no_grad(), torch.musa.stream(d2h):
                for src, dst in zip((output, lse), self.host):
                    # Preserve dense physical layout so this remains a DMA copy.
                    dst.copy_(src.as_strided((src.numel(),), (1,)), non_blocking=True)
                self.downloaded = d2h.record_event()
            self.source = _Transfer([output.detach(), lse.detach()], self.producer, self.downloaded)
        except BaseException:
            # Source arguments stay alive through this join even on a partial copy.
            self.producer.wait_stream(d2h)
            raise

    def finish_forward(self):
        # Called after the block's remaining forward work has been submitted.
        self.source.finish()

    def _prefetch(self, consumer):
        if self.prefetched is not None and self.prefetched.owner != consumer:
            # Autograd normally uses the original forward stream. If replay is
            # requested on another stream, rebuild there rather than allowing
            # a side-stream consumer to escape the allocation owner's lifetime.
            self.prefetched.finish()
            self.prefetched = None
        if self.prefetched is not None:
            return
        _, h2d = self.streams
        restored = []
        # Allocate on the future consumer, not on the copy stream. Recomputed
        # O/LSE remain live through FA backward via autograd's own references;
        # every subsequent use/reuse is then ordered on this allocation stream.
        with torch.no_grad(), torch.musa.stream(consumer):
            for src, (shape, stride) in zip(self.host, self.layout):
                flat = torch.empty_like(src, device=self.device)
                restored.append(flat.as_strided(shape, stride))
        # Cached allocations may still be in use by earlier consumer-stream
        # kernels, so DMA must also wait for their prior use before overwriting.
        h2d.wait_stream(consumer)
        h2d.wait_event(self.downloaded)
        try:
            with torch.no_grad(), torch.musa.stream(h2d):
                for src, dst in zip(self.host, restored):
                    dst.as_strided((dst.numel(),), (1,)).copy_(src, non_blocking=True)
                uploaded = h2d.record_event()
            self.prefetched = _Transfer(restored, consumer, uploaded)
        except BaseException:
            consumer.wait_stream(h2d)
            raise

    def prefetch(self):
        with self.lock:
            self._prefetch(self.producer)

    def discard_prefetch(self):
        with self.lock:
            if self.prefetched is not None:
                self.prefetched.finish()
                self.prefetched = None

    def restore(self):
        consumer = torch.musa.current_stream(self.device)
        with self.lock:
            self._prefetch(consumer)
            # Join before handing ownership to autograd. Do not release or
            # overwrite these tensors at checkpoint replay exit: FA backward
            # still needs them after replay has returned (or stopped early).
            tensors = self.prefetched.take()
            self.prefetched = None
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

            try:
                return block(x, segment_attention=segment_attention, **block_kwargs)
            finally:
                if self.on_host:
                    for entry in self.entries:
                        if entry is not None:
                            if replay is None:
                                entry.finish_forward()
                            else:
                                # Includes early-stop replay and failures. Only
                                # unused prefetches are released here; restored
                                # tensors belong to the original autograd graph.
                                entry.discard_prefetch()

        return checkpoint(run, hidden, use_reentrant=False, **kwargs)

    def prefetch_at_backward(self, output):
        owner = weakref.ref(self)

        def prefetch(grad):
            cache = owner()
            if cache is not None:
                cleanup = _PrefetchCleanup(cache.entries)
                # Partial autograd.grad may never replay the prefetched block.
                # The callback owns the cleanup guard until task completion;
                # its finalizer also runs if the engine discards an aborted task.
                torch.autograd.Variable._execution_engine.queue_callback(
                    lambda cleanup=cleanup: cleanup.finish()
                )
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
        self.depth = depth
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
            if on_host and index == self.depth - 1:
                # Bootstrap backward's first restore before replay starts. Put
                # this copy ahead of the previous block's lookahead in the DMA
                # queue; both can overlap this block's recomputation prefix.
                cache.prefetch_at_backward(output)
            if previous is not None:
                previous.prefetch_at_backward(output)
            previous = cache if on_host else None
            return output

        return run
