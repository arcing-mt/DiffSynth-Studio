"""Backend-independent FSDP2 controls for the MiniMax-H3 training entry point."""

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, field
import os

import torch


def _boolean(value):
    value = value.lower()
    if value in ("true", "1", "yes", "on"):
        return True
    if value in ("false", "0", "no", "off"):
        return False
    raise argparse.ArgumentTypeError("Expected true/false or 1/0.")


def _nonnegative(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("Expected a non-negative integer.")
    return value


def add_fsdp2_training_options(parser):
    group = parser.add_argument_group("FSDP2 training")
    group.add_argument("--fsdp2_sync_each_batch", type=_boolean, nargs="?", const=True, default=None,
                       help="Synchronize gradients every microstep while retaining gradient accumulation. Default: false.")
    group.add_argument("--fsdp2_reshard_after_backward", type=_boolean, nargs="?", const=True, default=None,
                       help="Reshard after every backward. False retains parameters between accumulation microsteps; the optimizer boundary always reshards. Default: true.")
    for name, help_text in (
        ("param", "Compute parameter dtype; auto follows the launcher. float32 requires an FP32-capable attention backend (e.g. DIFFSYNTH_ATTENTION_IMPLEMENTATION=torch)."),
        ("reduce", "Gradient reduction dtype; auto uses the compute parameter dtype."),
    ):
        group.add_argument(f"--fsdp2_{name}_dtype", choices=("auto", "bfloat16", "float16", "float32"), default=None,
                           help=help_text)
    group.add_argument("--fsdp2_output_dtype", choices=("none", "bfloat16", "float16", "float32"), default=None,
                       help="Cast FSDP module outputs to this dtype. Default: none.")
    group.add_argument("--fsdp2_cast_forward_inputs", type=_boolean, nargs="?", const=True, default=None,
                       help="Cast floating-point FSDP inputs to param_dtype. Default: false, preserving H3 position/angle precision.")
    group.add_argument("--fsdp2_forward_prefetch", type=_nonnegative, default=None,
                       help="Number of following DiT blocks to explicitly prefetch. Default: 0 (no explicit forward prefetch).")
    group.add_argument("--fsdp2_backward_prefetch", type=_nonnegative, default=None,
                       help="Number of preceding DiT blocks to explicitly prefetch. Default: 0 (PyTorch automatic backward prefetch, not disabled).")
    group.add_argument("--fsdp2_dp_replicate_size", type=int, default=None,
                       help="HSDP replication degree. Infer the omitted shard degree from world size.")
    group.add_argument("--fsdp2_dp_shard_size", type=int, default=None,
                       help="FSDP/HSDP shard degree. Infer the omitted replication degree from world size. Product must equal world size.")
    group.add_argument("--fsdp2_hsdp_defer_all_reduce", type=_boolean, nargs="?", const=True, default=None,
                       help="Defer HSDP replica all-reduce until the optimizer boundary while reduce-scattering every microstep. Requires HSDP and fsdp2_sync_each_batch=true. Default: false.")
    return parser


def fsdp2_accelerator_kwargs(args):
    """Build options that must be known before Accelerator creates its mesh/scaler."""
    kwargs = {}
    param_dtype = getattr(args, "fsdp2_param_dtype", None)
    if param_dtype not in (None, "auto"):
        kwargs["mixed_precision"] = {"bfloat16": "bf16", "float16": "fp16", "float32": "no"}[param_dtype]

    replicate = getattr(args, "fsdp2_dp_replicate_size", None)
    shard = getattr(args, "fsdp2_dp_shard_size", None)
    if replicate is not None or shard is not None:
        from accelerate import ParallelismConfig

        world = torch.distributed.get_world_size() if torch.distributed.is_initialized() else int(os.environ.get("WORLD_SIZE", "1"))
        if any(size is not None and size < 1 for size in (replicate, shard)):
            raise ValueError("FSDP2 replicate/shard sizes must be positive.")
        if replicate is None:
            replicate = world // shard
        if shard is None:
            shard = world // replicate
        if replicate * shard != world or replicate < 1 or shard < 1:
            raise ValueError(f"FSDP2 replicate_size ({replicate}) * shard_size ({shard}) must equal WORLD_SIZE ({world}).")
        if replicate > 1 and shard == 1:
            raise ValueError("HSDP requires dp_shard_size > 1; use DDP for replication without sharding.")
        kwargs["parallelism_config"] = ParallelismConfig(dp_replicate_size=replicate, dp_shard_size=shard)
    return kwargs


@dataclass
class FSDP2TrainingOptions:
    computation_dtype: torch.dtype
    reshard_after_backward: bool = True
    forward_prefetch: int = 0
    backward_prefetch: int = 0
    cast_forward_inputs: bool = False
    hsdp_defer_all_reduce: bool = False
    _reshard_enabled: bool = field(default=True, init=False, repr=False)
    _all_reduce_enabled: bool = field(default=True, init=False, repr=False)

    def prepare_model(self, model):
        self._reshard_enabled = True
        self._all_reduce_enabled = True
        model.pipe.dit._fsdp2_cast_forward_inputs = self.cast_forward_inputs
        if not (self.forward_prefetch or self.backward_prefetch):
            return
        from torch.distributed.fsdp import FSDPModule

        blocks = list(model.pipe.dit.blocks)
        if not blocks or not all(isinstance(block, FSDPModule) for block in blocks):
            raise ValueError("Explicit H3 prefetch requires FSDP2 to wrap each MiniMaxH3DiTBlock.")
        for index, block in enumerate(blocks):
            if self.forward_prefetch:
                block.set_modules_to_forward_prefetch(blocks[index + 1:index + 1 + self.forward_prefetch])
            if self.backward_prefetch:
                previous = blocks[max(0, index - self.backward_prefetch):index]
                block.set_modules_to_backward_prefetch(list(reversed(previous)))

    def _set_reshard(self, model, enabled):
        if self._reshard_enabled != enabled:
            model.set_reshard_after_backward(enabled)
            self._reshard_enabled = enabled

    def _set_all_reduce(self, model, enabled):
        if self._all_reduce_enabled != enabled:
            model.set_requires_all_reduce(enabled)
            self._all_reduce_enabled = enabled

    def _restore(self, model):
        self._set_reshard(model, True)
        self._set_all_reduce(model, True)

    @contextmanager
    def training(self, model):
        try:
            yield
        finally:
            # Also restore after errors between microsteps (e.g. data loading).
            self._restore(model)

    @contextmanager
    def microstep(self, accelerator, model):
        # Call inside accelerator.accumulate(), after it resolves the boundary.
        # This also handles a short final accumulation window at dataloader end.
        if self.reshard_after_backward and not self.hsdp_defer_all_reduce:
            yield
            return
        # Set only at window boundaries, not twice for every microstep.
        if not self.reshard_after_backward:
            self._set_reshard(model, accelerator.sync_gradients)
        if self.hsdp_defer_all_reduce:
            # sync_each_batch=true keeps reduce-scatter enabled and prevents
            # Accelerator.no_sync() from resetting both reduction flags.
            self._set_all_reduce(model, accelerator.sync_gradients)
        try:
            yield
        except BaseException:
            self._restore(model)
            raise


def configure_fsdp2_training(accelerator, args):
    requested = any(value is not None for name, value in vars(args).items() if name.startswith("fsdp2_"))
    if not getattr(accelerator, "is_fsdp2", False):
        if requested:
            raise ValueError("--fsdp2_* options require an Accelerate FSDP2 launch configuration.")
        return
    if getattr(args, "enable_model_cpu_offload", False):
        raise ValueError("FSDP2 requires model preparation; use fsdp_offload_params in the launch config instead of --enable_model_cpu_offload.")

    sync_each_batch = bool(getattr(args, "fsdp2_sync_each_batch", False))
    defer_all_reduce = bool(getattr(args, "fsdp2_hsdp_defer_all_reduce", False))
    if defer_all_reduce:
        if not sync_each_batch:
            raise ValueError("--fsdp2_hsdp_defer_all_reduce requires --fsdp2_sync_each_batch true.")
        parallelism = getattr(accelerator, "parallelism_config", None)
        if (getattr(parallelism, "dp_replicate_size", 1) <= 1
                or getattr(parallelism, "dp_shard_size", 1) <= 1):
            raise ValueError("--fsdp2_hsdp_defer_all_reduce requires HSDP with both replicate and shard sizes greater than 1.")

    from torch.distributed.fsdp import MixedPrecisionPolicy

    default_dtype = {"bf16": "bfloat16", "fp16": "float16", "no": "float32"}.get(accelerator.mixed_precision)
    if default_dtype is None:
        raise ValueError("H3 FSDP2 supports bf16, fp16, or no mixed precision.")
    param_name = getattr(args, "fsdp2_param_dtype", None)
    param_dtype = getattr(torch, default_dtype if param_name in (None, "auto") else param_name)
    reduce_name = getattr(args, "fsdp2_reduce_dtype", None)
    reduce_dtype = param_dtype if reduce_name in (None, "auto") else getattr(torch, reduce_name)
    output_name = getattr(args, "fsdp2_output_dtype", None)
    output_dtype = None if output_name in (None, "none") else getattr(torch, output_name)
    cast_inputs = bool(getattr(args, "fsdp2_cast_forward_inputs", False))
    accelerator.state.fsdp_plugin.mixed_precision_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype, reduce_dtype=reduce_dtype,
        output_dtype=output_dtype, cast_forward_inputs=cast_inputs,
    )
    # Keep Accelerator's normal accumulation/optimizer boundary; only change
    # whether accumulate() enters no_sync() on intermediate microsteps.
    accelerator.gradient_state.plugin_kwargs["sync_each_batch"] = sync_each_batch
    reshard = getattr(args, "fsdp2_reshard_after_backward", None)
    accelerator.diffsynth_fsdp2_options = FSDP2TrainingOptions(
        computation_dtype=param_dtype,
        reshard_after_backward=True if reshard is None else reshard,
        forward_prefetch=getattr(args, "fsdp2_forward_prefetch", None) or 0,
        backward_prefetch=getattr(args, "fsdp2_backward_prefetch", None) or 0,
        cast_forward_inputs=cast_inputs,
        hsdp_defer_all_reduce=defer_all_reduce,
    )
