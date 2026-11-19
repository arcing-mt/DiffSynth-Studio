"""Synthetic inputs, precision controls and block compilation for H3 training."""

import os
from types import MethodType

import torch

from diffsynth.models.minimax_h3_dit import (
    MiniMaxH3DiT, MiniMaxH3DiTBlock, MiniMaxH3Attention,
    MiniMaxH3TokenRefinerBlock, unpatchify_video, unpack_audio,
)


FP32_MODULES = (
    "video_patch_proj", "audio_patch_proj", "time_embedder",
    "final_layer.video_out", "final_layer.audio_out", "rope",
)

COMPILE_MODULE_TYPES = (MiniMaxH3DiTBlock, MiniMaxH3TokenRefinerBlock)


def h3_env_enabled(name, default=False):
    value = os.environ.get(name, "1" if default else "0").lower()
    if value not in ("1", "true", "yes", "on", "0", "false", "no", "off"):
        raise ValueError(f"{name} must be a boolean (0/1 or false/true), got {value!r}.")
    return value in ("1", "true", "yes", "on")


def _fp32_forward(self, x, **kwargs):
    # FSDP controls parameter dtype; disabling autocast also protects the GEMMs.
    if "dtype" in kwargs:
        kwargs["dtype"] = torch.float32
    with torch.autocast(device_type=x.device.type, enabled=False):
        return self._h3_original_forward(x.float(), **kwargs)


def configure_h3_fp32_modules(dit, model_configs=()):
    if not h3_env_enabled("MINIMAX_H3_FP32_MODULES"):
        return
    if type(dit) is not MiniMaxH3DiT or getattr(dit, "vdn_enabled", False) or getattr(dit, "quantize_config", None) is not None:
        raise ValueError("MINIMAX_H3_FP32_MODULES requires the ordinary, unquantized H3 DiT.")
    for name in FP32_MODULES:
        module = dit.get_submodule(name)
        module.float()
        if not hasattr(module, "_h3_original_forward"):
            module._h3_original_forward = module.forward
            module.forward = MethodType(_fp32_forward, module)

    # Reload only the protected weights without the loader's BF16 rounding.
    # Never materialize a full FP32 DiT on the accelerator.
    if model_configs:
        from diffsynth.core.vram.disk_map import DiskMap

        parameters = {name: p for name, p in dit.named_parameters()
                      if any(name.startswith(prefix + ".") for prefix in FP32_MODULES)}
        remaining = set(parameters)
        for config in model_configs:
            state = DiskMap(config.path, "cpu")
            for name in list(remaining):
                if name in state:
                    with torch.no_grad():
                        parameters[name].copy_(state[name])
                    remaining.remove(name)
            if not remaining:
                break
        if remaining:
            raise ValueError(f"Missing original H3 FP32 weights: {sorted(remaining)}")
    dit._h3_fp32_modules = True


def configure_h3_compile(dit, enabled=False, safe_norm=False, safe_attention=False):
    """Compile H3's repeated blocks before FSDP wrapping.

    This follows the benchmark zip: ``dynamic=False`` is used for the fixed
    packed shapes, while selected normalization/attention functions can be
    kept outside the compiled graph. The root DiT, pipeline and output
    projections remain eager. Only forwards are compiled, leaving FSDP hooks
    outside the graph and preserving module types and checkpoint keys.
    """
    if not enabled:
        return 0
    if getattr(dit, "_h3_compile_enabled", False):
        return 0
    if not hasattr(torch, "compile") or not hasattr(torch, "compiler"):
        raise RuntimeError("H3 block compilation requires torch.compile and torch.compiler.disable.")
    if type(dit) is not MiniMaxH3DiT or getattr(dit, "vdn_enabled", False) or getattr(dit, "quantize_config", None) is not None:
        raise ValueError("H3 block compilation requires the ordinary, unquantized H3 DiT.")

    if safe_norm:
        for module in dit.modules():
            if isinstance(module, torch.nn.RMSNorm) and not getattr(module, "_h3_compile_safe_norm", False):
                module.forward = torch.compiler.disable(module.forward)
                module._h3_compile_safe_norm = True

    if safe_attention:
        for module in dit.modules():
            if isinstance(module, MiniMaxH3Attention):
                module.compute_attention = torch.compiler.disable(module.compute_attention)

    compiled = 0
    for module in dit.modules():
        if type(module) in COMPILE_MODULE_TYPES and not getattr(module, "_h3_compile_enabled", False):
            module.compile(dynamic=False)
            module._h3_compile_enabled = True
            compiled += 1
    # Zip's AdaLayerNormOut excludes the separately managed output projections.
    dit.final_layer.modulate = torch.compile(dit.final_layer.modulate, dynamic=False)
    compiled += 1
    dit._h3_compile_enabled = True
    return compiled


def shard_h3_fp32_modules(model, accelerator):
    """Run before optimizer construction; Accelerate shards the remaining model."""
    if not getattr(model.pipe.dit, "_h3_fp32_modules", False):
        return
    if not getattr(accelerator, "is_fsdp2", False):
        if accelerator.distributed_type.name in ("DEEPSPEED", "FSDP"):
            raise ValueError("MINIMAX_H3_FP32_MODULES supports FSDP2, DDP, or single-device training; not FSDP1/DeepSpeed.")
        return
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    plugin = accelerator.state.fsdp_plugin
    if plugin.cpu_ram_efficient_loading:
        raise ValueError("MINIMAX_H3_FP32_MODULES requires fsdp_cpu_ram_efficient_loading=false.")
    mesh = getattr(accelerator, "torch_device_mesh", None)
    if mesh is not None:
        mesh = mesh[tuple(accelerator.parallelism_config.fsdp_dim_names)]
    model.to(device=accelerator.device)
    for name in FP32_MODULES:
        fully_shard(
            model.pipe.dit.get_submodule(name), mesh=mesh,
            reshard_after_forward=plugin.reshard_after_forward,
            offload_policy=plugin.cpu_offload,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.float32,
                reduce_dtype=plugin.mixed_precision_policy.reduce_dtype,
                cast_forward_inputs=False,
            ),
        )


class MiniMaxH3SyntheticDataset(torch.utils.data.Dataset):
    """Fixed per-rank/microstep samples, using the reproduction zip's formulas.

    DataLoader must keep order and use zero workers. Each rank caches only its
    accumulation window on its device, so later steps do not regenerate inputs.
    """

    load_from_cache = True
    shuffle = False

    def __init__(self, height, width, num_frames, repeat, world_size=1, accumulation=1,
                 device="cpu", dtype=torch.bfloat16, use_gradient_checkpointing=False,
                 use_gradient_checkpointing_offload=False):
        if height is None or width is None or min(height, width) < 32 or num_frames < 1 or repeat < 1:
            raise ValueError("Synthetic H3 inputs require height/width >= 32, positive num_frames and dataset_repeat.")
        self.grid = ((num_frames - 1) // 4 + 1, height // 32, width // 32)
        self.video_tokens = self.grid[0] * self.grid[1] * self.grid[2]
        self.audio_tokens = max(1, round(num_frames / 24 * 40) * 2)
        self.text_tokens = 256
        self.seq_len = self.video_tokens + self.audio_tokens + self.text_tokens
        self.repeat, self.world_size, self.accumulation = repeat, world_size, accumulation
        self.device, self.dtype = device, dtype
        self.checkpoint = use_gradient_checkpointing
        self.checkpoint_offload = use_gradient_checkpointing_offload
        self.samples = {}

    def __len__(self):
        return self.repeat

    def __getitem__(self, index):
        sample = (index % self.world_size) * self.accumulation + (index // self.world_size) % self.accumulation
        if sample not in self.samples:
            self.samples[sample] = self.make_sample(sample)
        return self.samples[sample]

    def make_sample(self, sample):
        def rand(field, shape, dtype=torch.float32):
            generator = torch.Generator().manual_seed(271828 + sample * 97 + field * 10007)
            return torch.randn(shape, generator=generator).bfloat16().to(device=self.device, dtype=dtype)

        t, h, w = self.grid
        video = rand(1, (self.video_tokens, 96))
        audio = rand(2, (self.audio_tokens, 32))
        text = rand(3, (self.text_tokens, 5120), self.dtype)
        text_pos = torch.arange(self.text_tokens, device=self.device)
        img_pos = torch.arange(self.text_tokens, self.text_tokens + self.video_tokens, device=self.device)
        audio_pos = torch.arange(self.text_tokens + self.video_tokens, self.seq_len, device=self.device)
        positions = torch.zeros((1, self.seq_len, 3), device=self.device)
        positions[0, img_pos] = torch.cartesian_prod(torch.arange(t), torch.arange(h), torch.arange(w)).to(self.device).float()
        # H3's modality ordering is video=0, text=1, audio=2.
        tags = torch.zeros(self.seq_len, device=self.device, dtype=torch.long)
        tags[audio_pos], tags[text_pos] = 2, 1
        shared = dict(
            input_latents=unpatchify_video(video, t, h * 2, w * 2),
            audio_input_latents=unpack_audio(audio, 1, self.audio_tokens),
            packed=dict(img_pos=img_pos, audio_pos=audio_pos, text_pos=text_pos,
                        img_position_ids=positions, token_tags=tags, seq_len=self.seq_len,
                        cu_seqlens=torch.tensor([0, self.seq_len], device=self.device, dtype=torch.int32)),
            use_gradient_checkpointing=self.checkpoint,
            use_gradient_checkpointing_offload=self.checkpoint_offload,
            synthetic_inputs=True,
        )
        return shared, {"prompt_embeds": text}, {}
