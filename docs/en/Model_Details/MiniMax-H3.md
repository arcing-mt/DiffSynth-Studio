# MiniMax-H3

MiniMax H3 is a general-purpose omni-modal generation system. It supports unified understanding of multimodal contexts composed of text, images, video, and audio, and can generate videos of up to 2K resolution and up to 15 seconds in length with native stereo audio. Thanks to a system design oriented toward task generalization, H3 already acquires broad multimodal context understanding and generation capabilities during pre-training, and therefore excels at following complex multimodal instructions.

## Installation

Before performing model inference and training, please install DiffSynth-Studio first.

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

For more information on installation, please refer to [Setup Dependencies](../Pipeline_Usage/Setup.md).

## Synthetic Inputs and Training Precision

`--synthetic_inputs` supplies fixed random video/audio latents and text embeddings for `sft:train`, without a dataset, processor, or reference media. It requires `--dataset_num_workers 0` and `--training_cfg_scale 1`. Samples have separate seeds per rank/accumulation microstep and are reused across updates. `--dataset_repeat` is the global sample count.

The reproduction zip's geometry is preserved: video tokens are `((num_frames-1)//4+1)*(height//32)*(width//32)`, text has 256 tokens, and audio has `max(1, round(num_frames/24*40)*2)` tokens. For 720×1280 at 24fps for 10 seconds, use `--height 720 --width 1280 --num_frames 240`: 52,800 video + 256 text + 800 audio = 53,856 tokens. The original formula floors height 720 to 22 spatial cells. These are synthetic shapes, not actual VAE encodings, so the VAE's `17n+5` frame constraint does not apply.

|Environment variable|Repository default|Behavior|
|-|-|-|
|`MINIMAX_H3_ZERO_TARGET_LOSS`|`1`|Sum the FP32 mean squares of video and audio predictions, without scheduler or audio-loss weights. Synthetic inputs use model time 0.5 directly; real inputs retain their original noising. Set to `0` to restore flow-matching targets, noising, and weighting|
|`MINIMAX_H3_FP32_MODULES`|`0`|At the training entry point, run `video_patch_proj`, `audio_patch_proj`, `time_embedder`, `final_layer.video_out`, `final_layer.audio_out`, and `rope` in FP32 with autocast disabled. These map to the zip's `proj_in`, `audio_proj_in`, `time_embedder`, `proj_out`, `audio_proj_out`, and `rope`; requires an ordinary, unquantized H3 DiT|

The FP32 option reloads only these modules' original weights to avoid BF16 rounding. FSDP2 shards them with separate FP32 policies, preserving the configured mesh, reshard/offload settings, and reduction dtype. Other modules retain their original compute precision. Time embeddings are converted to each AdaLN linear layer's parameter dtype before that layer. FSDP1, DeepSpeed, and `fsdp_cpu_ram_efficient_loading=true` are currently unsupported with this option.

`start_ref2va_multinode.sh` defaults `MINIMAX_H3_ZERO_TARGET_LOSS`, `MINIMAX_H3_FP32_MODULES`, `MINIMAX_H3_MUSA_FUSED_ROPE`, and `MINIMAX_H3_MUSA_FUSED_SWIGLU` to `0`, allowing independent environment overrides. The script therefore uses flow-matching loss by default; the repository defaults remain as listed above. It uses FP32 reduction, a fixed learning rate of `1e-5`, `warm_up_steps=0`, and `max_grad_norm=0`. Microbatch 1, accumulation 2, four nodes/32 devices, global batch 64, optimizer, compilation, profiler, and saving settings otherwise remain unchanged.

Set `MODEL_DIR` and `MASTER_ADDR` through environment variables before launching. `REPO_DIR` defaults to the script directory, `PYTHON_BIN` to the current environment's `python`, and `OUTPUT_DIR` to `$REPO_DIR/runs/ref2va-fsdp2`. All paths can be overridden; use absolute paths for path settings. The script contains generic examples, without deployment-specific paths or addresses.

The launcher also supports peak-memory monitoring. Set `MEMORY_MONITOR_ENABLE=1` to write `peak_memory_rank*.csv` and matching JSON summaries under `MEMORY_MONITOR_DIR` (default `$OUTPUT_DIR/peak_memory`) on each rank. Each row includes current/peak allocated and reserved memory, the microstep, optimizer step, and whether gradients synchronized; `MEMORY_MONITOR_INTERVAL` controls the sampling interval in microsteps. Monitoring synchronizes the device before and after each sampled step so asynchronous kernels are included, which adds overhead; it is disabled by default.

## FSDP2 Training Options

The H3 training script accepts the following `--fsdp2_*` options with an Accelerate FSDP2 launch configuration. Omitted options preserve the current defaults. Boolean values accept `true/false` or `1/0`; omitting a boolean's value means `true`.

|Option|Default behavior|Purpose|
|-|-|-|
|`--fsdp2_sync_each_batch`|`false`|Synchronize gradients on every microstep while retaining the optimizer update interval from `--gradient_accumulation_steps`. This avoids keeping full accumulated gradients, at the cost of more communication|
|`--fsdp2_reshard_after_backward`|`true`|Reshard parameters after each backward. `false` retains parameters between accumulation microsteps to avoid the next forward all-gather, using more memory. The final backward before each optimizer update always reshards|
|`--fsdp2_param_dtype`|`auto`|Unsharded compute-parameter dtype: `bfloat16`, `float16`, or `float32`. `auto` follows the launcher. An explicit value also selects the Accelerate precision mode and H3 weight-loading/input dtype; FP16 uses gradient scaling and FP32 selects `mixed_precision=no`|
|`--fsdp2_reduce_dtype`|`auto`|Gradient reduction dtype: `bfloat16`, `float16`, or `float32`. `auto` uses the compute dtype|
|`--fsdp2_output_dtype`|`none`|FSDP module output dtype: `none`, `bfloat16`, `float16`, or `float32`. `none` leaves outputs unchanged. The policy also applies to the loss returned by the root training module|
|`--fsdp2_cast_forward_inputs`|`false`|Cast floating-point FSDP inputs to the compute dtype. H3 defaults to false to preserve FP64 positions and FP32 RoPE angles; enabling this changes their precision. Configured independently of output dtype|
|`--fsdp2_forward_prefetch`|`0`|Number of following DiT blocks to explicitly prefetch; `0` leaves explicit forward prefetch unset|
|`--fsdp2_backward_prefetch`|`0`|Number of preceding DiT blocks to explicitly prefetch; `0` retains PyTorch's automatic backward prefetch, rather than disabling it|
|`--fsdp2_dp_replicate_size`|Keep launcher configuration|HSDP replication degree. If this is the only topology option specified, infer the shard degree from world size|
|`--fsdp2_dp_shard_size`|Keep launcher configuration|FSDP/HSDP shard degree. If this is the only topology option specified, infer the replication degree from world size|
|`--fsdp2_hsdp_defer_all_reduce`|`false`|Keep reduce-scatter on every microstep and defer replica all-reduce until the optimizer boundary. Requires HSDP and `fsdp2_sync_each_batch=true`; invalid combinations raise an error|

`param_dtype` controls compute parameters, not Accelerate's FP32 parameter-shard/optimizer storage under mixed precision. Full FP32 computation requires an FP32-capable attention backend, for example `DIFFSYNTH_ATTENTION_IMPLEMENTATION=torch`, since FlashAttention supports FP16/BF16 only. Parameter resharding and gradient synchronization are independent. Deeper prefetch can keep more full parameters resident. The replication and shard degrees must multiply to world size; replication greater than 1 also requires sharding greater than 1. By default each HSDP gradient synchronization includes reduce-scatter and replica all-reduce. With deferral enabled, the former runs every microstep and the latter at the optimizer boundary, including a short final accumulation window.

When output and compute dtypes differ, only hidden activations are converted back before the next block or final head; positions and RoPE angles need not be cast. Explicitly selecting a different output dtype still incurs output casts and the necessary activation casts back. H3 blocks using the same precision policy can normally keep `output_dtype=none`. With `reshard_after_backward=false`, state changes only at accumulation-window boundaries, and normal or exceptional training exit restores the default.

Reduction dtype still follows compute dtype by default, and prefetch still uses two independent depth controls. In HSDP, combining `fsdp2_sync_each_batch=true` with `fsdp2_hsdp_defer_all_reduce=true` selects VeOmni's schedule: reduce-scatter every microstep and replica all-reduce at the accumulation boundary. Disabling deferral preserves the original synchronization behavior. Neither option changes accumulation length or gradient-clipping/optimizer/scheduler boundaries. Changing synchronization frequency, grouping, or precision can alter floating-point rounding; an equivalent gradient-accumulation objective does not guarantee bitwise-identical results.

For example, reuse an existing data cache, accumulate four microsteps, synchronize gradients every microstep, and explicitly prefetch one block:

```shell
STAGE=train bash examples/minimax_h3/model_training/full/MiniMax-H3-FL2VA-FSDP2.sh \
  --gradient_accumulation_steps 4 \
  --fsdp2_sync_each_batch true \
  --fsdp2_reshard_after_backward true \
  --fsdp2_reduce_dtype float32 \
  --fsdp2_forward_prefetch 1 \
  --fsdp2_backward_prefetch 1
```

With eight processes, append `--fsdp2_dp_replicate_size 2 --fsdp2_dp_shard_size 4 --fsdp2_hsdp_defer_all_reduce true` to select 2×4 HSDP with deferred replica all-reduce. Omit the last option to keep synchronizing both groups every microstep. The shell script passes these options only to training; preprocessing uses a separate single-process configuration.

## MUSA Kernel Optimization

Set either independent environment switch before launching standard MiniMax-H3 training. Both default to disabled:

```shell
export MINIMAX_H3_MUSA_FUSED_ROPE=1
export MINIMAX_H3_MUSA_FUSED_SWIGLU=1
```

`MINIMAX_H3_MUSA_FUSED_ROPE` replaces DiT attention's half-split partial RoPE with `torch.rope`, preserving the unrotated dimensions (normally the first 96 of 128 head dimensions are rotated). The kernel consumes shared FP32 angles and computes cos/sin internally; Python does not precompute cos/sin on this path. When disabled, cos/sin are computed once per forward and reused by all layers' Q/K operations and gradient checkpoint recomputation.

`MINIMAX_H3_MUSA_FUSED_SWIGLU` replaces `silu(gate) * up` with `aten.swish_glu` and its registered fused backward in both DiT and token-refiner MLPs. Both switches accept `1/0`, `true/false`, `yes/no`, and `on/off`. Regardless of these switches, DiT and token-refiner `cu_seqlens` are each converted to host boundaries once per forward and shared across their layers.

After loading the pipeline, `examples/minimax_h3/model_training/train.py` calls `apply_minimax_h3_musa_patch(self.pipe.dit)` to enable these four optimizations on the standard H3 DiT, preserving parameters and checkpoint keys. CPU initialization followed by moving to MUSA is supported: dispatch checks the actual tensor device and dtype, and non-MUSA inputs use the original operators. Loading an inference pipeline alone does not install the patch. Comfy, VDN, and ControlNet are outside this optimization's scope. For manually constructed standard H3 training models, import `apply_minimax_h3_musa_patch` from `diffsynth.core.musa_patch` and call `apply_minimax_h3_musa_patch(model)` before distributed wrapping.

The fused operators require the corresponding APIs in torch-musa. They can differ from the original BF16 expressions due to rounding, so keep other settings fixed when comparing training results or performance. Use `PROFILE_ENABLE=1` to inspect the selected kernels and their durations.

## Quick Start

Running the following code will quickly load the [DiffSynth-Studio/MiniMax-H3-NF4](https://www.modelscope.cn/models/DiffSynth-Studio/MiniMax-H3-NF4) NF4-quantized model and perform text-to-video-audio inference. VRAM management is enabled, and the framework automatically controls the loading of model parameters based on available VRAM, requiring a minimum of 7GB VRAM.

```python
import torch
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
from diffsynth.utils.data.audio_video import write_video_audio

vram_config = {
    "offload_dtype": torch.bfloat16,
    "offload_device": "cpu",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cpu",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
pipe = MiniMaxH3Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="DiffSynth-Studio/MiniMax-H3-NF4", origin_file_pattern="minimax-h3-fl2va-nf4.safetensors", **vram_config),
        ModelConfig(model_id="DiffSynth-Studio/MiniMax-H3-NF4", origin_file_pattern="minimax-h3-text-encoder-nf4.safetensors", **vram_config),
        ModelConfig(model_id="DiffSynth-Studio/MiniMax-H3-NF4", origin_file_pattern="video_vae_nf4.safetensors", **vram_config),
        ModelConfig(model_id="DiffSynth-Studio/MiniMax-H3-NF4", origin_file_pattern="audio_vae_nf4.safetensors", **vram_config),
    ],
    processor_config=ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/processor/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 2,
)

# Text -> Video + Audio
prompt = "A girl is very happy, she is speaking in english: “I enjoy working with Diffsynth-Studio, it's a perfect framework.”"
video, audio = pipe(
    prompt=prompt,
    height=480, width=832, num_frames=124, num_inference_steps=50, seed=0,
)
write_video_audio(
    video=video, audio=audio,
    output_path="t2va.mp4", fps=24, audio_sample_rate=32000,
)
```

## Model Overview

|Model ID|Inference|Low VRAM Inference|Full Training|Full Training Validation|LoRA Training|LoRA Training Validation|
|-|-|-|-|-|-|-|
|[MiniMax/MiniMax-H3: FL2VA](https://www.modelscope.cn/models/MiniMax/MiniMax-H3)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference/MiniMax-H3-FL2VA.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-FL2VA.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/full/MiniMax-H3-FL2VA.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/validate_full/MiniMax-H3-FL2VA.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/lora/MiniMax-H3-FL2VA.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/validate_lora/MiniMax-H3-FL2VA.py)|
|[MiniMax/MiniMax-H3: Ref2VA](https://www.modelscope.cn/models/MiniMax/MiniMax-H3)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference/MiniMax-H3-Ref2VA.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-Ref2VA.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/full/MiniMax-H3-Ref2VA.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/validate_full/MiniMax-H3-Ref2VA.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/lora/MiniMax-H3-Ref2VA.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/validate_lora/MiniMax-H3-Ref2VA.py)|
|[MiniMax/MiniMax-H3: Retake](https://www.modelscope.cn/models/MiniMax/MiniMax-H3)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference/MiniMax-H3-Retake.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-Retake.py)|-|-|-|-|
|[DiffSynth-Studio/MiniMax-H3-NF4: FL2VA](https://www.modelscope.cn/models/DiffSynth-Studio/MiniMax-H3-NF4)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference/MiniMax-H3-NF4-FL2VA.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-NF4-FL2VA.py)|-|-|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/lora/MiniMax-H3-NF4-FL2VA.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/validate_lora/MiniMax-H3-NF4-FL2VA.py)|
|[DiffSynth-Studio/MiniMax-H3-NF4: Ref2VA](https://www.modelscope.cn/models/DiffSynth-Studio/MiniMax-H3-NF4)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference/MiniMax-H3-NF4-Ref2VA.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-NF4-Ref2VA.py)|-|-|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/lora/MiniMax-H3-NF4-Ref2VA.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/validate_lora/MiniMax-H3-NF4-Ref2VA.py)|
|[Comfy-Org/MiniMax-H3: FL2VA pruned](https://www.modelscope.cn/models/Comfy-Org/MiniMax-H3)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference/MiniMax-H3-Pruned-FL2VA.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-Pruned-FL2VA.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/full/MiniMax-H3-Pruned-FL2VA.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/validate_full/MiniMax-H3-Pruned-FL2VA.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/lora/MiniMax-H3-Pruned-FL2VA.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/validate_lora/MiniMax-H3-Pruned-FL2VA.py)|
|[Comfy-Org/MiniMax-H3: Ref2VA pruned](https://www.modelscope.cn/models/Comfy-Org/MiniMax-H3)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference/MiniMax-H3-Pruned-Ref2VA.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-Pruned-Ref2VA.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/full/MiniMax-H3-Pruned-Ref2VA.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/validate_full/MiniMax-H3-Pruned-Ref2VA.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/lora/MiniMax-H3-Pruned-Ref2VA.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/validate_lora/MiniMax-H3-Pruned-Ref2VA.py)|
|[DiffSynth-Studio/MiniMax-H3-NF4: FL2VA pruned](https://www.modelscope.cn/models/DiffSynth-Studio/MiniMax-H3-NF4)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference/MiniMax-H3-NF4-Pruned-FL2VA.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-NF4-Pruned-FL2VA.py)|-|-|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/lora/MiniMax-H3-NF4-Pruned-FL2VA.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/validate_lora/MiniMax-H3-NF4-Pruned-FL2VA.py)|
|[DiffSynth-Studio/MiniMax-H3-NF4: Ref2VA pruned](https://www.modelscope.cn/models/DiffSynth-Studio/MiniMax-H3-NF4)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference/MiniMax-H3-NF4-Pruned-Ref2VA.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-NF4-Pruned-Ref2VA.py)|-|-|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/lora/MiniMax-H3-NF4-Pruned-Ref2VA.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/validate_lora/MiniMax-H3-NF4-Pruned-Ref2VA.py)|
|[Comfy-Org/MiniMax-H3: FL2VA int8_convrot](https://www.modelscope.cn/models/Comfy-Org/MiniMax-H3)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference/MiniMax-H3-Int8-ConvRot-FL2VA.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-Int8-ConvRot-FL2VA.py)|-|-|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/lora/MiniMax-H3-Int8-ConvRot-FL2VA.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/validate_lora/MiniMax-H3-Int8-ConvRot-FL2VA.py)|
|[Comfy-Org/MiniMax-H3: Ref2VA int8_convrot](https://www.modelscope.cn/models/Comfy-Org/MiniMax-H3)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference/MiniMax-H3-Int8-ConvRot-Ref2VA.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-Int8-ConvRot-Ref2VA.py)|-|-|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/lora/MiniMax-H3-Int8-ConvRot-Ref2VA.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/validate_lora/MiniMax-H3-Int8-ConvRot-Ref2VA.py)|
|[Comfy-Org/MiniMax-H3: FL2VA pruned int8_convrot](https://www.modelscope.cn/models/Comfy-Org/MiniMax-H3)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference/MiniMax-H3-Int8-ConvRot-Pruned-FL2VA.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-Int8-ConvRot-Pruned-FL2VA.py)|-|-|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/lora/MiniMax-H3-Int8-ConvRot-Pruned-FL2VA.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/validate_lora/MiniMax-H3-Int8-ConvRot-Pruned-FL2VA.py)|
|[Comfy-Org/MiniMax-H3: Ref2VA pruned int8_convrot](https://www.modelscope.cn/models/Comfy-Org/MiniMax-H3)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference/MiniMax-H3-Int8-ConvRot-Pruned-Ref2VA.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-Int8-ConvRot-Pruned-Ref2VA.py)|-|-|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/lora/MiniMax-H3-Int8-ConvRot-Pruned-Ref2VA.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/validate_lora/MiniMax-H3-Int8-ConvRot-Pruned-Ref2VA.py)|
|[Comfy-Org/MiniMax-H3: FL2VA pruned fp8](https://www.modelscope.cn/models/Comfy-Org/MiniMax-H3)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference/MiniMax-H3-FP8-Pruned-FL2VA.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-FP8-Pruned-FL2VA.py)|-|-|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/lora/MiniMax-H3-FP8-Pruned-FL2VA.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/validate_lora/MiniMax-H3-FP8-Pruned-FL2VA.py)|
|[Comfy-Org/MiniMax-H3: Ref2VA pruned fp8](https://www.modelscope.cn/models/Comfy-Org/MiniMax-H3)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference/MiniMax-H3-FP8-Pruned-Ref2VA.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-FP8-Pruned-Ref2VA.py)|-|-|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/lora/MiniMax-H3-FP8-Pruned-Ref2VA.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/validate_lora/MiniMax-H3-FP8-Pruned-Ref2VA.py)|
|[lightx2v/Minimax-h3-Turbo: FL2VA 4steps](https://www.modelscope.cn/models/lightx2v/Minimax-h3-Turbo)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference/MiniMax-H3-FL2VA-Turbo.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-FL2VA-Turbo.py)|-|-|-|-|
|[DiffSynth-Studio/MiniMax-H3-Text-Embeddings](https://www.modelscope.cn/models/DiffSynth-Studio/MiniMax-H3-Text-Embeddings)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference/MiniMax-H3-Text-Embeddings.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-Text-Embeddings.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/full/MiniMax-H3-Text-Embeddings.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/validate_full/MiniMax-H3-Text-Embeddings.py)|-|-|
|[PAI/MiniMax-H3-Fun-Controlnet-Union](https://www.modelscope.cn/models/PAI/MiniMax-H3-Fun-Controlnet-Union)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference/MiniMax-H3-Fun-Controlnet-Union.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-Fun-Controlnet-Union.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/full/MiniMax-H3-Fun-Controlnet-Union.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/validate_full/MiniMax-H3-Fun-Controlnet-Union.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/lora/MiniMax-H3-Fun-Controlnet-Union.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/validate_lora/MiniMax-H3-Fun-Controlnet-Union.py)|
|[OpenVDN/vdn-minimax-h3: VDN-H3 50 steps](https://www.modelscope.cn/models/OpenVDN/vdn-minimax-h3)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference/MiniMax-H3-VDN-FL2VA.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-VDN-FL2VA.py)|-|-|-|-|
|[OpenVDN/vdn-minimax-h3: VDN-H3 8 steps](https://www.modelscope.cn/models/OpenVDN/vdn-minimax-h3)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference/MiniMax-H3-VDN-Turbo-FL2VA.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-VDN-Turbo-FL2VA.py)|-|-|-|-|
|[OpenVDN/vdn-minimax-h3: VDN-H3 FP8](https://www.modelscope.cn/models/OpenVDN/vdn-minimax-h3)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference/MiniMax-H3-VDN-FP8-FL2VA.py)|-|-|-|-|-|
|[OpenVDN/vdn-minimax-h3: VDN-H3 8 steps FP8](https://www.modelscope.cn/models/OpenVDN/vdn-minimax-h3)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference/MiniMax-H3-VDN-Turbo-FP8-FL2VA.py)|-|-|-|-|-|

The model weights are split into two partitions: the `FL2VA` partition serves text-to-video-audio and keyframe-guided generation, while the `Ref2VA` partition serves reference-driven generation. The two partitions have different DiT and text encoder weights, so choose the `origin_file_pattern` of the matching partition for your task.

In addition, [PAI/MiniMax-H3-Fun-Controlnet-Union](https://www.modelscope.cn/models/PAI/MiniMax-H3-Fun-Controlnet-Union) provides ControlNet-based conditional control, which is used together with the base weights of the `FL2VA` partition. By passing a control video (canny / depth / hed / mlsd / pose modes) through `control_video`, you can generate video and audio that follow the structural conditions.

## Model Inference

The model is loaded via `MiniMaxH3Pipeline.from_pretrained`, see [Loading Models](../Pipeline_Usage/Model_Inference.md#loading-models) for details. Besides `model_configs`, the loading parameters include:

* `processor_config`: `ModelConfig` of the Qwen3-VL processor, used to tokenize the prompt and reference images. Defaults to `FL2VA/processor/`; set it explicitly to `Ref2VA/processor/` when using the `Ref2VA` partition.
* `vram_limit`: VRAM budget for VRAM management (in GB). Leave empty for no limit.

The input parameters for `MiniMaxH3Pipeline` inference include:

* `prompt`: Prompt describing the content of the video as well as the lines spoken by the characters.
* `negative_prompt`: Negative prompt, defaults to `" "`. This model is CFG-distilled, so it takes no effect by default.
* `height`: Height of the video, defaults to 768, must be a multiple of 32.
* `width`: Width of the video, defaults to 1344, must be a multiple of 32.
* `num_frames`: Number of frames, defaults to 124. It is snapped up to the nearest `17n+5`, so the returned clip may be slightly longer than requested. The frame rate is fixed at 24.
* `num_inference_steps`: Number of inference steps, defaults to 50.
* `seed`: Random seed, defaults to 42.
* `rand_device`: Device used to generate the random Gaussian noise tensor, defaults to `"cpu"`. When set to `cuda`, results differ across GPUs.
* `cfg_scale`: Classifier-free guidance scale, defaults to 1.0. This model is CFG-distilled, keeping the default value is recommended.
* `flow_shift`: Flow matching timestep shift for the video modality, defaults to 12.0.
* `audio_flow_shift`: Flow matching timestep shift for the audio modality, defaults to 3.0. Video and audio use two independent sigma schedules.
* `tiled`: Whether to enable tiled VAE inference, defaults to `True`. Enabling it significantly reduces VRAM usage during VAE encoding/decoding, at the cost of a slight numerical error and a small increase in inference time.
* `tile_size`: Tile size during VAE encoding/decoding, defaults to 256.
* `tile_overlap`: Tile overlap during VAE encoding/decoding, defaults to 64.
* `keyframes`: List of keyframe images for keyframe-guided generation. Images are resized onto the target canvas.
* `keyframe_indices`: Frame indices of the keyframes in the video, either `0` (first frame) or `-1` (last frame), corresponding one-to-one with `keyframes`.
* `references`: List of reference conditions in request order. Each element is a dict in one of the following four forms:
    * `{"type": "image", "image": PIL.Image}`
    * `{"type": "video", "video": list[PIL.Image]}` (silent video)
    * `{"type": "audio", "audio": Tensor[C, L], "sample_rate": int}`
    * `{"type": "video_audio", "video": list[PIL.Image], "audio": Tensor[C, L], "sample_rate": int}`

    Video frame lists must ALREADY be at 24fps; the pipeline never resamples the frame rate. A `video` is always treated as silent, so conditioning on a reference video's own soundtrack requires `video_audio` with the waveform passed explicitly — the pipeline receives frame lists rather than files and cannot probe for a soundtrack itself. `diffsynth.utils.data.audio_video.read_video_audio` reads the frames and the soundtrack out of one file with their durations already aligned:

    ```python
    from diffsynth.utils.data.audio_video import read_video_audio

    frames, waveform, sample_rate = read_video_audio(
        "video.mp4", height=480, width=832, num_frames=124, fps=24,
        audio_sample_rate=pipe.audio_vae.sample_rate,
    )
    ```
* `ref_image_short_edge`: Target short edge of a reference image, defaults to 2048. A reference image is rescaled onto that short edge with its aspect ratio preserved (upscaling allowed) and both axes rounded to the nearest multiple of 32. No area cap applies.
* `ref_video_short_edge`: Target short edge of a reference video, defaults to 768.
* `ref_video_max_pixels`: Soft area cap for a reference video, defaults to `768 * 1344`. A reference video is first scaled onto the short edge, then scaled back down proportionally if its area exceeds the cap, and finally both axes are rounded to a multiple of 32. Footage wider than 16:9 usually hits the cap.
* `control_video`: Control video frame list (PIL images) for ControlNet-based conditional generation. It supports canny / depth / hed / mlsd / pose control modes (determined by the content of the control video), and the generated result follows the visual structure of the control video. The frame list must ALREADY be at 24fps; if shorter than `num_frames`, the last frame is repeated to pad it out, and extra frames are truncated. To use it, the `PAI/MiniMax-H3-Fun-Controlnet-Union` ControlNet weights must be loaded in `model_configs`.
* `control_scale`: Strength of the ControlNet condition, defaults to 1.0. Increasing it tightens the constraint of the control video on the generated structure.
* `retake_video`: Source video frame list for video retake, which must already be at 24fps. Frames are resized onto the target canvas and truncated to `num_frames`. Because `num_frames` is first snapped up to the nearest `17n+5`, a source clip is often a few frames short (e.g. a 121-frame clip against an aligned 124); the tail is padded by repeating the last frame, and that padded tail is regenerated rather than frozen. Supply at least `num_frames` frames to avoid this.
* `frame_regions_to_retake`: Half-open **frame-id** ranges of `retake_video` to regenerate, counted from 0, e.g. `[(17, 51)]`. Everything outside them is preserved from the source. The 17 frames of a VAE clip are coupled in latent space, so retaking any frame of a clip retakes the whole clip: each range is widened outwards to clip boundaries. Pass multiples of 17 to get exactly the range you asked for. Omit it, or pass an empty list, to preserve the whole source video, which turns `retake_video` into video-driven audio generation.
* `retake_audio`: Source waveform `Tensor[C, L]` for audio retake. It is converted to stereo and resampled to the audio VAE's sample rate, then trimmed or padded to the video duration.
* `retake_audio_sample_rate`: Sample rate of `retake_audio`, defaults to 32000.
* `seconds_regions_to_retake`: Half-open ranges of `retake_audio` to regenerate, in **seconds**, e.g. `[(0, 1), (4, 5)]`. The audio VAE compresses uniformly at 40 latent frames per second and has no clip structure, so ranges are used as given, at a 1/40 s granularity. Omitting it preserves the whole source audio, which turns `retake_audio` into audio-driven video generation.

    Video and audio retake are independent: either one can be used alone, and note the two use different units -- frame ids for video, seconds for audio. `read_video_audio` is the convenient way to get a time-aligned `(frames, waveform, sample_rate)` triple out of a single file:

    ```python
    source_video, source_audio, audio_sample_rate = read_video_audio(
        "video.mp4", height=480, width=832, num_frames=124, fps=24,
        audio_sample_rate=pipe.audio_vae.sample_rate,
    )

    def align_to_clips(start, end, total_frames, clip_frames=17):
        """Widen the half-open frame range [start, end) to whole clips. Frames count from 0."""
        first_clip, last_clip = start // clip_frames, (end - 1) // clip_frames
        return first_clip * clip_frames, min((last_clip + 1) * clip_frames, total_frames)

    video, audio = pipe(
        prompt=prompt, height=480, width=832, num_frames=124,
        retake_video=source_video,
        frame_regions_to_retake=[align_to_clips(24, 48, 124)],   # frames [24,48) -> (17, 51)
        retake_audio=source_audio,
        retake_audio_sample_rate=audio_sample_rate,
        seconds_regions_to_retake=[(0, 1), (4, 5)],              # seconds
    )
    ```
* `progress_bar_cmd`: Progress bar, defaults to `tqdm`. Set it to `lambda x: x` to disable the progress bar.

The pipeline returns a `(video, audio)` tuple, where the video is a list of PIL images and the audio is a waveform tensor. Use `diffsynth.utils.data.audio_video.write_video_audio` to mux them into an MP4:

```python
write_video_audio(video=video, audio=audio, output_path="video.mp4", fps=24, audio_sample_rate=pipe.audio_vae.sample_rate)
```

If VRAM is insufficient, please enable [VRAM management](../Pipeline_Usage/VRAM_management.md). We provide a recommended low VRAM configuration for each model in the example code, see the table in "Model Overview" above. We also provide NF4-quantized weights to further reduce VRAM requirements; the corresponding scripts are listed in the same table. The int8-quantized weights released by Comfy-Org are supported as well, via the comfy-kitchen backend, which requires `pip install "diffsynth[quant]"`.

## Model Training

Models in the MiniMax-H3 series are trained uniformly via [`examples/minimax_h3/model_training/train.py`](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/train.py). The script parameters include:

* General Training Parameters
    * Dataset Configuration
        * `--dataset_base_path`: Root directory of the dataset.
        * `--dataset_metadata_path`: Path to the dataset metadata file.
        * `--dataset_repeat`: Number of dataset repeats per epoch.
        * `--dataset_num_workers`: Number of processes per DataLoader.
        * `--data_file_keys`: Field names to load from metadata, typically paths to image or video files, separated by `,`.
    * Model Loading Configuration
        * `--model_paths`: Paths to load models from, in JSON format.
        * `--model_id_with_origin_paths`: Model IDs with original paths, separated by commas.
        * `--extra_inputs`: Additional input parameters required by the model Pipeline, separated by `,`.
        * `--fp8_models`: Models to load in FP8 format, currently only supported for models whose parameters are not updated by gradients.
        * `--quant_options`: Dynamically quantize loaded models. Semicolon-separated entries, each `<model_string>:<method>[/<exclude_modules>]`, where `<model_string>` matches an entry in `--model_paths`/`--model_id_with_origin_paths`, `method` is a registered method (e.g. `bitsandbytes_nf4`), and `exclude_modules` optionally lists layers kept in full precision.
    * Basic Training Configuration
        * `--learning_rate`: Learning rate.
        * `--num_epochs`: Number of epochs.
        * `--trainable_models`: Trainable models, e.g., `dit`, `vae`, `text_encoder`.
        * `--find_unused_parameters`: Whether unused parameters exist in DDP training.
        * `--weight_decay`: Weight decay magnitude.
        * `--task`: Training task, defaults to `sft`.
    * Output Configuration
        * `--output_path`: Path to save the model.
        * `--remove_prefix_in_ckpt`: Remove prefix in the model's state dict.
        * `--save_steps`: Interval in training steps to save the model.
        * `--disable_model_save`: Disable step, epoch, and end-of-training model saves, including synchronization and weight gathering for saving. Disabled by default; takes precedence over `--save_steps`. Logs, profiler traces, and preprocessing caches still follow their existing settings.
    * LoRA Configuration
        * `--lora_base_model`: Which model to add LoRA to.
        * `--lora_target_modules`: Which layers to add LoRA to.
        * `--lora_rank`: Rank of LoRA.
        * `--lora_checkpoint`: Path to LoRA checkpoint.
        * `--preset_lora_path`: Path to preset LoRA checkpoint for LoRA differential training.
        * `--preset_lora_model`: Which model to integrate preset LoRA into, e.g., `dit`.
    * Gradient Configuration
        * `--use_gradient_checkpointing`: Whether to enable gradient checkpointing.
        * `--use_gradient_checkpointing_offload`: Whether to offload gradient checkpointing to CPU memory.
        * `--gradient_accumulation_steps`: Number of gradient accumulation steps.
    * Resolution Configuration
        * `--height`: Height of the video. Leave `height` and `width` empty to enable dynamic resolution.
        * `--width`: Width of the video. Leave `height` and `width` empty to enable dynamic resolution.
        * `--max_pixels`: Maximum pixel area, images larger than this will be scaled down during dynamic resolution.
        * `--num_frames`: Number of frames for the video.
* MiniMax-H3 Specific Parameters
    * `--processor_path`: Path of the Qwen3-VL processor, supports the `model_id:origin_file_pattern` form, used to tokenize the prompt.
    * `--initialize_model_on_cpu`: Whether to initialize models on CPU.
    * `--silent_on_missing_audio`: Whether to use silent audio as a fallback when no audio track is present in the video data.
    * `--training_cfg_scale`: Inverse-CFG scale for preserving MiniMax-H3 guidance distillation during fine-tuning. Values greater than 1 enable a no-grad unconditional branch. Set `MINIMAX_H3_ZERO_TARGET_LOSS=0` for flow-matching training.
    * `--audio_loss_weight`: Audio weight for flow-matching loss, used only with `MINIMAX_H3_ZERO_TARGET_LOSS=0`. A value of 0 excludes audio loss while still noising and forwarding the audio stream.

We provide an example dataset for testing, which can be downloaded with the following command:

```shell
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --local_dir ./data/diffsynth_example_dataset
```

The LoRA training scripts use a two-stage workflow: first preprocess and cache the dataset with `--task "sft:data_process"`, then run the actual training with `--task "sft:train"`. The split is required because the DiT and the Qwen3-VL text encoder cannot be resident on one GPU at the same time. LoRA is applied to the `qkv_proj,out_proj` modules of the DiT by default, with a rank of 32. Full training follows the same two-stage workflow; its second stage enables DeepSpeed ZeRO-3 through [`accelerate_config_zero3.yaml`](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/full/accelerate_config_zero3.yaml) and selects the trained model with `--trainable_models "dit"`.

LoRA training on the NF4 quantized weights is a single-stage workflow: once quantized, every component fits on one GPU, so no dataset cache is needed. In this case `--lora_target_modules` must be given explicitly, because quantized weights are stored packed in the state dict and the automatic search cannot recognise their shape.

Keyframe-guided (FL2VA) training appends `input_image,end_image` to `--extra_inputs`, taking the first and last frame of the training video as conditions respectively, so the dataset needs no extra column. Reference-driven (Ref2VA) training uses `metadata.json`, whose `references` field is a list of reference blocks supporting four types -- `image`, `video`, `audio` and `video_audio`:

```json
[
  {
    "video": "train_video.mp4",
    "prompt": "...",
    "input_audio": "train_video.mp4",
    "references": [
      {"type": "image", "image": "0.png"}
    ],
    "frame_rate": 24
  }
]
```

`references` must appear in both `--data_file_keys` and `--extra_inputs`: the former loads the files according to their type, the latter injects the reference blocks into the pipeline. Reference images are handed to the pipeline at native resolution (it rescales them by its own reference short edge internally), while reference videos are cropped to the training canvas and sampled at 24fps.

ControlNet (control-video-driven) training also builds on `metadata.json`: each data entry provides a `control_video` field (path to the control video), and `control_video` is appended to both `--data_file_keys` and `--extra_inputs`. Full training trains the ControlNet with `--trainable_models "controlnet"` while the DiT stays frozen, and saves the weights with `--remove_prefix_in_ckpt "pipe.controlnet."`; LoRA training keeps the ControlNet frozen and applies LoRA to the DiT instead, so the model adapts to new data distributions while retaining its control capability.

We provide recommended training scripts for each model, please refer to the table in "Model Overview" above. For guidance on writing model training scripts, see [Model Training](../Pipeline_Usage/Model_Training.md); for more advanced training algorithms, see [Training Framework Overview](https://github.com/modelscope/DiffSynth-Studio/tree/main/docs/en/Training/).
