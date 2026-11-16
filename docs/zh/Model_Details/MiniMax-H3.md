# MiniMax-H3

MiniMax H3 是一个通用的全模态生成系统。它支持对由文本、图像、视频和音频组成的多模态上下文进行统一理解，并能生成最高达 2K 分辨率、最长 15 秒、包含原生立体声音频的视频。得益于其面向任务泛化的系统设计，H3 在预训练阶段就已具备广泛的多模态上下文理解与生成能力，从而在遵循复杂多模态指令方面表现出色。

## 安装

在使用本项目进行模型推理和训练前，请先安装 DiffSynth-Studio。

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

更多关于安装的信息，请参考[安装依赖](../Pipeline_Usage/Setup.md)。

## FSDP2 训练选项

H3 训练脚本支持以下 `--fsdp2_*` 选项，需使用 FSDP2 的 Accelerate 启动配置。未指定时保留当前默认行为；布尔值可写 `true/false` 或 `1/0`，也可省略值表示 `true`。

|选项|默认行为|作用|
|-|-|-|
|`--fsdp2_sync_each_batch`|`false`|每个 microstep 都同步梯度，同时仍按 `--gradient_accumulation_steps` 累积并更新参数；开启可避免累积期间保留完整梯度，但增加通信|
|`--fsdp2_reshard_after_backward`|`true`|每次 backward 后重新分片参数；设为 `false` 时在累积中间步保留参数，减少下一次 forward 的 all-gather，但增加显存；更新参数前的最后一步始终重新分片|
|`--fsdp2_param_dtype`|`auto`|未分片参数的计算精度：`bfloat16`、`float16`、`float32`；`auto` 跟随启动配置。显式指定时也同步设置 Accelerate 的精度模式、H3 权重加载及输入 dtype；FP16 使用对应的梯度缩放，FP32 使用 `mixed_precision=no`|
|`--fsdp2_reduce_dtype`|`auto`|梯度归约精度：`bfloat16`、`float16`、`float32`；`auto` 使用计算精度|
|`--fsdp2_output_dtype`|`none`|FSDP 模块输出精度：`none`、`bfloat16`、`float16`、`float32`；`none` 不转换输出。此策略也作用于根训练模块返回的 loss|
|`--fsdp2_cast_forward_inputs`|`false`|是否将 FSDP 的浮点输入转换到计算精度。H3 默认关闭以保留 FP64 位置和 FP32 RoPE 角度；开启会改变这些张量的精度。与输出精度独立配置|
|`--fsdp2_forward_prefetch`|`0`|显式预取后续几个 DiT block；`0` 不设置显式 forward 预取|
|`--fsdp2_backward_prefetch`|`0`|显式预取前面几个 DiT block；`0` 保留 PyTorch 自动 backward 预取，并非禁用预取|
|`--fsdp2_dp_replicate_size`|不覆盖启动配置|HSDP 复制组大小；只指定这一项时，从 world size 推导分片组大小|
|`--fsdp2_dp_shard_size`|不覆盖启动配置|FSDP/HSDP 分片组大小；只指定这一项时，从 world size 推导复制组大小|
|`--fsdp2_hsdp_defer_all_reduce`|`false`|每个 microstep 保留 reduce-scatter，只把复制组 all-reduce 延迟到 optimizer 更新前；要求 HSDP 且 `fsdp2_sync_each_batch=true`，不满足时直接报错|

`param_dtype` 控制计算参数，Accelerate 混合精度下的 FP32 参数分片/optimizer 存储不由这个选项直接改变。使用全 FP32 计算时，应选择支持 FP32 的 attention 后端，例如设置 `DIFFSYNTH_ATTENTION_IMPLEMENTATION=torch`，因为 FlashAttention 仅支持 FP16/BF16。参数重新分片与梯度同步是独立控制项。预取深度越大，可能同时保留更多完整参数。HSDP 要求复制组大小与分片组大小的乘积等于总进程数；复制组大于 1 时，分片组也必须大于 1。默认每次 HSDP 梯度同步包括 reduce-scatter 和复制组 all-reduce；开启延迟选项后，两级通信分别按 microstep 和 optimizer 更新边界执行，尾部不足一个累积窗口时也会完成 all-reduce。

输出与计算 dtype 不同时，只在进入后续 block 或输出头前将 hidden 激活转回计算 dtype，不要求转换位置或 RoPE 角度。显式选择不同输出 dtype 仍会产生输出转换及必要的激活转回；同精度、同策略的 H3 block 通常保留 `output_dtype=none` 即可。`reshard_after_backward=false` 只在累积窗口首尾切换状态，训练正常结束或异常退出时恢复默认状态。

默认归约精度仍跟随计算精度，预取仍是两个独立的层数配置。HSDP 中同时设置 `fsdp2_sync_each_batch=true` 和 `fsdp2_hsdp_defer_all_reduce=true`，可采用与 VeOmni 相同的“每步 reduce-scatter、窗口末尾 all-reduce”调度；关闭延迟选项则保留原有同步行为。两个选项均不会改变梯度累积次数、梯度裁剪或 optimizer/scheduler 更新边界。同步频率、分组或精度变化可能改变浮点舍入，即使数学上的梯度累积目标相同，也不保证逐位一致。

例如，复用现有缓存，以 4 个 microstep 累积梯度，同时每步同步梯度并显式预取一层：

```shell
STAGE=train bash examples/minimax_h3/model_training/full/MiniMax-H3-FL2VA-FSDP2.sh \
  --gradient_accumulation_steps 4 \
  --fsdp2_sync_each_batch true \
  --fsdp2_reshard_after_backward true \
  --fsdp2_reduce_dtype float32 \
  --fsdp2_forward_prefetch 1 \
  --fsdp2_backward_prefetch 1
```

8 个进程下，追加 `--fsdp2_dp_replicate_size 2 --fsdp2_dp_shard_size 4 --fsdp2_hsdp_defer_all_reduce true` 可选择 2×4 HSDP 并采用上述延迟调度。去掉最后一个选项则仍然每步同步两个组。这些选项只在训练阶段传给上述脚本；数据预处理阶段使用独立的单进程配置。

## MUSA 算子优化

在启动普通 MiniMax-H3 训练之前，可通过两个独立环境变量开启 MUSA 融合算子，默认均关闭：

```shell
export MINIMAX_H3_MUSA_FUSED_ROPE=1
export MINIMAX_H3_MUSA_FUSED_SWIGLU=1
```

`MINIMAX_H3_MUSA_FUSED_ROPE` 使用 `torch.rope` 替换 DiT attention 的 half-split partial RoPE，保留未旋转的维度（默认旋转 head 的前 96/128 维）。kernel 直接接收共享的 FP32 角度表，内部计算 cos/sin，Python 侧不会额外预计算 cos/sin。关闭这个开关时，每次 forward 预计算一次 cos/sin，供各层 Q/K 和梯度 checkpoint 重计算复用。

`MINIMAX_H3_MUSA_FUSED_SWIGLU` 使用 `aten.swish_glu` 及其已注册的融合反向替换 `silu(gate) * up`，覆盖 DiT 和 token refiner 的 MLP。两个开关均支持 `1/0`、`true/false`、`yes/no`、`on/off`。无论是否开启融合算子，DiT 和 token refiner 各自的 `cu_seqlens` 都只在每次 forward 转换一次主机端边界，供所有层复用。

`examples/minimax_h3/model_training/train.py` 在加载 pipeline 后调用 `apply_minimax_h3_musa_patch(self.pipe.dit)`，仅对普通 H3 DiT 启用这四项优化，不修改权重或 checkpoint 键名。先在 CPU 初始化再搬到 MUSA 也能生效；执行时会检查 tensor 的设备和 dtype，非 MUSA 输入使用原算子。单独加载推理 pipeline 不会安装补丁。Comfy、VDN 和 ControlNet 不在此次优化范围内。手动构建普通 H3 训练模型时，可在分布式包装前调用 `from diffsynth.core.musa_patch import apply_minimax_h3_musa_patch` 后执行 `apply_minimax_h3_musa_patch(model)`。

融合算子需要当前 torch-musa 提供相应 API。它们与原有 BF16 逐算子计算存在舍入差异，进行训练效果或性能对比时请保持其他配置一致。可以结合 `PROFILE_ENABLE=1` 检查实际 kernel 和耗时。

## 快速开始

运行以下代码可以快速加载 [DiffSynth-Studio/MiniMax-H3-NF4](https://www.modelscope.cn/models/DiffSynth-Studio/MiniMax-H3-NF4) NF4 量化模型并进行文生音视频推理。显存管理已启动，框架会自动根据剩余显存控制模型参数的加载，最低 7G 显存即可运行。

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

## 模型总览

|模型 ID|推理|低显存推理|全量训练|全量训练后验证|LoRA 训练|LoRA 训练后验证|
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
|[OpenVDN/vdn-minimax-h3: VDN-H3 50 步](https://www.modelscope.cn/models/OpenVDN/vdn-minimax-h3)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference/MiniMax-H3-VDN-FL2VA.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-VDN-FL2VA.py)|-|-|-|-|
|[OpenVDN/vdn-minimax-h3: VDN-H3 8 步](https://www.modelscope.cn/models/OpenVDN/vdn-minimax-h3)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference/MiniMax-H3-VDN-Turbo-FL2VA.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-VDN-Turbo-FL2VA.py)|-|-|-|-|
|[OpenVDN/vdn-minimax-h3: VDN-H3 FP8](https://www.modelscope.cn/models/OpenVDN/vdn-minimax-h3)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference/MiniMax-H3-VDN-FP8-FL2VA.py)|-|-|-|-|-|
|[OpenVDN/vdn-minimax-h3: VDN-H3 8 步 FP8](https://www.modelscope.cn/models/OpenVDN/vdn-minimax-h3)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference/MiniMax-H3-VDN-Turbo-FP8-FL2VA.py)|-|-|-|-|-|

模型权重分为两个分区：`FL2VA` 分区服务文生音视频与首尾帧引导生成，`Ref2VA` 分区服务参考驱动生成，两者的 DiT 与文本编码器权重不同，需按任务选择对应分区的 `origin_file_pattern`。

此外，[PAI/MiniMax-H3-Fun-Controlnet-Union](https://www.modelscope.cn/models/PAI/MiniMax-H3-Fun-Controlnet-Union) 提供 ControlNet 条件控制能力，需配合 `FL2VA` 分区的基础权重使用。通过 `control_video` 传入控制视频（canny / depth / hed / mlsd / pose 等控制模式），即可生成与结构条件一致的视频与音频。

## 模型推理

模型通过 `MiniMaxH3Pipeline.from_pretrained` 加载，详见[加载模型](../Pipeline_Usage/Model_Inference.md#加载模型)。加载时除 `model_configs` 外还包括：

* `processor_config`: Qwen3-VL processor 的 `ModelConfig`，用于对提示词及参考图像进行 tokenize，默认指向 `FL2VA/processor/`。使用 `Ref2VA` 分区时需显式改为 `Ref2VA/processor/`。
* `vram_limit`: 显存管理的显存上限（单位 GB），留空则不限制。

`MiniMaxH3Pipeline` 推理的输入参数包括：

* `prompt`: 提示词，描述视频中出现的内容以及人物说出的台词。
* `negative_prompt`: 负向提示词，默认值为 `" "`。该模型为 CFG 蒸馏模型，默认不生效。
* `height`: 视频高度，默认值为 768，需保证为 32 的倍数。
* `width`: 视频宽度，默认值为 1344，需保证为 32 的倍数。
* `num_frames`: 视频帧数，默认值为 124，会被向上对齐到最近的 `17n+5`，因此实际输出可能略长于请求值。视频帧率固定为 24。
* `num_inference_steps`: 推理步数，默认值为 50。
* `seed`: 随机种子，默认值为 42。
* `rand_device`: 生成随机高斯噪声矩阵的计算设备，默认为 `"cpu"`。当设置为 `cuda` 时，在不同 GPU 上会导致不同的生成结果。
* `cfg_scale`: Classifier-free guidance 的参数，默认值为 1.0。该模型为 CFG 蒸馏模型，建议保持默认值。
* `flow_shift`: 视频模态的 flow matching 时间步 shift，默认值为 12.0。
* `audio_flow_shift`: 音频模态的 flow matching 时间步 shift，默认值为 3.0。视频与音频使用两条独立的 sigma 调度。
* `tiled`: 是否启用 VAE 分块推理，默认为 `True`。设置为 `True` 时可显著减少 VAE 编解码阶段的显存占用，会产生少许误差，以及少量推理时间延长。
* `tile_size`: VAE 编解码阶段的分块大小，默认为 256。
* `tile_overlap`: VAE 编解码阶段的分块重叠大小，默认为 64。
* `keyframes`: 关键帧图像列表，用于首尾帧引导生成，图像会被缩放到目标画幅。
* `keyframe_indices`: 关键帧在视频中的帧索引列表，取值为 `0`（首帧）或 `-1`（尾帧），与 `keyframes` 一一对应。
* `references`: 参考条件列表，按请求顺序给出，每个元素为字典，支持以下四种形式：
    * `{"type": "image", "image": PIL.Image}`
    * `{"type": "video", "video": list[PIL.Image]}`（无声视频）
    * `{"type": "audio", "audio": Tensor[C, L], "sample_rate": int}`
    * `{"type": "video_audio", "video": list[PIL.Image], "audio": Tensor[C, L], "sample_rate": int}`

    其中传入的视频帧列表必须已经是 24fps，Pipeline 不会重采样帧率。`video` 一律按无声处理，因此若要把参考视频自带的声轨也作为音频条件，必须使用 `video_audio` 并显式传入波形 —— Pipeline 接收的是帧列表而非文件，无法自行探测声轨。可用 `diffsynth.utils.data.audio_video.read_video_audio` 从同一个文件同时读出画面与声轨，两者的时长会自动对齐：

    ```python
    from diffsynth.utils.data.audio_video import read_video_audio

    frames, waveform, sample_rate = read_video_audio(
        "video.mp4", height=480, width=832, num_frames=124, fps=24,
        audio_sample_rate=pipe.audio_vae.sample_rate,
    )
    ```
* `ref_image_short_edge`: 参考图像的短边目标长度，默认值为 2048。参考图像保持长宽比缩放至该短边（允许放大），两轴各自向最近的 32 倍数取整，不受面积上限约束。
* `ref_video_short_edge`: 参考视频的短边目标长度，默认值为 768。
* `ref_video_max_pixels`: 参考视频的面积软上限，默认值为 `768 * 1344`。参考视频先按短边定标，若面积超过该上限则等比缩回，最后两轴各自取整到 32 的倍数。宽于 16:9 的素材通常会触发该上限。
* `control_video`: 控制视频帧列表（PIL 图像），用于 ControlNet 条件控制生成，支持 canny / depth / hed / mlsd / pose 等控制模式（由控制视频的内容决定），生成结果会跟随控制视频的画面结构。帧列表须已是 24fps，不足 `num_frames` 帧时会重复最后一帧补齐，多出的部分会被截断。使用该参数前，需在 `model_configs` 中加载 `PAI/MiniMax-H3-Fun-Controlnet-Union` 的 ControlNet 权重。
* `control_scale`: ControlNet 控制强度，默认值为 1.0。增大该值会加强控制视频对生成结构的约束。
* `retake_video`: 视频重绘（retake）的源视频帧列表，必须已经是 24fps。帧会被缩放到目标画幅并截取前 `num_frames` 帧。由于 `num_frames` 会先向上对齐到最近的 `17n+5`，源视频常常会差几帧（例如 121 帧的素材对应对齐后的 124 帧），此时会重复最后一帧补齐，且补出的尾部会被重新生成而非冻结。源素材帧数不少于 `num_frames` 即可避免。
* `frame_regions_to_retake`: `retake_video` 中需要重新生成的**帧号**区间，左闭右开、帧从 0 计数，例如 `[(17, 51)]`。区间之外的内容会从源视频原样保留。一个 VAE clip 的 17 帧在隐空间是耦合的，重绘 clip 内任何一帧就等于重绘整个 clip，因此区间会向外扩展到 clip 边界；传 17 的倍数即可得到与请求完全一致的范围。不传该参数（或传入空列表）时整段源视频都会被保留，此时 `retake_video` 相当于视频驱动的音频生成。
* `retake_audio`: 音频重绘（retake）的源波形 `Tensor[C, L]`。会被转为立体声并重采样到音频 VAE 的采样率，再按视频时长截断或补齐。
* `retake_audio_sample_rate`: `retake_audio` 的采样率，默认值为 32000。
* `seconds_regions_to_retake`: `retake_audio` 中需要重新生成的时间区间，单位为**秒**、左闭右开，例如 `[(0, 1), (4, 5)]`。音频 VAE 是均匀压缩（每秒 40 个隐空间帧）、没有 clip 结构，因此区间按给定值直接使用，时间分辨率为 1/40 秒。不传该参数时整段源音频都会被保留，此时 `retake_audio` 相当于音频驱动的视频生成。

    视频与音频的 retake 相互独立：可以只用其中之一，注意两者单位不同 —— 视频用帧号，音频用秒。推荐用 `read_video_audio` 从同一个文件中读出时长已对齐的 `(帧列表, 波形, 采样率)`：

    ```python
    source_video, source_audio, audio_sample_rate = read_video_audio(
        "video.mp4", height=480, width=832, num_frames=124, fps=24,
        audio_sample_rate=pipe.audio_vae.sample_rate,
    )

    def align_to_clips(start, end, total_frames, clip_frames=17):
        """把左闭右开的帧区间 [start, end) 扩展到完整 clip，帧从 0 计数。"""
        first_clip, last_clip = start // clip_frames, (end - 1) // clip_frames
        return first_clip * clip_frames, min((last_clip + 1) * clip_frames, total_frames)

    video, audio = pipe(
        prompt=prompt, height=480, width=832, num_frames=124,
        retake_video=source_video,
        frame_regions_to_retake=[align_to_clips(24, 48, 124)],   # 帧 [24,48) -> (17, 51)
        retake_audio=source_audio,
        retake_audio_sample_rate=audio_sample_rate,
        seconds_regions_to_retake=[(0, 1), (4, 5)],              # 秒
    )
    ```
* `progress_bar_cmd`: 进度条，默认为 `tqdm`。可通过设置为 `lambda x: x` 来屏蔽进度条。

Pipeline 返回 `(video, audio)` 二元组，视频为 PIL 图像列表，音频为波形张量，可通过 `diffsynth.utils.data.audio_video.write_video_audio` 混流写出 MP4：

```python
write_video_audio(video=video, audio=audio, output_path="video.mp4", fps=24, audio_sample_rate=pipe.audio_vae.sample_rate)
```

如果显存不足，请开启[显存管理](../Pipeline_Usage/VRAM_management.md)，我们在示例代码中提供了每个模型推荐的低显存配置，详见前文“模型总览”中的表格。此外我们还提供了 NF4 量化版本的权重，可进一步降低显存需求，对应脚本同样见“模型总览”表格。我们同时支持 Comfy-Org 发布的 int8 量化权重，通过 comfy-kitchen 后端加载，需要安装 `pip install "diffsynth[quant]"`。

## 模型训练

MiniMax-H3 系列模型统一通过 [`examples/minimax_h3/model_training/train.py`](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/train.py) 进行训练，脚本的参数包括：

* 通用训练参数
    * 数据集基础配置
        * `--dataset_base_path`: 数据集的根目录。
        * `--dataset_metadata_path`: 数据集的元数据文件路径。
        * `--dataset_repeat`: 每个 epoch 中数据集重复的次数。
        * `--dataset_num_workers`: 每个 Dataloader 的进程数量。
        * `--data_file_keys`: 元数据中需要加载的字段名称，通常是图像或视频文件的路径，以 `,` 分隔。
    * 模型加载配置
        * `--model_paths`: 要加载的模型路径。JSON 格式。
        * `--model_id_with_origin_paths`: 带原始路径的模型 ID。用逗号分隔。
        * `--extra_inputs`: 模型 Pipeline 所需的额外输入参数，以 `,` 分隔。
        * `--fp8_models`: 以 FP8 格式加载的模型，目前仅支持参数不被梯度更新的模型。
        * `--quant_options`：对加载的模型进行动态量化。以 `;` 分隔多个条目，每个为 `<模型字符串>:<method>[/<exclude_modules>]`，`<模型字符串>` 需与 `--model_paths`/`--model_id_with_origin_paths` 中的一致，`method` 为已注册的量化方法（如 `bitsandbytes_nf4`），`exclude_modules` 为可选的保持全精度的层。
    * 训练基础配置
        * `--learning_rate`: 学习率。
        * `--num_epochs`: 轮数（Epoch）。
        * `--trainable_models`: 可训练的模型，例如 `dit`、`vae`、`text_encoder`。
        * `--find_unused_parameters`: DDP 训练中是否存在未使用的参数。
        * `--weight_decay`: 权重衰减大小。
        * `--task`: 训练任务，默认为 `sft`。
    * 输出配置
        * `--output_path`: 模型保存路径。
        * `--remove_prefix_in_ckpt`: 在模型文件的 state dict 中移除前缀。
        * `--save_steps`: 保存模型的训练步数间隔。
    * LoRA 配置
        * `--lora_base_model`: LoRA 添加到哪个模型上。
        * `--lora_target_modules`: LoRA 添加到哪些层上。
        * `--lora_rank`: LoRA 的秩（Rank）。
        * `--lora_checkpoint`: LoRA 检查点的路径。
        * `--preset_lora_path`: 预置 LoRA 检查点路径，用于 LoRA 差分训练。
        * `--preset_lora_model`: 预置 LoRA 融入的模型，例如 `dit`。
    * 梯度配置
        * `--use_gradient_checkpointing`: 是否启用 gradient checkpointing。
        * `--use_gradient_checkpointing_offload`: 是否将 gradient checkpointing 卸载到内存中。
        * `--gradient_accumulation_steps`: 梯度累积步数。
    * 分辨率配置
        * `--height`: 视频的高度。将 `height` 和 `width` 留空以启用动态分辨率。
        * `--width`: 视频的宽度。将 `height` 和 `width` 留空以启用动态分辨率。
        * `--max_pixels`: 最大像素面积，动态分辨率时大于此值的图片会被缩小。
        * `--num_frames`: 视频的帧数。
* MiniMax-H3 专有参数
    * `--processor_path`: Qwen3-VL processor 的路径，支持 `model_id:origin_file_pattern` 形式，用于对 prompt 进行 tokenize。
    * `--initialize_model_on_cpu`: 是否在 CPU 上初始化模型。
    * `--silent_on_missing_audio`: 视频数据不含音轨时，是否以静音音频作为兜底。
    * `--training_cfg_scale`: 微调时用于保留 MiniMax-H3 指导蒸馏的逆 CFG 系数。大于 1 时启用一路无梯度的无条件分支，等于 1 时保持标准 flow matching 损失。
    * `--audio_loss_weight`: MiniMax-H3 损失中音频项的权重。为 1 时视频与音频等权重，为 0 时只用视频项训练，音频流仍会加噪并送入模型。

我们构建了一个样例数据集，以方便您进行测试，通过以下命令可以下载这个数据集：

```shell
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --local_dir ./data/diffsynth_example_dataset
```

LoRA 训练脚本采用两阶段流程：先以 `--task "sft:data_process"` 预处理并缓存数据集，再以 `--task "sft:train"` 执行正式训练。之所以必须分阶段，是因为 DiT 与 Qwen3-VL 文本编码器无法同时载入单卡。LoRA 默认作用于 DiT 的 `qkv_proj,out_proj` 模块，rank 为 32。全量训练同样采用两阶段流程，第二阶段通过 [`accelerate_config_zero3.yaml`](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/full/accelerate_config_zero3.yaml) 启用 DeepSpeed ZeRO-3，并以 `--trainable_models "dit"` 指定训练对象。

NF4 量化版本的 LoRA 训练为单阶段流程：量化后所有组件可同时载入单卡，无需预先缓存数据集。此时必须显式指定 `--lora_target_modules`，因为量化权重在 state dict 中以打包形式存储，自动探测无法识别其形状。

首尾帧引导（FL2VA）训练在 `--extra_inputs` 中追加 `input_image,end_image`，分别取训练视频的首帧与尾帧作为条件，数据集无需额外列。参考驱动（Ref2VA）训练使用 `metadata.json`，其中 `references` 字段为一组参考块，支持 `image`、`video`、`audio`、`video_audio` 四种类型：

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

`references` 需同时出现在 `--data_file_keys` 与 `--extra_inputs` 中，前者负责按类型加载文件，后者负责将参考块注入 Pipeline。参考图像以原生分辨率交给 Pipeline（其内部会按参考短边重新缩放），参考视频则裁剪到训练画布并按 24fps 采样。

ControlNet（控制视频驱动）训练同样基于 `metadata.json`，在数据条目中提供 `control_video` 字段（控制视频路径），并将 `control_video` 追加到 `--data_file_keys` 与 `--extra_inputs` 中。全量训练以 `--trainable_models "controlnet"` 训练 ControlNet 网络（DiT 冻结），并以 `--remove_prefix_in_ckpt "pipe.controlnet."` 保存权重；LoRA 训练则保持 ControlNet 冻结，对 DiT 施加 LoRA，从而让模型在保持控制能力的同时适配新的数据分布。

我们为每个模型编写了推荐的训练脚本，请参考前文“模型总览”中的表格。关于如何编写模型训练脚本，请参考[模型训练](../Pipeline_Usage/Model_Training.md)；更多高阶训练算法，请参考[训练框架详解](https://github.com/modelscope/DiffSynth-Studio/tree/main/docs/zh/Training/)。
