#!/usr/bin/env bash
# MiniMax-H3 Ref2VA：多机 FSDP2 全参数训练
# 直接使用 perf/profiling 分支原生 train.py；不安装包，不使用训练 wrapper。
#
# 四台机器使用同一个脚本，仅 NODE_RANK 分别为 0、1、2、3。
# 示例（在容器内执行；执行此脚本就会启动训练）：
#   MODEL_DIR=/path/to/MiniMax-H3 MASTER_ADDR=192.0.2.1 NODE_RANK=0 bash ./start_ref2va_multinode.sh
#   MODEL_DIR=/path/to/MiniMax-H3 MASTER_ADDR=192.0.2.1 NODE_RANK=1 bash ./start_ref2va_multinode.sh
#
# 示例路径和地址为占位值，请通过环境变量传入实际配置。
# MODEL_DIR、REPO_DIR、OUTPUT_DIR 使用绝对路径；使用合成输入，不读取数据缓存。
set -euo pipefail

# ---------- 1. 代码、解释器与数据路径 ----------
REPO_DIR="${REPO_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# MODEL_DIR 下应有 Ref2VA/transformer/；合成输入不需要 processor。
MODEL_DIR="${MODEL_DIR:?请设置 MODEL_DIR 为模型目录的绝对路径}"

# 四台填写同一个运行名称。若输出盘不共享，分别保存在各机同名目录。
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/runs/ref2va-fsdp2}"

# ---------- 2. 多机参数 ----------
NNODES="${NNODES:-4}"                  # 机器数
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"    # 每台 GPU 数
NODE_RANK="${NODE_RANK:-0}"            # 本机编号：0 .. NNODES-1
MASTER_ADDR="${MASTER_ADDR:?请设置 MASTER_ADDR 为所有节点可访问的 rank0 地址}"
MASTER_PORT="${MASTER_PORT:-29831}"    # 四台相同，使用未占用端口
WORLD_SIZE=$((NNODES * GPUS_PER_NODE))

# ---------- 3. 训练参数 ----------
GLOBAL_BATCH_SIZE=64
# 仓库当前 DataLoader 的 microbatch 固定为 1。
MICRO_BATCH_SIZE=1
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / WORLD_SIZE / MICRO_BATCH_SIZE))
TRAIN_STEPS=30
# 合成样本总次数：1920；4机32卡时每rank处理60次，累积2次更新一次。
DATASET_REPEAT=$((TRAIN_STEPS * GLOBAL_BATCH_SIZE))
HEIGHT=720
WIDTH=1280
NUM_FRAMES=240  # 24fps、10秒；合成输入不经过 VAE，不要求 17n+5。
# 沿用 zip 的 floor 网格：60 * (720//32) * (1280//32) = 52800 视频 token。
LEARNING_RATE=1e-5
WEIGHT_DECAY=0.01
LR_WARMUP_STEPS=0
MAX_GRAD_NORM=0
OPTIMIZER="torch_musa.optim.FusedAdamW"

# 仅校验参数关系，不检查或更改机器环境。
if (( GRAD_ACCUM_STEPS < 1 || WORLD_SIZE * MICRO_BATCH_SIZE * GRAD_ACCUM_STEPS != GLOBAL_BATCH_SIZE )); then
    echo "GLOBAL_BATCH_SIZE 必须是 WORLD_SIZE * MICRO_BATCH_SIZE 的正整数倍。" >&2
    exit 1
fi
if (( NODE_RANK < 0 || NODE_RANK >= NNODES )); then
    echo "NODE_RANK 必须在 0 .. NNODES-1 范围内。" >&2
    exit 1
fi

# ---------- 4. 分支原生 profiler ----------
# 默认为关闭；需要采集时在命令前加 PROFILE_ENABLE=1。
# 保持分支默认窗口：记录第3、4、5个 microstep，不是 optimizer step。
export PROFILE_ENABLE="${PROFILE_ENABLE:-0}"
export PROFILE_START_STEP=3
export PROFILE_END_STEP=6
export PROFILE_RANK0_ONLY=true
export PROFILE_RECORD_SHAPES=true
export PROFILE_MEMORY=false
export PROFILE_STACK=true
export PROFILE_TRACE_DIR="$OUTPUT_DIR/profiler"

# ---------- 5. 明确列出的运行设置 ----------
export PYTHONPATH="$REPO_DIR"
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export DIFFSYNTH_SKIP_DOWNLOAD=True
export PYTORCH_MUSA_ALLOC_CONF=expandable_segments:True

# 与此前四机一致的多机通信兼容设置。
# 若网卡名不同，启动前显式传 NET_IFACE；不自动探测或切换。
export TORCH_MUSA_FSDP2_ENABLE_CE_COMM=0
export MCCL_SOCKET_IFNAME="${NET_IFACE:-bond0}"
export GLOO_SOCKET_IFNAME="${NET_IFACE:-bond0}"
export MCCL_DEBUG=WARN
export TORCH_DISTRIBUTED_DEBUG=OFF

# 融合算子默认关闭，可通过环境变量启用；不启用 torch.compile。
export MINIMAX_H3_MUSA_FUSED_ROPE="${MINIMAX_H3_MUSA_FUSED_ROPE:-0}"
export MINIMAX_H3_MUSA_FUSED_SWIGLU="${MINIMAX_H3_MUSA_FUSED_SWIGLU:-0}"

# 本脚本默认使用 flow-matching loss，并关闭指定模块 FP32；均可通过环境变量覆盖。
export MINIMAX_H3_ZERO_TARGET_LOSS="${MINIMAX_H3_ZERO_TARGET_LOSS:-0}"
export MINIMAX_H3_FP32_MODULES="${MINIMAX_H3_FP32_MODULES:-0}"

# ---------- 6. FSDP2 配置 ----------
# YAML 在用户真正执行启动脚本时生成；每台独立文件名，避免并发覆盖。
mkdir -p "$OUTPUT_DIR"
FSDP_CONFIG="$OUTPUT_DIR/accelerate_fsdp2_node${NODE_RANK}.yaml"
cat > "$FSDP_CONFIG" <<YAML
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: FSDP
downcast_bf16: 'no'
enable_cpu_affinity: false
fsdp_config:
  fsdp_version: 2
  fsdp_transformer_layer_cls_to_wrap: MiniMaxH3DiTBlock
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_reshard_after_forward: true
  fsdp_state_dict_type: FULL_STATE_DICT
  fsdp_cpu_ram_efficient_loading: false
  fsdp_activation_checkpointing: false
  fsdp_offload_params: false
machine_rank: $NODE_RANK
mixed_precision: bf16
num_machines: $NNODES
num_processes: $WORLD_SIZE
rdzv_backend: static
same_network: true
use_cpu: false
YAML

# 这里只用 Python 标准库把本地权重文件名转换为 JSON 参数，不加载模型。
MODEL_PATHS=$("$PYTHON_BIN" -c '
import glob, json, sys
print(json.dumps([sorted(glob.glob(sys.argv[1] + "/Ref2VA/transformer/model*.safetensors"))]))
' "$MODEL_DIR")

# ---------- 7. 原生训练入口 ----------
# sync_each_batch=true：用户已明确同意；使用仓库原生参数。
# disable_model_save：分支原生参数，关闭所有模型保存。
# 开启普通梯度检查点；没有CPU/disk/optimizer/activation offload。
# 参数输出在日志中可直接审阅，不替换任何Python类或函数。
printf 'node_rank=%s nodes=%s world_size=%s microbatch=%s accumulation=%s global_batch=%s\n' \
    "$NODE_RANK" "$NNODES" "$WORLD_SIZE" "$MICRO_BATCH_SIZE" "$GRAD_ACCUM_STEPS" "$GLOBAL_BATCH_SIZE"
cd "$REPO_DIR"
set -x
exec "$PYTHON_BIN" -m accelerate.commands.launch \
    --config_file "$FSDP_CONFIG" \
    --num_machines "$NNODES" \
    --num_processes "$WORLD_SIZE" \
    --machine_rank "$NODE_RANK" \
    --main_process_ip "$MASTER_ADDR" \
    --main_process_port "$MASTER_PORT" \
    examples/minimax_h3/model_training/train.py \
    --synthetic_inputs \
    --dataset_repeat "$DATASET_REPEAT" \
    --dataset_num_workers 0 \
    --height "$HEIGHT" --width "$WIDTH" --num_frames "$NUM_FRAMES" \
    --model_paths "$MODEL_PATHS" \
    --trainable_models dit --task sft:train \
    --learning_rate "$LEARNING_RATE" \
    --weight_decay "$WEIGHT_DECAY" \
    --warm_up_steps "$LR_WARMUP_STEPS" \
    --num_epochs 1 \
    --customized_optimizer "$OPTIMIZER" \
    --max_grad_norm "$MAX_GRAD_NORM" \
    --gradient_accumulation_steps "$GRAD_ACCUM_STEPS" \
    --use_gradient_checkpointing \
    --fsdp2_sync_each_batch true \
    --fsdp2_reshard_after_backward true \
    --fsdp2_param_dtype bfloat16 \
    --fsdp2_reduce_dtype float32 \
    --fsdp2_output_dtype none \
    --fsdp2_cast_forward_inputs false \
    --fsdp2_forward_prefetch 0 \
    --fsdp2_backward_prefetch 0 \
    --fsdp2_hsdp_defer_all_reduce false \
    --audio_loss_weight 1.0 \
    --find_unused_parameters \
    --enable_csv_log \
    --remove_prefix_in_ckpt pipe.dit. \
    --output_path "$OUTPUT_DIR" \
    --disable_model_save
