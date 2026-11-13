#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root. Defaults target 8 MUSA devices with FusedAdamW.
# Set DIFFSYNTH_MODEL_BASE_PATH / DIFFSYNTH_SKIP_DOWNLOAD to use local models.
# Reuse an existing cache: STAGE=train CACHE_DIR=/path/to/cache bash <this-script>
# Other devices may override CUSTOMIZED_OPTIMIZER (e.g. torch.optim.AdamW).
STAGE="${STAGE:-all}"
DATASET_PATH="${DATASET_PATH:-data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-FL2VA}"
CACHE_DIR="${CACHE_DIR:-./models/train/MiniMax-H3-FL2VA-full-split-cache}"
OUTPUT_PATH="${OUTPUT_PATH:-./models/train/MiniMax-H3-FL2VA-full-fsdp2}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_MUSA_ALLOC_CONF="${PYTORCH_MUSA_ALLOC_CONF:-expandable_segments:True}"

case "$STAGE" in
  all|data_process|train) ;;
  *) echo "STAGE must be all, data_process, or train" >&2; exit 2 ;;
esac

if [[ "$STAGE" == all || "$STAGE" == data_process ]]; then
  if [[ "$DATASET_PATH" == data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-FL2VA ]]; then
    modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset \
      --include 'minimax_h3/MiniMax-H3-FL2VA/*' --local_dir ./data/diffsynth_example_dataset
  fi

  # Stage 1: cache preprocessing outputs in a single process, without FSDP.
  # input_image / end_image use the first and last frame of the training video.
  python examples/minimax_h3/model_training/train.py \
    --dataset_base_path "$DATASET_PATH" \
    --dataset_metadata_path "$DATASET_PATH/metadata.csv" \
    --data_file_keys video,input_audio \
    --extra_inputs input_audio,input_image,end_image \
    --height 480 \
    --width 832 \
    --num_frames 124 \
    --dataset_repeat 1 \
    --model_id_with_origin_paths 'MiniMax/MiniMax-H3:FL2VA/text_encoder/model*.safetensors,MiniMax/MiniMax-H3:FL2VA/video_vae/source/model.safetensors,MiniMax/MiniMax-H3:FL2VA/audio_vae/model.safetensors' \
    --learning_rate 1e-5 \
    --num_epochs 1 \
    --remove_prefix_in_ckpt pipe.dit. \
    --output_path "$CACHE_DIR" \
    --trainable_models dit \
    --use_gradient_checkpointing \
    --task sft:data_process
fi

if [[ "$STAGE" == all || "$STAGE" == train ]]; then
  # Stage 2: BF16 FSDP2 compute with FP32 parameter shards and Adam states.
  # FusedAdamW avoids the default AdamW optimizer-step OOM at this MUSA size.
  accelerate launch \
    --config_file examples/minimax_h3/model_training/full/accelerate_config_fsdp2.yaml \
    --main_process_port "${MAIN_PROCESS_PORT:-29500}" \
    examples/minimax_h3/model_training/train.py \
    --dataset_base_path "$CACHE_DIR" \
    --data_file_keys video,input_audio \
    --extra_inputs input_audio,input_image,end_image \
    --height 480 \
    --width 832 \
    --num_frames 124 \
    --dataset_repeat "${DATASET_REPEAT:-100}" \
    --model_id_with_origin_paths 'MiniMax/MiniMax-H3:FL2VA/transformer/model*.safetensors' \
    --learning_rate 1e-5 \
    --num_epochs "${NUM_EPOCHS:-2}" \
    --customized_optimizer "${CUSTOMIZED_OPTIMIZER:-torch_musa.optim.FusedAdamW}" \
    --max_grad_norm 1.0 \
    --remove_prefix_in_ckpt pipe.dit. \
    --output_path "$OUTPUT_PATH" \
    --trainable_models dit \
    --use_gradient_checkpointing \
    --find_unused_parameters \
    --enable_csv_log \
    --task sft:train \
    "$@"
fi
