#!/usr/bin/env bash
# Continue pretraining bedio/gemma-3-270m-gate-inr (pretrained init) on DKYoon/SlimPajama-6B.
# Evaluates on SlimPajama-6B test split. Saves every 100M tokens; test perplexity
# logged every 1M tokens.

set -euo pipefail

# ---------- wandb config (edit or export before running) ----------
MODEL_ID="bedio/gemma-3-270m-gate-inr"
DATASET_ID="DKYoon/SlimPajama-6B"
_MODEL_TAG="${MODEL_ID##*/}"      # gemma-3-270m-gate-inr
_DATASET_TAG="${DATASET_ID##*/}"  # SlimPajama-6B

OUTPUT_DIR="./checkpoints/${_MODEL_TAG}-${_DATASET_TAG}-cont-v0"

export WANDB_PROJECT="${WANDB_PROJECT:-${_DATASET_TAG}}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${_MODEL_TAG}-cont-$(date +%Y%m%d-%H%M)}"
export WANDB_HTTP_TIMEOUT=300
export WANDB_INIT_TIMEOUT=120
# ------------------------------------------------------------------

# Set CUDA_VISIBLE_DEVICES to target specific GPUs, e.g.:
#   CUDA_VISIBLE_DEVICES=0,1 bash run_pretrain_slimpajama_gateinr_v0.sh
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  export CUDA_VISIBLE_DEVICES
  NPROC_PER_NODE=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l | tr -d ' ')
else
  NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
fi

mkdir -p "$OUTPUT_DIR"
mkdir -p ./logs

torchrun \
  --standalone \
  --nproc_per_node="$NPROC_PER_NODE" \
  pretrain.py \
  \
  --input_model_filename "$MODEL_ID" \
  --init_from_pretrained True \
  --output_dir "$OUTPUT_DIR" \
  \
  --do_train True \
  --do_eval True \
  \
  --model_max_length 2048 \
  --fp16 False \
  --bf16 True \
  \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 8 \
  \
  --learning_rate 3e-4 \
  --weight_decay 0.1 \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --adam_epsilon 1e-8 \
  --lr_scheduler_type "cosine" \
  --warmup_steps 2000 \
  \
  --logging_steps 10 \
  --logging_dir "./logs" \
  --report_to "wandb" \
  \
  --tokens_per_checkpoint 100000000 \
  --eval_tokens_interval 1000000 \
  \
  --eval_strategy "no" \
  --ddp_find_unused_parameters False \
  --log_on_each_node False \
  --dataloader_num_workers 0 \
  \
  --dataset_name "$DATASET_ID" \
  --dataset_subset "" \
  --eval_split "test" \
  --total_tokens 6000000000 \
  --eval_max_samples 500 \
  --streaming True \
  --buffer_size 10000 \
  --num_proc 8 \
  \
  --gradient_checkpointing True \
  --seed 42
