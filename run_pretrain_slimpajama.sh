#!/usr/bin/env bash
# Train MobileLLM-R1-360M from scratch (random init from config) on DKYoon/SlimPajama-6B.
# Checkpoints every 1 B tokens in HuggingFace format; test perplexity, FLOPs, and
# token count logged to wandb on every checkpoint.

set -euo pipefail

# ---------- wandb config (edit or export before running) ----------
export WANDB_PROJECT="${WANDB_PROJECT:-slimpajama-pretrain}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-mobilellm-360m-slimpajama-$(date +%Y%m%d-%H%M)}"
# ------------------------------------------------------------------

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

torchrun \
  --nproc_per_node="$NPROC_PER_NODE" \
  pretrain.py \
  \
  --input_model_filename "facebook/MobileLLM-R1-360M-base" \
  --output_dir "./checkpoints/mobilellm-slimpajama" \
  \
  --do_train True \
  --do_eval True \
  \
  --model_max_length 2048 \
  --fp16 False \
  --bf16 True \
  \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 4 \
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
  --save_strategy "steps" \
  --save_total_limit 3 \
  --tokens_per_checkpoint 1000000000 \
  \
  --eval_strategy "no" \
  --ddp_find_unused_parameters False \
  --log_on_each_node False \
  --dataloader_num_workers 0 \
  \
  --dataset_name "DKYoon/SlimPajama-6B" \
  --dataset_subset "" \
  --total_tokens 6000000000 \
  --eval_max_samples 500 \
  --streaming True \
  --buffer_size 10000 \
  --num_proc 8 \
  \
  --gradient_checkpointing False \
  --seed 42
