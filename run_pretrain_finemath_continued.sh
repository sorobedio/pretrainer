#!/usr/bin/env bash
# Continue pretraining from bedio/MobileLLM-R1-360M-base-expanded-from-MobileLLM-R1-140M-base-inr
# on DKYoon/SlimPajama-6B. Checkpoints every 1M tokens; test perplexity, FLOPs, and
# token count logged to wandb on every checkpoint.

set -euo pipefail

# ---------- wandb config (edit or export before running) ----------
export WANDB_PROJECT="${WANDB_PROJECT:-slimpajama-continued}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-mobilellm-360m-slimpajama-continued-$(date +%Y%m%d-%H%M)}"
# ------------------------------------------------------------------

# Set CUDA_VISIBLE_DEVICES to target specific GPUs, e.g.:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_pretrain_finemath_continued.sh
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  export CUDA_VISIBLE_DEVICES
  NPROC_PER_NODE=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l | tr -d ' ')
else
  NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
fi

torchrun \
  --nproc_per_node="$NPROC_PER_NODE" \
  pretrain.py \
  \
  --input_model_filename "bedio/360M-from-140M-inr_rescale-rand-embed_tokens-lm_head" \
  --init_from_pretrained True \
  --output_dir "./checkpoints/mobilellm-finemath-continued" \
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
  --learning_rate 1e-4 \
  --weight_decay 0.1 \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --adam_epsilon 1e-8 \
  --lr_scheduler_type "cosine" \
  --warmup_steps 500 \
  --max_steps 500000 \
  \
  --logging_steps 10 \
  --logging_dir "./logs" \
  --report_to "wandb" \
  \
  --save_strategy "steps" \
  --save_total_limit 5 \
  --tokens_per_checkpoint 1000000 \
  \
  --eval_strategy "no" \
  --ddp_find_unused_parameters False \
  --log_on_each_node False \
  --dataloader_num_workers 0 \
  \
  --dataset_name "DKYoon/SlimPajama-6B" \
  --dataset_subset "" \
  --eval_split "test" \
  --eval_max_samples 500 \
  --streaming True \
  --buffer_size 10000 \
  --num_proc 8 \
  \
  --gradient_checkpointing False \
  --seed 42
