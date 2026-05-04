#!/usr/bin/env bash
# Continue pretraining bedio/MobileLLM-R1-360M-HyperCloned-from-140M on FineWeb-edu 10BT.
# Evaluates on SlimPajama-6B test split. Saves every 100M tokens; test perplexity
# logged every 1M tokens.

set -euo pipefail

# ---------- wandb config (edit or export before running) ----------
export WANDB_PROJECT="${WANDB_PROJECT:-fineweb-continued}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-mobilellm-360m-hyperclone-fineweb10B-cont-$(date +%Y%m%d-%H%M)}"
# ------------------------------------------------------------------

# Set CUDA_VISIBLE_DEVICES to target specific GPUs, e.g.:
#   CUDA_VISIBLE_DEVICES=0,1 bash run_pretrain_fineweb_hyperclone_cont.sh
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  export CUDA_VISIBLE_DEVICES
  NPROC_PER_NODE=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l | tr -d ' ')
else
  NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
fi

mkdir -p ./checkpoints/mobilellm-360m-hyperclone-fineweb-cont
mkdir -p ./logs

torchrun \
  --nproc_per_node="$NPROC_PER_NODE" \
  pretrain.py \
  \
  --input_model_filename "bedio/MobileLLM-R1-360M-HyperCloned-from-140M" \
  --init_from_pretrained True \
  --output_dir "./checkpoints/mobilellm-360m-hyperclone-fineweb-cont" \
  \
  --do_train True \
  --do_eval True \
  \
  --model_max_length 4096 \
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
  --tokens_per_checkpoint 100000000 \
  --eval_tokens_interval 1000000 \
  \
  --eval_strategy "no" \
  --ddp_find_unused_parameters False \
  --log_on_each_node False \
  --dataloader_num_workers 0 \
  \
  --dataset_name "HuggingFaceFW/fineweb-edu" \
  --dataset_subset "sample-10BT" \
  --eval_dataset_name "DKYoon/SlimPajama-6B" \
  --eval_dataset_subset "" \
  --eval_split "test" \
  --total_tokens 10000000000 \
  --eval_max_samples 500 \
  --streaming True \
  --buffer_size 10000 \
  --num_proc 8 \
  \
  --gradient_checkpointing False \
  --seed 42
