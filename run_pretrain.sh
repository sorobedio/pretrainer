#!/usr/bin/env bash
# Train MobileLLM-R1-950M from scratch on finemath-4plus.
# Checkpoints every 1 B tokens; loss + FLOPs logged to wandb.
#
# Single-node multi-GPU example (8 GPUs):
#   bash run_pretrain.sh
#
# Multi-node example (2 nodes, 8 GPUs each):
#   torchrun --nnodes=2 --nproc_per_node=8 \
#            --node_rank=$NODE_RANK \
#            --master_addr=$MASTER_ADDR \
#            --master_port=29500 \
#            pretrain.py <args …>

set -euo pipefail

# ---------- wandb config (edit or export before running) ----------
export WANDB_PROJECT="${WANDB_PROJECT:-finemath-pretrain}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-mobilellm-950m-$(date +%Y%m%d-%H%M)}"
# ------------------------------------------------------------------

NNODES="${NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

torchrun \
  --nnodes="$NNODES" \
  --nproc_per_node="$NPROC_PER_NODE" \
  pretrain.py \
  \
  --input_model_filename "facebook/MobileLLM-R1-360M-base" \
  --output_dir "/c2//checkpoints/mobilellm-finemath" \
  \
  --do_train True \
  --do_eval False \
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
  --max_steps 500000 \
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
  --dataset_name "HuggingFaceTB/finemath" \
  --dataset_subset "finemath-4plus" \
  --streaming True \
  --buffer_size 10000 \
  --num_proc 8 \
  \
  --gradient_checkpointing False \
  --seed 42
