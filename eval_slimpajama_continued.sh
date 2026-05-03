#!/usr/bin/env bash
# Evaluate checkpoints from run_pretrain_slimpajama_continued.sh
# Model: bedio/360M-from-140M-inr_rescale, trained on DKYoon/SlimPajama-6B
# Training used 2 GPUs, bs=4, grad_acc=4, seq_len=2048 → tokens/step=65536
# Checkpoints: every 100M tokens (step interval=1525), total ~1B tokens

set -euo pipefail

CHECKPOINT_DIR="${CHECKPOINT_DIR:-./contcheckpoints/360m-inr_scaled-slimpajama-continued}"
OUTPUT_PATH="${OUTPUT_PATH:-${CHECKPOINT_DIR}/eval_results.json}"
MODE="${MODE:-perplexity}"   # perplexity | benchmarks | all
DEVICE="${DEVICE:-cuda}"

python eval_checkpoints_list.py \
    --checkpoint_dir  "$CHECKPOINT_DIR" \
    --output_path     "$OUTPUT_PATH" \
    --wandb_project   "slimpajama-continued" \
    --wandb_run       "360m-inr-scaled-slimpajama-eval" \
    --tokens_per_step 65536 \
    --mode            "$MODE" \
    --batch_size      8 \
    --seq_len         2048 \
    --device          "$DEVICE"
