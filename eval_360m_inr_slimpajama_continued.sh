#!/usr/bin/env bash
# Evaluate checkpoints from contcheckpoints/360m-inr-slimpajama-continued
# Model: bedio/360M-from-140M-inr, trained on DKYoon/SlimPajama-6B
# Training used 8 GPUs, bs=4, grad_acc=4, seq_len=2048 → tokens/step=262144
# Checkpoints: every 100M tokens (step interval=381), total ~2.3B tokens

set -euo pipefail

CHECKPOINT_DIR="${CHECKPOINT_DIR:-./contcheckpoints/360m-inr-slimpajama-continued}"
OUTPUT_PATH="${OUTPUT_PATH:-${CHECKPOINT_DIR}/eval_results.json}"
DEVICE="${DEVICE:-cuda}"

python eval_checkpoints_list.py \
    --checkpoint_dir  "$CHECKPOINT_DIR" \
    --output_path     "$OUTPUT_PATH" \
    --wandb_project   "slimpajama-continued" \
    --wandb_run       "360m-inr-slimpajama-eval" \
    --tokens_per_step 262144 \
    --mode            perplexity \
    --batch_size      8 \
    --seq_len         2048 \
    --device          "$DEVICE" \
    --ppl_datasets \
        "Salesforce/wikitext:wikitext-2-raw-v1:test" \
        "cimec/lambada::test" \
        "afmck/text8::test"
