#!/usr/bin/env bash
# Evaluate checkpoints from /c2/soro/checkpoints-tuner/mobilellm-360m-slimpajama-scratch
# Model: MobileLLM-R1-360M trained from scratch on DKYoon/SlimPajama-6B
# Training used 1 GPU, bs=4, grad_acc=4, seq_len=2048 → tokens/step=32768
# Checkpoints: every 100M tokens (step interval=3051), total ~1B tokens

set -euo pipefail

CHECKPOINT_DIR="${CHECKPOINT_DIR:-/c2/soro/checkpoints-tuner/mobilellm-360m-slimpajama-scratch}"
OUTPUT_PATH="${OUTPUT_PATH:-${CHECKPOINT_DIR}/eval_results.json}"
DEVICE="${DEVICE:-cuda}"

python eval_checkpoints_list.py \
    --checkpoint_dir  "$CHECKPOINT_DIR" \
    --output_path     "$OUTPUT_PATH" \
    --wandb_project   "slimpajama-scratch" \
    --wandb_run       "mobilellm-360m-slimpajama-scratch-eval" \
    --tokens_per_step 32768 \
    --mode            perplexity \
    --batch_size      8 \
    --seq_len         2048 \
    --device          "$DEVICE" \
    --ppl_datasets \
        "Salesforce/wikitext:wikitext-2-raw-v1:test" \
        "cimec/lambada::test" \
        "afmck/text8::test"
