#!/usr/bin/env bash
# Evaluate checkpoints from run_pretrain_slimpajama.sh
#   Model:  bedio/gemma-3-270m-cloned-init  (pretrained init)
#   Data:   HuggingFaceFW/fineweb-edu sample-10BT
#   Setup:  1 GPU  |  bs=4  |  grad_acc=4  |  seq=4096
#   tokens_per_step = 1 × 4 × 4 × 4096 = 65 536

set -euo pipefail

CHECKPOINT_DIR="${CHECKPOINT_DIR:-./checkpoints/gemma-pretrained}"
OUTPUT_PATH="${OUTPUT_PATH:-${CHECKPOINT_DIR}/eval_results.json}"
WANDB_PROJECT="${WANDB_PROJECT:-slimpajama-continued}"
WANDB_RUN="${WANDB_RUN:-gemma-270m-pretrained-fineweb-eval-$(date +%Y%m%d-%H%M)}"

TOKENS_PER_STEP=65536

python eval_checkpoints_list.py \
  --checkpoint_dir  "$CHECKPOINT_DIR" \
  --output_path     "$OUTPUT_PATH" \
  --wandb_project   "$WANDB_PROJECT" \
  --wandb_run       "$WANDB_RUN" \
  --mode            all \
  --tokens_per_step $TOKENS_PER_STEP \
  --seq_len         4096 \
  --batch_size      4 \
  --use_vllm \
  --max_model_len   4096 \
  --gpu_memory_utilization 0.85 \
  --ppl_datasets \
    "Salesforce/wikitext:wikitext-2-raw-v1:test" \
    "cimec/lambada::test" \
    "afmck/text8::test"
