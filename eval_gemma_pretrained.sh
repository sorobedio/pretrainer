#!/usr/bin/env bash
# Evaluate checkpoints/gemma-pretrained on perplexity + benchmarks.
# Trained by run_pretrain_slimpajama.sh (pretrained init):
#   bedio/gemma-3-270m-gate-inr  |  FineWeb-edu  |  1 GPU  |  bs=4  |  grad_acc=4  |  seq=4096
#   tokens_per_step = 1 × 4 × 4 × 4096 = 65 536
#
# Usage:
#   bash eval_gemma_pretrained.sh                  # wikitext perplexity only
#   EVAL_MODE=all bash eval_gemma_pretrained.sh    # wikitext + lambada + text8 + benchmarks

set -euo pipefail

CHECKPOINT_DIR="${CHECKPOINT_DIR:-./checkpoints/gemma-pretrained}"
OUTPUT_PATH="${OUTPUT_PATH:-${CHECKPOINT_DIR}/eval_results.json}"
WANDB_PROJECT="${WANDB_PROJECT:-slimpajama-continued}"
WANDB_RUN="${WANDB_RUN:-gemma-pretrained-eval-$(date +%Y%m%d-%H%M)}"

# wikitext  → perplexity on wikitext only
# all       → wikitext + lambada + text8 + all benchmarks
EVAL_MODE="${EVAL_MODE:-wikitext}"

TOKENS_PER_STEP=65536

if [ "$EVAL_MODE" = "all" ]; then
  PPL_DATASETS=(
    "Salesforce/wikitext:wikitext-2-raw-v1:test"
    "cimec/lambada::test"
    "afmck/text8::test"
  )
  MODE_ARG="all"
else
  PPL_DATASETS=("Salesforce/wikitext:wikitext-2-raw-v1:test")
  MODE_ARG="perplexity"
fi

python eval_checkpoints_list.py \
  --checkpoint_dir  "$CHECKPOINT_DIR" \
  --output_path     "$OUTPUT_PATH" \
  --wandb_project   "$WANDB_PROJECT" \
  --wandb_run       "$WANDB_RUN" \
  --mode            "$MODE_ARG" \
  --tokens_per_step $TOKENS_PER_STEP \
  --seq_len         4096 \
  --batch_size      4 \
  --use_vllm \
  --max_model_len   4096 \
  --gpu_memory_utilization 0.85 \
  --ppl_datasets    "${PPL_DATASETS[@]}"
