#!/usr/bin/env bash
# Evaluate all checkpoints in checkpoints/llama-3b-scratch-fineweb-v2.
# Trained by run_pretrain_llama3b_scratch.sh:
#   meta-llama/Llama-3.2-3B  |  FineWeb-edu 10BT  |  4 GPUs  |  bs=4  |  grad_acc=4  |  seq=4096
#   tokens_per_step = 4 × 4 × 4 × 4096 = 262 144
#
# Runs perplexity (wikitext, lambada, text8) + benchmarks (winogrande, mmlu,
# arc_challenge, arc_easy, hellaswag) via vLLM.
# Results logged to wandb project: fineweb-continued

set -euo pipefail

CHECKPOINT_DIR="${CHECKPOINT_DIR:-./checkpoints/llama-3b-scratch-fineweb-v2}"
OUTPUT_PATH="${OUTPUT_PATH:-${CHECKPOINT_DIR}/eval_results.json}"
WANDB_PROJECT="${WANDB_PROJECT:-fineweb-continued}"
WANDB_RUN="${WANDB_RUN:-llama-3b-scratch-fineweb-eval-$(date +%Y%m%d-%H%M)}"

# tokens_per_step: 4 GPUs × bs 4 × grad_acc 4 × seq 4096
TOKENS_PER_STEP=262144

python eval_checkpoints_list.py \
  --checkpoint_dir  "$CHECKPOINT_DIR" \
  --output_path     "$OUTPUT_PATH" \
  --wandb_project   "$WANDB_PROJECT" \
  --wandb_run       "$WANDB_RUN" \
  --mode            all \
  --tokens_per_step $TOKENS_PER_STEP \
  --seq_len         4096 \
  --batch_size      4 \
  --max_checkpoints 10 \
  --use_vllm \
  --max_model_len   4096 \
  --gpu_memory_utilization 0.85 \
  --ppl_datasets \
    "Salesforce/wikitext:wikitext-2-raw-v1:test" \
    "cimec/lambada::test" \
    "afmck/text8::test"
