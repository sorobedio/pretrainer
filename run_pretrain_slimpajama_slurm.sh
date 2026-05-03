#!/bin/bash
#SBATCH --job-name=slimpajama_pretrain
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256000M
#SBATCH --time=72:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --account=def-boris

# ── Modules & environment ──
module load cuda/12.6
source ~/scratch/soro_env/bin/activate

# ── Directories ──
mkdir -p logs
mkdir -p /c2/soro/checkpoints/mobilellm-360m-slimpajama-scratch

# ── NCCL ──
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"

# ── HuggingFace cache on scratch ──
export HF_HOME=$SCRATCH/.cache/huggingface
export TRANSFORMERS_CACHE=$SCRATCH/.cache/huggingface
mkdir -p $HF_HOME

# ── Wandb ──
export WANDB_PROJECT="slimpajama"
export WANDB_RUN_NAME="mobilellm-360m-slimpajama-scratch_${SLURM_JOB_ID}"

# ── Launch ──
torchrun \
    --nproc_per_node=2 \
    --nnodes=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    pretrain.py \
    \
    --input_model_filename "bedio/360M-from-140M-inr-rand-embed_tokens-lm_head" \
    --init_from_pretrained False \
    --output_dir "./checkpoints/mobilellm-360m-slimpajama-raw" \
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
    \
    --logging_steps 10 \
    --logging_dir "./checkpoints/mobilellm-360m-slimpajama-raw/logs" \
    --report_to "wandb" \
    \
    --save_total_limit 0 \
    --variable_checkpoint_schedule True \
    --eval_tokens_interval 1000000 \
    \
    --eval_strategy "no" \
    --ddp_find_unused_parameters False \
    --log_on_each_node False \
    --dataloader_num_workers 4 \
    \
    --dataset_name "DKYoon/SlimPajama-6B" \
    --dataset_subset "" \
    --eval_split "test" \
    --total_tokens 6000000000 \
    --eval_max_samples 500 \
    --streaming True \
    --buffer_size 10000 \
    --num_proc 8 \
    \
    --gradient_checkpointing False \
    --seed 42
