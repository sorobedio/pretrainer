#!/bin/bash
#SBATCH --job-name=buildfinedata
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128000M
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --account=def-boris

# ── Modules & environment ──
source ~/scratch/soro_env/bin/activate

# ── Directories ──
mkdir -p logs
mkdir -p /c2/soro/finesubset

# ── HuggingFace cache on scratch ──
export HF_HOME=$SCRATCH/.cache/huggingface
export TRANSFORMERS_CACHE=$SCRATCH/.cache/huggingface
mkdir -p $HF_HOME

echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"

# ── Launch ──
python buildfinedata.py
