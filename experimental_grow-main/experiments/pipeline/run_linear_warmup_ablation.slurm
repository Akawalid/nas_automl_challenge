#!/bin/bash

#SBATCH --job-name=linear_warmup_ablation
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --array=0-1
#SBATCH --output=/scratch/%u/dag_experiments/linear_warmup_ablation/slurm_logs/%x-%A_%a.out
#SBATCH --error=/scratch/%u/dag_experiments/linear_warmup_ablation/slurm_logs/%x-%A_%a.err

set -euo pipefail

EXPERIMENT_DIR="${EXPERIMENT_DIR:-${SLURM_SUBMIT_DIR:-$HOME/dev/experimental_grow}}"
BASE_SCRIPT="$EXPERIMENT_DIR/experiments/pipeline/run_init_scaling_ablation.slurm"

if [ ! -f "$BASE_SCRIPT" ]; then
    echo "Base SLURM script not found: $BASE_SCRIPT"
    exit 1
fi

# Two parallel runs: local initialization and init/scaling ablation.
export INIT_STRATEGIES="${INIT_STRATEGIES:-local init_scaling_ablation}"
export DATASETS="${DATASETS:-multnist}"
export SEEDS="${SEEDS:-0}"
export RUN_DIR="${RUN_DIR:-/scratch/$USER/dag_experiments/linear_warmup_ablation}"
export EXPERIMENT_NAME_PREFIX="${EXPERIMENT_NAME_PREFIX:-linear_warmup}"
export WARMUP_EPOCHS="${WARMUP_EPOCHS:-10}"

# Keep both runs in the same W&B project as the existing ablations.
export WANDB_PROJECT="${WANDB_PROJECT:-demeter-init-scaling-ablation}"

exec bash "$BASE_SCRIPT" \
    --training.ablations.linear_warmup.enabled true \
    --training.ablations.linear_warmup.epochs "$WARMUP_EPOCHS" \
    --training.ablations.global_scheduler.enabled false \
    "$@"
