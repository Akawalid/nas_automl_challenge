#!/bin/bash

#SBATCH --job-name=global_scheduler_350
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/%u/dag_experiments/global_scheduler_350/slurm_logs/%x-%j.out
#SBATCH --error=/scratch/%u/dag_experiments/global_scheduler_350/slurm_logs/%x-%j.err

set -euo pipefail

EXPERIMENT_DIR="${EXPERIMENT_DIR:-${SLURM_SUBMIT_DIR:-$HOME/dev/experimental_grow}}"
BASE_SCRIPT="$EXPERIMENT_DIR/experiments/pipeline/run_init_scaling_ablation.slurm"

if [ ! -f "$BASE_SCRIPT" ]; then
    echo "Base SLURM script not found: $BASE_SCRIPT"
    exit 1
fi

# Single ablation run: local initialization, seed 0 and the base dataset.
export INIT_STRATEGIES="${INIT_STRATEGIES:-local}"
export DATASETS="${DATASETS:-multnist}"
export SEEDS="${SEEDS:-0}"
export RUN_DIR="${RUN_DIR:-/scratch/$USER/dag_experiments/global_scheduler_350}"
export EXPERIMENT_NAME_PREFIX="${EXPERIMENT_NAME_PREFIX:-global_scheduler_350}"

# Keep this run in the same W&B project as the initialization ablations.
export WANDB_PROJECT="${WANDB_PROJECT:-demeter-init-scaling-ablation}"

exec bash "$BASE_SCRIPT" \
    --training.ablations.linear_warmup.enabled false \
    --training.ablations.post_growth_scheduler.enabled false \
    --training.ablations.global_scheduler.enabled true \
    --training.ablations.global_scheduler.total_epochs 350 \
    "$@"
