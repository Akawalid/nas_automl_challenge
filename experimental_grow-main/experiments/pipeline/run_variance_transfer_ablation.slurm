#!/bin/bash

#SBATCH --job-name=variance_transfer
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --array=0-1
#SBATCH --output=/scratch/%u/dag_experiments/variance_transfer/slurm_logs/%x-%A_%a.out
#SBATCH --error=/scratch/%u/dag_experiments/variance_transfer/slurm_logs/%x-%A_%a.err

set -euo pipefail

EXPERIMENT_DIR="${EXPERIMENT_DIR:-${SLURM_SUBMIT_DIR:-$HOME/dev/experimental_grow}}"
BASE_SCRIPT="$EXPERIMENT_DIR/experiments/pipeline/run_init_scaling_ablation.slurm"

if [ ! -f "$BASE_SCRIPT" ]; then
    echo "Base SLURM script not found: $BASE_SCRIPT"
    exit 1
fi

# Two ablation runs: local initialization and init/scaling initialization.
# The array limit is %1 by default so only one run is active at a time.
export INIT_STRATEGIES="${INIT_STRATEGIES:-local init_scaling_ablation}"
export DATASETS="${DATASETS:-multnist}"
export SEEDS="${SEEDS:-0}"
export RUN_DIR="${RUN_DIR:-/scratch/$USER/dag_experiments/variance_transfer}"
export EXPERIMENT_NAME_PREFIX="${EXPERIMENT_NAME_PREFIX:-variance_transfer}"
export VARIANCE_TRANSFER_RESCALING="${VARIANCE_TRANSFER_RESCALING:-vt_constraint_old_shape}"

# Keep these runs in the same W&B project as the other ablations.
export WANDB_PROJECT="${WANDB_PROJECT:-demeter-init-scaling-ablation}"

exec bash "$BASE_SCRIPT" \
    --training.ablations.linear_warmup.enabled false \
    --training.ablations.global_scheduler.enabled false \
    --training.ablations.post_growth_scheduler.enabled false \
    --training.ablations.growth_label_smoothing.enabled false \
    --growth.ablations.noisy_pre_activities_grad.enabled false \
    --growth.ablations.variance_transfer.enabled true \
    --growth.ablations.variance_transfer.rescaling "$VARIANCE_TRANSFER_RESCALING" \
    "$@"
