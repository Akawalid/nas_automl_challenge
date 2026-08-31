#!/bin/bash

set -euo pipefail

DATA_DIR="${DATA_DIR:-$HOME/datasets}"
DATASETS="${DATASETS:-cifar10 cifar100}"
SEEDS="${SEEDS:-0}"
INIT_STRATEGIES="${INIT_STRATEGIES:-local init_scaling_ablation}"
WANDB_PROJECT="${WANDB_PROJECT:-demeter-cifar-init-noisy-ablation}"
SCRIPT_PATH="${SCRIPT_PATH:-experiments/pipeline/run_cifar_init_noisy_ablation.slurm}"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-1}"

mkdir -p \
    /scratch/"$USER"/dag_experiments/cifar_init_ablation/slurm_logs \
    /scratch/"$USER"/dag_experiments/cifar_noisy_pre_grad_005/slurm_logs \
    /scratch/"$USER"/dag_experiments/cifar_noisy_pre_grad_010/slurm_logs

INIT_JOB=$(
    DATA_DIR="$DATA_DIR" \
    DATASETS="$DATASETS" \
    SEEDS="$SEEDS" \
    INIT_STRATEGIES="$INIT_STRATEGIES" \
    RUN_DIR="/scratch/$USER/dag_experiments/cifar_init_ablation" \
    EXPERIMENT_NAME_PREFIX="cifar_init_ablation" \
    WANDB_PROJECT="$WANDB_PROJECT" \
    sbatch --parsable --job-name=cifar_init --array=0-3%"$ARRAY_CONCURRENCY" \
        --output="/scratch/$USER/dag_experiments/cifar_init_ablation/slurm_logs/%x-%A_%a.out" \
        --error="/scratch/$USER/dag_experiments/cifar_init_ablation/slurm_logs/%x-%A_%a.err" \
        "$SCRIPT_PATH" \
        --growth.ablations.noisy_pre_activities_grad.enabled false \
        --growth.ablations.variance_transfer.enabled false
)

NOISE5_JOB=$(
    DATA_DIR="$DATA_DIR" \
    DATASETS="$DATASETS" \
    SEEDS="$SEEDS" \
    INIT_STRATEGIES="$INIT_STRATEGIES" \
    RUN_DIR="/scratch/$USER/dag_experiments/cifar_noisy_pre_grad_005" \
    EXPERIMENT_NAME_PREFIX="cifar_noisy_pre_grad_005" \
    WANDB_PROJECT="$WANDB_PROJECT" \
    sbatch --parsable --dependency=afterok:"$INIT_JOB" --job-name=cifar_noise5 --array=0-3%"$ARRAY_CONCURRENCY" \
        --output="/scratch/$USER/dag_experiments/cifar_noisy_pre_grad_005/slurm_logs/%x-%A_%a.out" \
        --error="/scratch/$USER/dag_experiments/cifar_noisy_pre_grad_005/slurm_logs/%x-%A_%a.err" \
        "$SCRIPT_PATH" \
        --growth.ablations.noisy_pre_activities_grad.enabled true \
        --growth.ablations.noisy_pre_activities_grad.std 0.05 \
        --growth.ablations.noisy_pre_activities_grad.relative true \
        --growth.ablations.variance_transfer.enabled false
)

NOISE10_JOB=$(
    DATA_DIR="$DATA_DIR" \
    DATASETS="$DATASETS" \
    SEEDS="$SEEDS" \
    INIT_STRATEGIES="$INIT_STRATEGIES" \
    RUN_DIR="/scratch/$USER/dag_experiments/cifar_noisy_pre_grad_010" \
    EXPERIMENT_NAME_PREFIX="cifar_noisy_pre_grad_010" \
    WANDB_PROJECT="$WANDB_PROJECT" \
    sbatch --parsable --dependency=afterok:"$NOISE5_JOB" --job-name=cifar_noise10 --array=0-3%"$ARRAY_CONCURRENCY" \
        --output="/scratch/$USER/dag_experiments/cifar_noisy_pre_grad_010/slurm_logs/%x-%A_%a.out" \
        --error="/scratch/$USER/dag_experiments/cifar_noisy_pre_grad_010/slurm_logs/%x-%A_%a.err" \
        "$SCRIPT_PATH" \
        --growth.ablations.noisy_pre_activities_grad.enabled true \
        --growth.ablations.noisy_pre_activities_grad.std 0.10 \
        --growth.ablations.noisy_pre_activities_grad.relative true \
        --growth.ablations.variance_transfer.enabled false
)

echo "INIT_JOB=$INIT_JOB"
echo "NOISE5_JOB=$NOISE5_JOB"
echo "NOISE10_JOB=$NOISE10_JOB"
