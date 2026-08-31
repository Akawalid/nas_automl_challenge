#!/bin/bash

set -euo pipefail

# CIFAR ablation suite. Defaults exclude the homogeneous nodes reserved for
# local-base benchmarking.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATA_DIR="${DATA_DIR:-/home/tau/$USER/datasets}"
DATASETS="${DATASETS:-cifar10 cifar100}"
SEEDS="${SEEDS:-0}"
INIT_STRATEGIES="${INIT_STRATEGIES:-local init_scaling_ablation}"
SCRIPT_PATH="${SCRIPT_PATH:-$SCRIPT_DIR/run_cifar_init_noisy_ablation.slurm}"
BENCHMARK_NODELIST="${BENCHMARK_NODELIST:-margpu010}"
WANDB_PROJECT="${WANDB_PROJECT:-demeter-cifar-ablation-suite}"
SBATCH_TIME="${SBATCH_TIME:-}"

if [[ "$SCRIPT_PATH" != /* ]]; then
    SCRIPT_PATH="$REPO_ROOT/$SCRIPT_PATH"
fi

if [ ! -f "$SCRIPT_PATH" ]; then
    echo "SLURM script not found: $SCRIPT_PATH"
    echo "Set SCRIPT_PATH to the correct run_cifar_init_noisy_ablation.slurm path."
    exit 1
fi

cd "$REPO_ROOT"

read -r -a DATASET_LIST <<< "$DATASETS"
read -r -a SEED_LIST <<< "$SEEDS"
read -r -a INIT_STRATEGY_LIST <<< "$INIT_STRATEGIES"

array_end() {
    local datasets_count=$1
    local seeds_count=$2
    local init_count=$3
    echo $((datasets_count * seeds_count * init_count - 1))
}

submit_ablation() {
    local job_name=$1
    local run_dir=$2
    local name_prefix=$3
    local init_strategies=$4
    shift 4

    read -r -a submit_init_list <<< "$init_strategies"
    local end
    end="$(array_end "${#DATASET_LIST[@]}" "${#SEED_LIST[@]}" "${#submit_init_list[@]}")"
    if [ "$end" -lt 0 ]; then
        echo "Skipping $job_name: empty DATASETS, SEEDS, or INIT_STRATEGIES"
        return
    fi

    mkdir -p "$run_dir/slurm_logs"

    local sbatch_args=(
        --parsable
        --job-name="$job_name"
        --array=0-"$end"
        --output="$run_dir/slurm_logs/%x-%A_%a.out"
        --error="$run_dir/slurm_logs/%x-%A_%a.err"
    )

    if [ -n "$SBATCH_TIME" ]; then
        sbatch_args+=(--time="$SBATCH_TIME")
    fi

    if [ -n "$BENCHMARK_NODELIST" ]; then
        sbatch_args+=(--exclude="$BENCHMARK_NODELIST")
    fi

    echo
    echo "Submitting $job_name"
    echo "  Run dir: $run_dir"
    echo "  Prefix: $name_prefix"
    echo "  Init strategies: $init_strategies"
    echo "  Array: 0-$end"
    echo "  Requested wall time: ${SBATCH_TIME:-from slurm script}"
    echo "  Excluding benchmark nodes: ${BENCHMARK_NODELIST:-none}"

    local job_id
    job_id=$(
        DATA_DIR="$DATA_DIR" \
        DATASETS="$DATASETS" \
        SEEDS="$SEEDS" \
        INIT_STRATEGIES="$init_strategies" \
        RUN_DIR="$run_dir" \
        EXPERIMENT_NAME_PREFIX="$name_prefix" \
        WANDB_PROJECT="$WANDB_PROJECT" \
        sbatch "${sbatch_args[@]}" "$SCRIPT_PATH" "$@"
    )
    echo "  JOB_ID=$job_id"
}

echo "Submitting CIFAR ablation suite"
echo "Datasets: $DATASETS"
echo "Seeds: $SEEDS"
echo "Default init strategies: $INIT_STRATEGIES"
echo "W&B project: $WANDB_PROJECT"
echo "Requested wall time: ${SBATCH_TIME:-from slurm script}"
echo "Benchmark nodes excluded from CIFAR: ${BENCHMARK_NODELIST:-none}"

submit_ablation \
    cifar_init \
    "/scratch/$USER/dag_experiments/cifar_init_ablation" \
    cifar_init_ablation \
    "$INIT_STRATEGIES" \
    --training.ablations.linear_warmup.enabled false \
    --training.ablations.global_scheduler.enabled false \
    --training.ablations.post_growth_scheduler.enabled false \
    --training.ablations.growth_label_smoothing.enabled false \
    --growth.ablations.noisy_pre_activities_grad.enabled false \
    --growth.ablations.variance_transfer.enabled false

submit_ablation \
    cifar_linear_warmup \
    "/scratch/$USER/dag_experiments/cifar_linear_warmup" \
    cifar_linear_warmup \
    "$INIT_STRATEGIES" \
    --training.ablations.linear_warmup.enabled true \
    --training.ablations.linear_warmup.epochs "${WARMUP_EPOCHS:-10}" \
    --training.ablations.global_scheduler.enabled false \
    --training.ablations.post_growth_scheduler.enabled false \
    --training.ablations.growth_label_smoothing.enabled false \
    --growth.ablations.noisy_pre_activities_grad.enabled false \
    --growth.ablations.variance_transfer.enabled false

submit_ablation \
    cifar_post_growth \
    "/scratch/$USER/dag_experiments/cifar_post_growth_scheduler" \
    cifar_post_growth_scheduler \
    "$INIT_STRATEGIES" \
    --training.ablations.linear_warmup.enabled false \
    --training.ablations.global_scheduler.enabled false \
    --training.ablations.post_growth_scheduler.enabled true \
    --training.ablations.growth_label_smoothing.enabled false \
    --growth.ablations.noisy_pre_activities_grad.enabled false \
    --growth.ablations.variance_transfer.enabled false

submit_ablation \
    cifar_label_smoothing \
    "/scratch/$USER/dag_experiments/cifar_growth_label_smoothing" \
    cifar_growth_label_smoothing \
    "$INIT_STRATEGIES" \
    --training.label_smoothing "${LABEL_SMOOTHING:-0.1}" \
    --training.ablations.linear_warmup.enabled false \
    --training.ablations.global_scheduler.enabled false \
    --training.ablations.post_growth_scheduler.enabled false \
    --training.ablations.growth_label_smoothing.enabled true \
    --growth.ablations.noisy_pre_activities_grad.enabled false \
    --growth.ablations.variance_transfer.enabled false

submit_ablation \
    cifar_noise5 \
    "/scratch/$USER/dag_experiments/cifar_noisy_pre_grad_005" \
    cifar_noisy_pre_grad_005 \
    "$INIT_STRATEGIES" \
    --training.ablations.linear_warmup.enabled false \
    --training.ablations.global_scheduler.enabled false \
    --training.ablations.post_growth_scheduler.enabled false \
    --training.ablations.growth_label_smoothing.enabled false \
    --growth.ablations.noisy_pre_activities_grad.enabled true \
    --growth.ablations.noisy_pre_activities_grad.std 0.05 \
    --growth.ablations.noisy_pre_activities_grad.relative true \
    --growth.ablations.variance_transfer.enabled false

submit_ablation \
    cifar_noise10 \
    "/scratch/$USER/dag_experiments/cifar_noisy_pre_grad_010" \
    cifar_noisy_pre_grad_010 \
    "$INIT_STRATEGIES" \
    --training.ablations.linear_warmup.enabled false \
    --training.ablations.global_scheduler.enabled false \
    --training.ablations.post_growth_scheduler.enabled false \
    --training.ablations.growth_label_smoothing.enabled false \
    --growth.ablations.noisy_pre_activities_grad.enabled true \
    --growth.ablations.noisy_pre_activities_grad.std 0.10 \
    --growth.ablations.noisy_pre_activities_grad.relative true \
    --growth.ablations.variance_transfer.enabled false

submit_ablation \
    cifar_variance_transfer \
    "/scratch/$USER/dag_experiments/cifar_variance_transfer" \
    cifar_variance_transfer \
    "$INIT_STRATEGIES" \
    --training.ablations.linear_warmup.enabled false \
    --training.ablations.global_scheduler.enabled false \
    --training.ablations.post_growth_scheduler.enabled false \
    --training.ablations.growth_label_smoothing.enabled false \
    --growth.ablations.noisy_pre_activities_grad.enabled false \
    --growth.ablations.variance_transfer.enabled true \
    --growth.ablations.variance_transfer.rescaling "${VARIANCE_TRANSFER_RESCALING:-vt_constraint_old_shape}"

submit_ablation \
    cifar_global_sched \
    "/scratch/$USER/dag_experiments/cifar_global_scheduler_350" \
    cifar_global_scheduler_350 \
    local \
    --training.ablations.linear_warmup.enabled false \
    --training.ablations.post_growth_scheduler.enabled false \
    --training.ablations.global_scheduler.enabled true \
    --training.ablations.global_scheduler.total_epochs 350 \
    --training.ablations.growth_label_smoothing.enabled false \
    --growth.ablations.noisy_pre_activities_grad.enabled false \
    --growth.ablations.variance_transfer.enabled false
