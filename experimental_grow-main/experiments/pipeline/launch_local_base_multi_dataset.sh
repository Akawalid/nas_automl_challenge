#!/bin/bash

set -euo pipefail

# Local baseline across the NAS datasets.
# Default for the currently visible Margaret02 GPU partition: keep benchmark on
# margpu010 and let CIFAR use the other visible nodes.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATASETS="${DATASETS:-multnist cifartile gutenberg geoclassing chesseract}"
SEEDS="${SEEDS:-0 1 2 3 4}"
INIT_STRATEGIES="${INIT_STRATEGIES:-local}"
EXPERIMENT_NAME_PREFIX="${EXPERIMENT_NAME_PREFIX:-local_base_multi_dataset}"
WANDB_PROJECT="${WANDB_PROJECT:-demeter-local-base-multidataset}"
RUN_DIR="${RUN_DIR:-/scratch/$USER/dag_experiments/local_base_multi_dataset}"
SCRIPT_PATH="${SCRIPT_PATH:-$SCRIPT_DIR/run_init_scaling_ablation.slurm}"
BENCHMARK_NODE="${BENCHMARK_NODE:-margpu008}"
BENCHMARK_ALLOWED_NODES="${BENCHMARK_ALLOWED_NODES:-}"
BENCHMARK_EXCLUDE_NODES="${BENCHMARK_EXCLUDE_NODES:-}"
BENCHMARK_EXCLUDE_FALLBACK="${BENCHMARK_EXCLUDE_FALLBACK:-margpu001,margpu002,margpu003,margpu004,margpu005,margpu006,margpu007,margpu008,margpu009,margpu010,margpu011,margpu012,margpu013,margpu017,margpu018,margpu024}"
SBATCH_TIME="${SBATCH_TIME:-}"

if [[ "$SCRIPT_PATH" != /* ]]; then
    SCRIPT_PATH="$REPO_ROOT/$SCRIPT_PATH"
fi

if [ ! -f "$SCRIPT_PATH" ]; then
    echo "SLURM script not found: $SCRIPT_PATH"
    echo "Set SCRIPT_PATH to the correct run_init_scaling_ablation.slurm path."
    exit 1
fi

cd "$REPO_ROOT"

read -r -a DATASET_LIST <<< "$DATASETS"
read -r -a SEED_LIST <<< "$SEEDS"
read -r -a INIT_STRATEGY_LIST <<< "$INIT_STRATEGIES"

build_exclude_nodes() {
    if ! command -v sinfo >/dev/null 2>&1 || ! command -v scontrol >/dev/null 2>&1; then
        echo "$BENCHMARK_EXCLUDE_FALLBACK"
        return
    fi

    local allowed_nodes=()
    local gpu_nodes=()
    local host_expr
    local node
    local allowed_node
    local keep
    local excluded_nodes=()

    mapfile -t allowed_nodes < <(scontrol show hostnames "$BENCHMARK_ALLOWED_NODES" | sort -u)
    mapfile -t gpu_nodes < <(
        sinfo -h -p gpu -o "%N" | while read -r host_expr; do
            [ -n "$host_expr" ] && scontrol show hostnames "$host_expr"
        done | sort -u
    )

    for node in "${gpu_nodes[@]}"; do
        keep=0
        for allowed_node in "${allowed_nodes[@]}"; do
            if [ "$node" = "$allowed_node" ]; then
                keep=1
                break
            fi
        done
        if [ "$keep" -eq 0 ]; then
            excluded_nodes+=("$node")
        fi
    done

    local IFS=,
    echo "${excluded_nodes[*]}"
}

TOTAL_TASKS=$(( ${#DATASET_LIST[@]} * ${#SEED_LIST[@]} * ${#INIT_STRATEGY_LIST[@]} ))
if [ "$TOTAL_TASKS" -le 0 ]; then
    echo "No tasks to submit. Check DATASETS, SEEDS, and INIT_STRATEGIES."
    exit 1
fi
ARRAY_RANGE="${ARRAY_RANGE:-0-$((TOTAL_TASKS - 1))}"

mkdir -p "$RUN_DIR/slurm_logs"

SBATCH_ARGS=(
    --parsable
    --job-name=local_base_multi_dataset
    --array="$ARRAY_RANGE"
    --output="$RUN_DIR/slurm_logs/%x-%A_%a.out"
    --error="$RUN_DIR/slurm_logs/%x-%A_%a.err"
)

if [ -n "$SBATCH_TIME" ]; then
    SBATCH_ARGS+=(--time="$SBATCH_TIME")
fi

if [ -n "$BENCHMARK_NODE" ]; then
    SBATCH_ARGS+=(--nodelist="$BENCHMARK_NODE")
elif [ -n "$BENCHMARK_ALLOWED_NODES" ]; then
    if [ -z "$BENCHMARK_EXCLUDE_NODES" ]; then
        BENCHMARK_EXCLUDE_NODES="$(build_exclude_nodes)"
    fi
    if [ -n "$BENCHMARK_EXCLUDE_NODES" ]; then
        SBATCH_ARGS+=(--exclude="$BENCHMARK_EXCLUDE_NODES")
    fi
fi

echo "Submitting local baseline multi-dataset sweep"
echo "Datasets: $DATASETS"
echo "Seeds: $SEEDS"
echo "Init strategies: $INIT_STRATEGIES"
echo "Total array tasks: $TOTAL_TASKS"
echo "Submitted array range: $ARRAY_RANGE"
echo "Requested wall time: ${SBATCH_TIME:-from slurm script}"
echo "Benchmark node: ${BENCHMARK_NODE:-none}"
echo "Allowed benchmark nodes: ${BENCHMARK_ALLOWED_NODES:-any}"
echo "Excluded non-benchmark nodes: ${BENCHMARK_EXCLUDE_NODES:-none}"
echo "Run dir: $RUN_DIR"
echo "W&B project: $WANDB_PROJECT"

JOB_ID=$(
    DATASETS="$DATASETS" \
    SEEDS="$SEEDS" \
    INIT_STRATEGIES="$INIT_STRATEGIES" \
    EXPERIMENT_NAME_PREFIX="$EXPERIMENT_NAME_PREFIX" \
    WANDB_PROJECT="$WANDB_PROJECT" \
    RUN_DIR="$RUN_DIR" \
    sbatch "${SBATCH_ARGS[@]}" "$SCRIPT_PATH"
)

echo "JOB_ID=$JOB_ID"
