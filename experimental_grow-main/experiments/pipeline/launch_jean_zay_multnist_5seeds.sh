#!/bin/bash

set -euo pipefail

# Launch the local-growth MultiNIST baseline for seeds 0..4 on Jean Zay.
# Defaults target one H100 per array task. GPU_TYPE=a100 or GPU_TYPE=v100 can
# be used when the active IDRIS project has hours only on those partitions.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

: "${IDRPROJ:?IDRPROJ is not set. Activate the intended Jean Zay project first.}"
: "${WORK:?WORK is not set. Activate an IDRIS project with a WORK space.}"
: "${SCRATCH:?SCRATCH is not set. Activate an IDRIS project with a SCRATCH space.}"

GPU_TYPE="${GPU_TYPE:-h100}"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$REPO_ROOT/.." && pwd)}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-$REPO_ROOT}"
GROMO_DIR="${GROMO_DIR:-$PROJECT_DIR/gromo}"
CONDA_ENV="${CONDA_ENV:-$WORK/envs/demeter}"
DATA_DIR="${DATA_DIR:-$WORK/datasets/multnist}"
RUN_DIR="${RUN_DIR:-$SCRATCH/demeter/local_base_multnist}"
CONFIG_PATH="${CONFIG_PATH:-experiments/pipeline/experiments_config.yaml}"
SEEDS="${SEEDS:-0 1 2 3 4}"
ARRAY_RANGE="${ARRAY_RANGE:-0-4%5}"
MAX_PARALLEL="${MAX_PARALLEL:-5}"

case "$GPU_TYPE" in
    h100)
        ACCOUNT="${ACCOUNT:-$IDRPROJ@h100}"
        PARTITION="${PARTITION:-gpu_p6}"
        CONSTRAINT="${CONSTRAINT:-h100}"
        QOS="${QOS:-qos_gpu_h100-t4}"
        SBATCH_TIME="${SBATCH_TIME:-48:00:00}"
        CPUS_PER_TASK="${CPUS_PER_TASK:-24}"
        MODULES="${MODULES:-arch/h100 miniforge/24.9.0}"
        ;;
    a100)
        ACCOUNT="${ACCOUNT:-$IDRPROJ@a100}"
        PARTITION="${PARTITION:-gpu_p5}"
        CONSTRAINT="${CONSTRAINT:-a100}"
        QOS="${QOS:-qos_gpu_a100-t3}"
        SBATCH_TIME="${SBATCH_TIME:-20:00:00}"
        CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
        MODULES="${MODULES:-arch/a100 miniforge/24.9.0}"
        ;;
    v100)
        ACCOUNT="${ACCOUNT:-$IDRPROJ@v100}"
        PARTITION="${PARTITION:-gpu_p13}"
        CONSTRAINT="${CONSTRAINT:-v100-32g}"
        QOS="${QOS:-qos_gpu-t4}"
        SBATCH_TIME="${SBATCH_TIME:-48:00:00}"
        CPUS_PER_TASK="${CPUS_PER_TASK:-10}"
        MODULES="${MODULES:-miniforge/24.9.0}"
        ;;
    *)
        echo "Unsupported GPU_TYPE=$GPU_TYPE (expected h100, a100, or v100)."
        exit 1
        ;;
esac

if [ ! -d "$GROMO_DIR/src/gromo" ]; then
    echo "Gromo checkout not found: $GROMO_DIR"
    exit 1
fi
if [ ! -d "$CONDA_ENV" ]; then
    echo "Conda environment not found: $CONDA_ENV"
    exit 1
fi
if [ ! -f "$DATA_DIR/MultNIST.zip" ] || [ ! -d "$DATA_DIR/MultNIST_extracted" ]; then
    echo "MultiNIST is not prefetched under $DATA_DIR."
    echo "Follow README_JEAN_ZAY_MULTNIST.md before submitting."
    exit 1
fi

mkdir -p "$RUN_DIR/slurm_logs"

# Keep ARRAY_RANGE overridable for a one-task smoke test. For the normal five
# seeds, MAX_PARALLEL controls how many GPUs may run concurrently.
if [ "$ARRAY_RANGE" = "0-4%5" ] && [ "$MAX_PARALLEL" != "5" ]; then
    ARRAY_RANGE="0-4%$MAX_PARALLEL"
fi

echo "Submitting MultiNIST seeds: $SEEDS"
echo "GPU/account: $GPU_TYPE / $ACCOUNT"
echo "Partition/constraint: $PARTITION / $CONSTRAINT"
echo "QoS/time: $QOS / $SBATCH_TIME"
echo "Array: $ARRAY_RANGE"
echo "Environment: $CONDA_ENV"
echo "Data: $DATA_DIR"
echo "Outputs: $RUN_DIR"

PROJECT_DIR="$PROJECT_DIR" \
EXPERIMENT_DIR="$EXPERIMENT_DIR" \
GROMO_DIR="$GROMO_DIR" \
CONDA_ENV="$CONDA_ENV" \
DATA_DIR="$DATA_DIR" \
RUN_DIR="$RUN_DIR" \
LOGGER_DIR="$RUN_DIR/wandb" \
PIPELINE_TMP_DIR="$RUN_DIR/tmp" \
CONFIG_PATH="$CONFIG_PATH" \
DATASETS="multnist" \
SEEDS="$SEEDS" \
INIT_STRATEGIES="local" \
EXPERIMENT_NAME_PREFIX="local_base_multi_dataset" \
WANDB_PROJECT="demeter-local-base-multnist-jean-zay" \
WANDB_MODE="${WANDB_MODE:-offline}" \
MODULES="$MODULES" \
PURGE_MODULES=1 \
USE_UV=0 \
sbatch --parsable \
    --job-name=multnist_local_5seeds \
    --account="$ACCOUNT" \
    --partition="$PARTITION" \
    --constraint="$CONSTRAINT" \
    --qos="$QOS" \
    --nodes=1 \
    --ntasks=1 \
    --gres=gpu:1 \
    --cpus-per-task="$CPUS_PER_TASK" \
    --hint=nomultithread \
    --time="$SBATCH_TIME" \
    --array="$ARRAY_RANGE" \
    --output="$RUN_DIR/slurm_logs/%x-%A_%a.out" \
    --error="$RUN_DIR/slurm_logs/%x-%A_%a.err" \
    "$SCRIPT_DIR/run_multnist_jean_zay.slurm" \
    "$@"
