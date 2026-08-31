#!/bin/bash

#SBATCH --job-name=cifar_init_noisy
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --array=0-3
#SBATCH --output=/scratch/%u/dag_experiments/cifar_init_noisy_ablation/slurm_logs/%x-%A_%a.out
#SBATCH --error=/scratch/%u/dag_experiments/cifar_init_noisy_ablation/slurm_logs/%x-%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/dev}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-$PROJECT_DIR/experimental_grow}"
GROMO_DIR="${GROMO_DIR:-$PROJECT_DIR/gromo}"
DATA_DIR="${DATA_DIR:-/home/tau/$USER/datasets}"
RUN_DIR="${RUN_DIR:-/scratch/$USER/dag_experiments/cifar_init_noisy_ablation}"
LOGGER_DIR="${LOGGER_DIR:-$RUN_DIR/wandb}"
PIPELINE_TMP_DIR="${PIPELINE_TMP_DIR:-$RUN_DIR/tmp}"
WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$RUN_DIR/.cache/wandb}"
WANDB_DATA_DIR="${WANDB_DATA_DIR:-$RUN_DIR/.local/share/wandb}"
WANDB_ARTIFACT_DIR="${WANDB_ARTIFACT_DIR:-$RUN_DIR/artifacts}"
CONFIG_PATH="${CONFIG_PATH:-experiments/pipeline/experiments_config.yaml}"
CONDA_ENV="${CONDA_ENV:-cct}"
VENV_PATH="${VENV_PATH:-}"
MODULES="${MODULES:-}"
PURGE_MODULES="${PURGE_MODULES:-1}"
USE_UV="${USE_UV:-0}"
GROWTH_STRATEGY="${GROWTH_STRATEGY:-restricted}"
INIT_STRATEGIES="${INIT_STRATEGIES:-local init_scaling_ablation}"
DATASETS="${DATASETS:-cifar10 cifar100}"
SEEDS="${SEEDS:-0}"
ES_ABS_DELTA="${ES_ABS_DELTA:-0.005}"
EXPERIMENT_NAME_PREFIX="${EXPERIMENT_NAME_PREFIX:-cifar_init_noisy_ablation}"
WANDB_PROJECT="${WANDB_PROJECT:-demeter-cifar-init-noisy-ablation}"

normalize_init_strategy() {
    case "$1" in
        init_scaling_ablation|initablation|ablation)
            echo "init_scaling_ablation"
            ;;
        local|local_initialization)
            echo "local"
            ;;
        *)
            echo "$1"
            ;;
    esac
}

read -r -a DATASET_LIST <<< "$DATASETS"
read -r -a SEED_LIST <<< "$SEEDS"
read -r -a INIT_STRATEGY_LIST_RAW <<< "$INIT_STRATEGIES"
INIT_STRATEGY_LIST=()
for raw_strategy in "${INIT_STRATEGY_LIST_RAW[@]}"; do
    INIT_STRATEGY_LIST+=("$(normalize_init_strategy "$raw_strategy")")
done

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
NUM_SEEDS="${#SEED_LIST[@]}"
NUM_INIT="${#INIT_STRATEGY_LIST[@]}"
INIT_INDEX=$((TASK_ID % NUM_INIT))
REST=$((TASK_ID / NUM_INIT))
SEED_INDEX=$((REST % NUM_SEEDS))
DATASET_INDEX=$((REST / NUM_SEEDS))

TOTAL_TASKS=$(( ${#DATASET_LIST[@]} * NUM_SEEDS * NUM_INIT ))
if [ "$DATASET_INDEX" -ge "${#DATASET_LIST[@]}" ]; then
    echo "Invalid SLURM_ARRAY_TASK_ID=$TASK_ID. Expected --array=0-$((TOTAL_TASKS - 1)) for ${#DATASET_LIST[@]} dataset(s) x ${NUM_SEEDS} seed(s) x ${NUM_INIT} init strategy(ies)."
    exit 1
fi

DATASET="${DATASET_LIST[$DATASET_INDEX]}"
SEED="${SEED_LIST[$SEED_INDEX]}"
CURRENT_INIT_STRATEGY="${INIT_STRATEGY_LIST[$INIT_INDEX]}"

if [ "$DATASET" != "cifar10" ] && [ "$DATASET" != "cifar100" ]; then
    echo "Unsupported CIFAR ablation dataset '$DATASET'. Expected cifar10 or cifar100."
    exit 1
fi

if [ "$PURGE_MODULES" = "1" ] && command -v module >/dev/null 2>&1; then
    module purge
fi

if [ -n "$MODULES" ]; then
    if command -v module >/dev/null 2>&1; then
        for module_name in $MODULES; do
            module load "$module_name"
        done
    else
        echo "Requested MODULES='$MODULES', but the module command is not available."
        exit 1
    fi
fi

if [ -n "$VENV_PATH" ]; then
    source "$VENV_PATH/bin/activate"
elif [ -d "$EXPERIMENT_DIR/.venv" ]; then
    source "$EXPERIMENT_DIR/.venv/bin/activate"
else
    if [ -f "$HOME/.local/miniconda3/etc/profile.d/conda.sh" ]; then
        . "$HOME/.local/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        . "$HOME/miniconda3/etc/profile.d/conda.sh"
    else
        source "$HOME/.bashrc"
    fi
    conda activate "$CONDA_ENV"
fi

cd "$EXPERIMENT_DIR"
if [ ! -d "$DATA_DIR" ]; then
    echo "DATA_DIR does not exist: $DATA_DIR"
    exit 1
fi
if [ ! -r "$DATA_DIR" ]; then
    echo "DATA_DIR is not readable: $DATA_DIR"
    exit 1
fi

mkdir -p "$LOGGER_DIR" "$PIPELINE_TMP_DIR" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR" "$WANDB_ARTIFACT_DIR"

export PYTHONPATH="$GROMO_DIR/src:$EXPERIMENT_DIR:${PYTHONPATH:-}"
export WANDB_DIR="$LOGGER_DIR"
export WANDB_CACHE_DIR
export WANDB_DATA_DIR
export WANDB_ARTIFACT_DIR

echo "Job id: ${SLURM_JOB_ID:-manual}"
echo "Array task id: $TASK_ID"
echo "Host: $(hostname)"
echo "Project dir: $PROJECT_DIR"
echo "Experiment dir: $EXPERIMENT_DIR"
echo "Gromo dir: $GROMO_DIR"
echo "Dataset root: $DATA_DIR"
echo "Run dir: $RUN_DIR"
echo "Logger dir: $LOGGER_DIR"
echo "Pipeline tmp dir: $PIPELINE_TMP_DIR"
echo "Dataset: $DATASET"
echo "Seed: $SEED"
echo "Init strategies: ${INIT_STRATEGY_LIST[*]}"
echo "Current init strategy: $CURRENT_INIT_STRATEGY"
echo "Config: $CONFIG_PATH"
echo "Python: $(command -v python || true)"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
fi

if [ "$USE_UV" = "1" ]; then
    if command -v uv >/dev/null 2>&1; then
        PYTHON_RUN=(uv run python)
    else
        echo "USE_UV=1 but uv is not available in the job environment."
        exit 1
    fi
else
    PYTHON_RUN=(python)
fi

"${PYTHON_RUN[@]}" -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available())"

RUN_LABEL="${CURRENT_INIT_STRATEGY}_${DATASET}_seed${SEED}"
RUN_EXPERIMENT_NAME="${EXPERIMENT_NAME_PREFIX}_${RUN_LABEL}"
RUN_LOGGER_DIR="$LOGGER_DIR/$RUN_LABEL"
RUN_PIPELINE_TMP_DIR="$PIPELINE_TMP_DIR/$RUN_LABEL"

mkdir -p "$RUN_LOGGER_DIR" "$RUN_PIPELINE_TMP_DIR"

RUN_PROCESS_TMP_DIR="/tmp/${USER}_${SLURM_JOB_ID:-manual}_${TASK_ID}"
mkdir -p "$RUN_PROCESS_TMP_DIR"
trap "rm -rf \"$RUN_PROCESS_TMP_DIR\"" EXIT

export TMPDIR="$RUN_PROCESS_TMP_DIR"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export PYTHONFAULTHANDLER=1

echo "Process tmp dir: $RUN_PROCESS_TMP_DIR"

"${PYTHON_RUN[@]}" experiments/pipeline/run_pipeline.py \
    --config "$CONFIG_PATH" \
    --experiment.seed "$SEED" \
    --dataset.name "$DATASET" \
    --dataset.path "$DATA_DIR" \
    --dataset.augment true \
    --growth.strategy "$GROWTH_STRATEGY" \
    --training.es_abs_delta "$ES_ABS_DELTA" \
    --logger.exp_name "$WANDB_PROJECT" \
    --experiment.name "$RUN_EXPERIMENT_NAME" \
    --logger.path "$RUN_LOGGER_DIR" \
    --logger.tmpdir "$RUN_PIPELINE_TMP_DIR" \
    --growth.initialization_strategy "$CURRENT_INIT_STRATEGY" \
    "$@"
