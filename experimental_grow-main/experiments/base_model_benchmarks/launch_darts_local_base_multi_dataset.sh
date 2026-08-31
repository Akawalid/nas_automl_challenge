#!/bin/bash

set -euo pipefail

# DARTS benchmark across the NAS datasets used by the Demeter local baseline.
# This mirrors experiments/pipeline/launch_local_base_multi_dataset.sh for
# scheduling: gpu partition, margpu010 by default, scratch run directory.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DARTS_DIR="${DARTS_DIR:-$HOME/dev/DemeterBM/DARTS}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-$REPO_ROOT}"
RUN_DIR="${RUN_DIR:-/scratch/$USER/dag_experiments/base_model_benchmarks/darts_local_base_multi_dataset}"
DATA_DIR="${DATA_DIR:-$RUN_DIR/datasets}"
DATASETS="${DATASETS:-multnist cifartile gutenberg geoclassing chesseract}"
SEEDS="${SEEDS:-0 1 2}"
WANDB_PROJECT="${WANDB_PROJECT:-darts-local-base-multidataset}"
BENCHMARK_NODE="${BENCHMARK_NODE:-margpu010}"
SBATCH_TIME="${SBATCH_TIME:-5-00:10:00}"
CONDA_ENV="${CONDA_ENV:-cct}"
SEARCH_EPOCHS="${SEARCH_EPOCHS:-50}"
EVAL_EPOCHS="${EVAL_EPOCHS:-200}"
WORKERS="${WORKERS:-4}"

if [ ! -d "$DARTS_DIR" ]; then
    echo "DARTS_DIR does not exist: $DARTS_DIR"
    exit 1
fi

if [ ! -f "$DARTS_DIR/search.py" ]; then
    echo "search.py not found under DARTS_DIR: $DARTS_DIR"
    exit 1
fi

if [ ! -f "$DARTS_DIR/datasets.py" ]; then
    echo "DARTS datasets.py not found. Use the stelladk/DARTS dev branch."
    exit 1
fi

if ! grep -q -- "--eval_epochs" "$DARTS_DIR/config.py"; then
    echo "DARTS config.py does not expose --eval_epochs. Use the stelladk/DARTS dev branch."
    exit 1
fi

if [ ! -f "$EXPERIMENT_DIR/tools/augmentations.py" ]; then
    echo "Expected Demeter augmentations not found: $EXPERIMENT_DIR/tools/augmentations.py"
    exit 1
fi

case "$RUN_DIR" in
    /scratch/*)
        ;;
    *)
        echo "RUN_DIR must be under /scratch to avoid writing generated outputs elsewhere: $RUN_DIR"
        exit 1
        ;;
esac

case "$DATA_DIR" in
    /scratch/*)
        ;;
    *)
        echo "DATA_DIR must be under /scratch because DARTS may download/extract datasets: $DATA_DIR"
        exit 1
        ;;
esac

if [ ! -d /scratch ]; then
    echo "/scratch is not mounted on this host. Submit from the cluster host with scratch mounted."
    exit 1
fi

read -r -a DATASET_LIST <<< "$DATASETS"
read -r -a SEED_LIST <<< "$SEEDS"

TOTAL_TASKS=$(( ${#DATASET_LIST[@]} * ${#SEED_LIST[@]} ))
if [ "$TOTAL_TASKS" -le 0 ]; then
    echo "No tasks to submit. Check DATASETS and SEEDS."
    exit 1
fi
ARRAY_RANGE="${ARRAY_RANGE:-0-$((TOTAL_TASKS - 1))}"

mkdir -p "$RUN_DIR/slurm_logs" "$RUN_DIR/wandb" "$RUN_DIR/work" \
    "$RUN_DIR/tmp" "$RUN_DIR/.cache/wandb" "$RUN_DIR/artifacts" \
    "$RUN_DIR/.cache/xdg" "$RUN_DIR/.config/wandb" "$RUN_DIR/.config/matplotlib" "$RUN_DIR/pycache" \
    "$RUN_DIR/torch" "$RUN_DIR/cuda_cache" "$DATA_DIR"

SBATCH_ARGS=(
    --parsable
    --job-name=darts_local_base_multi_dataset
    --partition=gpu
    --gres=gpu:1
    --ntasks=1
    --cpus-per-task=8
    --time="$SBATCH_TIME"
    --array="$ARRAY_RANGE"
    --output="$RUN_DIR/slurm_logs/%x-%A_%a.out"
    --error="$RUN_DIR/slurm_logs/%x-%A_%a.err"
)

if [ -n "$BENCHMARK_NODE" ]; then
    SBATCH_ARGS+=(--nodelist="$BENCHMARK_NODE")
fi

echo "Submitting DARTS local baseline multi-dataset sweep"
echo "DARTS dir: $DARTS_DIR"
echo "Experiment dir: $EXPERIMENT_DIR"
echo "Datasets: $DATASETS"
echo "Seeds: $SEEDS"
echo "Total array tasks: $TOTAL_TASKS"
echo "Submitted array range: $ARRAY_RANGE"
echo "Benchmark node: ${BENCHMARK_NODE:-none}"
echo "Run dir: $RUN_DIR"
echo "Dataset dir: $DATA_DIR"
echo "W&B project: $WANDB_PROJECT"
echo "Search epochs: $SEARCH_EPOCHS"
echo "Eval epochs: $EVAL_EPOCHS"

JOB_ID=$(
    DARTS_DIR="$DARTS_DIR" \
    EXPERIMENT_DIR="$EXPERIMENT_DIR" \
    DATA_DIR="$DATA_DIR" \
    RUN_DIR="$RUN_DIR" \
    DATASETS="$DATASETS" \
    SEEDS="$SEEDS" \
    WANDB_PROJECT="$WANDB_PROJECT" \
    CONDA_ENV="$CONDA_ENV" \
    SEARCH_EPOCHS="$SEARCH_EPOCHS" \
    EVAL_EPOCHS="$EVAL_EPOCHS" \
    WORKERS="$WORKERS" \
    sbatch "${SBATCH_ARGS[@]}" --wrap='
        set -euo pipefail

        if [ -f "$HOME/.local/miniconda3/etc/profile.d/conda.sh" ]; then
            . "$HOME/.local/miniconda3/etc/profile.d/conda.sh"
        elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
            . "$HOME/miniconda3/etc/profile.d/conda.sh"
        else
            source "$HOME/.bashrc"
        fi
        conda activate "$CONDA_ENV"

        read -r -a DATASET_LIST <<< "$DATASETS"
        read -r -a SEED_LIST <<< "$SEEDS"

        DATASET_INDEX=$((SLURM_ARRAY_TASK_ID / ${#SEED_LIST[@]}))
        SEED_INDEX=$((SLURM_ARRAY_TASK_ID % ${#SEED_LIST[@]}))

        DATASET="${DATASET_LIST[$DATASET_INDEX]}"
        SEED="${SEED_LIST[$SEED_INDEX]}"

        case "$DATASET" in
            multnist)
                BATCH_SIZE=56
                ;;
            cifartile|geoclassing)
                BATCH_SIZE=16
                ;;
            gutenberg|chesseract)
                BATCH_SIZE=128
                ;;
            *)
                echo "Unsupported dataset for DARTS benchmark: $DATASET"
                exit 1
                ;;
        esac

        RUN_LABEL="darts_${DATASET}_seed${SEED}"
        WORK_DIR="$RUN_DIR/work/$RUN_LABEL"
        TMP_DIR="$RUN_DIR/tmp/$RUN_LABEL"
        PROCESS_TMP_DIR="/tmp/${USER}_${SLURM_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}"

        mkdir -p "$WORK_DIR" "$TMP_DIR" "$DATA_DIR" "$RUN_DIR/wandb" \
            "$RUN_DIR/.cache/wandb" "$RUN_DIR/artifacts" "$RUN_DIR/.cache/xdg" \
            "$RUN_DIR/.config/wandb" "$RUN_DIR/.config/matplotlib" "$RUN_DIR/pycache" "$RUN_DIR/torch" \
            "$RUN_DIR/cuda_cache" "$PROCESS_TMP_DIR"

        trap "rm -rf \"$PROCESS_TMP_DIR\"" EXIT

        cd "$WORK_DIR"

        export PYTHONPATH="$EXPERIMENT_DIR:$DARTS_DIR:${PYTHONPATH:-}"
        export PYTHONDONTWRITEBYTECODE=1
        export PYTHONPYCACHEPREFIX="$RUN_DIR/pycache"
        export XDG_CACHE_HOME="$RUN_DIR/.cache/xdg"
        export XDG_CONFIG_HOME="$RUN_DIR/.config"
        export MPLCONFIGDIR="$RUN_DIR/.config/matplotlib"
        export TMPDIR="$PROCESS_TMP_DIR"
        export TMP="$TMPDIR"
        export TEMP="$TMPDIR"
        export PYTHONFAULTHANDLER=1
        export TORCH_HOME="$RUN_DIR/torch"
        export CUDA_CACHE_PATH="$RUN_DIR/cuda_cache"
        export WANDB_DIR="$RUN_DIR/wandb"
        export WANDB_CACHE_DIR="$RUN_DIR/.cache/wandb"
        export WANDB_CONFIG_DIR="$RUN_DIR/.config/wandb"
        export WANDB_ARTIFACT_DIR="$RUN_DIR/artifacts"
        export WANDB_DATA_DIR="$RUN_DIR/wandb"

        echo "Job id: ${SLURM_JOB_ID:-manual}"
        echo "Array task id: ${SLURM_ARRAY_TASK_ID:-manual}"
        echo "Host: $(hostname)"
        echo "DARTS dir: $DARTS_DIR"
        echo "Experiment dir: $EXPERIMENT_DIR"
        echo "Dataset root: $DATA_DIR"
        echo "Run dir: $RUN_DIR"
        echo "Work dir: $WORK_DIR"
        echo "Tmp dir: $TMP_DIR"
        echo "Process tmp dir: $PROCESS_TMP_DIR"
        echo "Dataset: $DATASET"
        echo "Seed: $SEED"
        echo "Batch size: $BATCH_SIZE"
        echo "Python: $(command -v python || true)"

        if command -v nvidia-smi >/dev/null 2>&1; then
            nvidia-smi
        fi

        python -c "import torch; print(\"torch:\", torch.__version__); print(\"cuda:\", torch.cuda.is_available())"

        python "$DARTS_DIR/search.py" \
            --name "$RUN_LABEL" \
            --dataset "$DATASET" \
            --data "$DATA_DIR" \
            --batch_size "$BATCH_SIZE" \
            --epochs "$SEARCH_EPOCHS" \
            --eval_epochs "$EVAL_EPOCHS" \
            --seed "$SEED" \
            --gpus 0 \
            --workers "$WORKERS" \
            --api wandb \
            --exp_name "$WANDB_PROJECT" \
            --log_path "$RUN_DIR/wandb" \
            --tmpdir "$TMP_DIR"
    '
)

echo "JOB_ID=$JOB_ID"
