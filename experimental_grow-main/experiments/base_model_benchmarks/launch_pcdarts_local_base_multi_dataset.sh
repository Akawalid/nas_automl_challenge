#!/bin/bash

set -euo pipefail

# PC-DARTS benchmark across the NAS datasets used by the Demeter local baseline.
# Scheduling and scratch policy mirror experiments/pipeline/launch_local_base_multi_dataset.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PCDARTS_DIR="${PCDARTS_DIR:-$HOME/dev/DemeterBM/PC-DARTS}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-$REPO_ROOT}"
RUN_DIR="${RUN_DIR:-/scratch/$USER/dag_experiments/base_model_benchmarks/pcdarts_local_base_multi_dataset}"
DATA_DIR="${DATA_DIR:-$RUN_DIR/datasets}"
DATASETS="${DATASETS:-multnist cifartile gutenberg geoclassing chesseract}"
SEEDS="${SEEDS:-0 1 2}"
WANDB_PROJECT="${WANDB_PROJECT:-pcdarts-local-base-multidataset}"
BENCHMARK_NODE="${BENCHMARK_NODE:-margpu010}"
SBATCH_TIME="${SBATCH_TIME:-5-00:10:00}"
CONDA_ENV="${CONDA_ENV:-cct}"
SEARCH_EPOCHS="${SEARCH_EPOCHS:-50}"
EVAL_EPOCHS="${EVAL_EPOCHS:-200}"
STAGGER_SECONDS="${STAGGER_SECONDS:-10}"
INIT_GENOTYPE="${INIT_GENOTYPE:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [ ! -d "$PCDARTS_DIR" ]; then
    echo "PCDARTS_DIR does not exist: $PCDARTS_DIR"
    echo "Expected repo: git clone --branch dev https://github.com/stelladk/PC-DARTS.git $PCDARTS_DIR"
    exit 1
fi

if [ ! -f "$PCDARTS_DIR/custom_train_search.py" ]; then
    echo "custom_train_search.py not found under PCDARTS_DIR: $PCDARTS_DIR"
    echo "Use the stelladk/PC-DARTS dev branch."
    exit 1
fi

if [ ! -f "$PCDARTS_DIR/dataset.py" ]; then
    echo "PC-DARTS dataset.py not found. Use the stelladk/PC-DARTS dev branch."
    exit 1
fi

if ! grep -q -- "--eval_epochs" "$PCDARTS_DIR/custom_train_search.py"; then
    echo "custom_train_search.py does not expose --eval_epochs. Use the stelladk/PC-DARTS dev branch."
    exit 1
fi

if ! grep -q -- "in_channels=in_channels" "$PCDARTS_DIR/custom_train_search.py"; then
    echo "custom_train_search.py is not the multi-channel version needed for Chesseract."
    echo "Use the stelladk/PC-DARTS dev branch."
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
        echo "DATA_DIR must be under /scratch because PC-DARTS may download/extract datasets: $DATA_DIR"
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
    "$RUN_DIR/.cache/xdg" "$RUN_DIR/.config/wandb" "$RUN_DIR/.config/matplotlib" \
    "$RUN_DIR/pycache" "$RUN_DIR/torch" "$RUN_DIR/cuda_cache" \
    "$RUN_DIR/locks" "$DATA_DIR"

SBATCH_ARGS=(
    --parsable
    --job-name=pcdarts_local_base_multi_dataset
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

echo "Submitting PC-DARTS local baseline multi-dataset sweep"
echo "PC-DARTS dir: $PCDARTS_DIR"
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
echo "Init genotype: ${INIT_GENOTYPE:-none}"
echo "Extra args: ${EXTRA_ARGS:-none}"

JOB_ID=$(
    PCDARTS_DIR="$PCDARTS_DIR" \
    EXPERIMENT_DIR="$EXPERIMENT_DIR" \
    DATA_DIR="$DATA_DIR" \
    RUN_DIR="$RUN_DIR" \
    DATASETS="$DATASETS" \
    SEEDS="$SEEDS" \
    WANDB_PROJECT="$WANDB_PROJECT" \
    CONDA_ENV="$CONDA_ENV" \
    SEARCH_EPOCHS="$SEARCH_EPOCHS" \
    EVAL_EPOCHS="$EVAL_EPOCHS" \
    STAGGER_SECONDS="$STAGGER_SECONDS" \
    INIT_GENOTYPE="$INIT_GENOTYPE" \
    EXTRA_ARGS="$EXTRA_ARGS" \
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
                BATCH_SIZE=128
                ;;
            cifartile|geoclassing)
                BATCH_SIZE=16
                ;;
            gutenberg|chesseract)
                BATCH_SIZE=128
                ;;
            *)
                echo "Unsupported dataset for PC-DARTS benchmark: $DATASET"
                exit 1
                ;;
        esac

        SLEEP_TIME=$((SLURM_ARRAY_TASK_ID * STAGGER_SECONDS))
        echo "Array task ${SLURM_ARRAY_TASK_ID}: sleeping ${SLEEP_TIME}s before starting"
        sleep "$SLEEP_TIME"

        RUN_LABEL="pcdarts_${DATASET}_seed${SEED}"
        WORK_DIR="$RUN_DIR/work/$RUN_LABEL"
        TMP_DIR="$RUN_DIR/tmp/$RUN_LABEL"
        PROCESS_TMP_DIR="/tmp/${USER}_${SLURM_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}"
        TMP_PREFIX="${TMP_DIR%/}/"
        LOCK_FILE="$RUN_DIR/locks/${DATASET}.lock"

        mkdir -p "$WORK_DIR" "$TMP_DIR" "$DATA_DIR" "$RUN_DIR/wandb" \
            "$RUN_DIR/.cache/wandb" "$RUN_DIR/artifacts" "$RUN_DIR/.cache/xdg" \
            "$RUN_DIR/.config/wandb" "$RUN_DIR/.config/matplotlib" "$RUN_DIR/pycache" \
            "$RUN_DIR/torch" "$RUN_DIR/cuda_cache" "$RUN_DIR/locks" "$PROCESS_TMP_DIR"

        trap "rm -rf \"$PROCESS_TMP_DIR\"" EXIT

        cd "$WORK_DIR"

        export PYTHONPATH="$EXPERIMENT_DIR:$PCDARTS_DIR:${PYTHONPATH:-}"
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
        echo "PC-DARTS dir: $PCDARTS_DIR"
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

        PREFETCH_CMD=(
            python -c "import os, argparse; from dataset import get_dataset; args=argparse.Namespace(dataset=os.environ[\"DATASET\"], data=os.environ[\"DATA_DIR\"], image_size=None, num_classes=None, cutout=False, cutout_length=16, no_augment=False, grayscale=False, dataset_mean=None, dataset_std=None, data_augmentation=None); train_data, test_data, n_classes, in_channels = get_dataset(args); print(f\"prefetched dataset={args.dataset} train={len(train_data)} test={len(test_data)} classes={n_classes} channels={in_channels}\")"
        )
        if command -v flock >/dev/null 2>&1; then
            flock "$LOCK_FILE" "${PREFETCH_CMD[@]}"
        else
            "${PREFETCH_CMD[@]}"
        fi

        CMD=(
            python "$PCDARTS_DIR/custom_train_search.py"
            --dataset "$DATASET"
            --data "$DATA_DIR"
            --batch_size "$BATCH_SIZE"
            --eval_batch_size "$BATCH_SIZE"
            --epochs "$SEARCH_EPOCHS"
            --eval_epochs "$EVAL_EPOCHS"
            --seed "$SEED"
            --gpu 0
            --api wandb
            --exp_name "$WANDB_PROJECT"
            --log_path "$RUN_DIR/wandb"
            --tmpdir "$TMP_PREFIX"
        )

        if [ -n "$INIT_GENOTYPE" ]; then
            CMD+=(--init_genotype "$INIT_GENOTYPE")
        fi

        if [ -n "$EXTRA_ARGS" ]; then
            read -r -a EXTRA_ARG_LIST <<< "$EXTRA_ARGS"
            CMD+=("${EXTRA_ARG_LIST[@]}")
        fi

        printf "Command:"
        printf " %q" "${CMD[@]}"
        printf "\n"
        "${CMD[@]}"
    '
)

echo "JOB_ID=$JOB_ID"
