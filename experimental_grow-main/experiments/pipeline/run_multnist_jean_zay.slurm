#!/bin/bash

# Dedicated Jean Zay worker for the five-seed MultiNIST array.
# Resource selection is supplied by launch_jean_zay_multnist_5seeds.sh, so this
# file intentionally contains no Margaret partition, node, or path directives.

set -euo pipefail

EXPERIMENT_DIR="${EXPERIMENT_DIR:?EXPERIMENT_DIR is required}"
GROMO_DIR="${GROMO_DIR:?GROMO_DIR is required}"
CONDA_ENV="${CONDA_ENV:?CONDA_ENV is required}"
DATA_DIR="${DATA_DIR:?DATA_DIR is required}"
RUN_DIR="${RUN_DIR:?RUN_DIR is required}"
LOGGER_DIR="${LOGGER_DIR:-$RUN_DIR/wandb}"
PIPELINE_TMP_DIR="${PIPELINE_TMP_DIR:-$RUN_DIR/tmp}"
CONFIG_PATH="${CONFIG_PATH:-experiments/pipeline/experiments_config.yaml}"
MODULES="${MODULES:?MODULES is required}"
SEEDS="${SEEDS:-0 1 2 3 4}"
EXPERIMENT_NAME_PREFIX="${EXPERIMENT_NAME_PREFIX:-local_base_multi_dataset}"
WANDB_PROJECT="${WANDB_PROJECT:-demeter-local-base-multnist-jean-zay}"

module purge
for module_name in $MODULES; do
    module load "$module_name"
done

if ! command -v conda >/dev/null 2>&1; then
    echo "Conda is unavailable after loading: $MODULES"
    exit 1
fi
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

read -r -a SEED_LIST <<< "$SEEDS"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$TASK_ID" -ge "${#SEED_LIST[@]}" ]; then
    echo "Array task $TASK_ID has no matching seed in: $SEEDS"
    exit 1
fi
SEED="${SEED_LIST[$TASK_ID]}"

if [ ! -f "$DATA_DIR/MultNIST.zip" ] || [ ! -d "$DATA_DIR/MultNIST_extracted" ]; then
    echo "MultiNIST cache is incomplete under $DATA_DIR"
    exit 1
fi

RUN_LABEL="local_multnist_seed${SEED}"
RUN_LOGGER_DIR="$LOGGER_DIR/$RUN_LABEL"
RUN_TMP_DIR="$PIPELINE_TMP_DIR/$RUN_LABEL"
mkdir -p "$RUN_LOGGER_DIR" "$RUN_TMP_DIR" "$RUN_DIR/.cache/wandb" \
    "$RUN_DIR/artifacts" "$RUN_DIR/torch" "$RUN_DIR/cuda_cache"

export PYTHONPATH="$GROMO_DIR/src:$EXPERIMENT_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export WANDB_DIR="$RUN_LOGGER_DIR"
export WANDB_CACHE_DIR="$RUN_DIR/.cache/wandb"
export WANDB_ARTIFACT_DIR="$RUN_DIR/artifacts"
export TORCH_HOME="$RUN_DIR/torch"
export CUDA_CACHE_PATH="$RUN_DIR/cuda_cache"
export TMPDIR="$RUN_TMP_DIR"

cd "$EXPERIMENT_DIR"

echo "Job id: ${SLURM_JOB_ID:-manual}"
echo "Array task: $TASK_ID"
echo "Host: $(hostname)"
echo "Seed: $SEED"
echo "Dataset: multnist"
echo "Environment: $CONDA_ENV"
echo "Data: $DATA_DIR"
echo "Output: $RUN_DIR"
echo "Python: $(command -v python)"
module list
nvidia-smi
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); assert torch.cuda.is_available()"

python experiments/pipeline/run_pipeline.py \
    --config "$CONFIG_PATH" \
    --experiment.seed "$SEED" \
    --experiment.name "${EXPERIMENT_NAME_PREFIX}_${RUN_LABEL}" \
    --dataset.name multnist \
    --dataset.path "$DATA_DIR" \
    --growth.strategy restricted \
    --growth.initialization_strategy local \
    --training.es_abs_delta 0.00122 \
    --logger.exp_name "$WANDB_PROJECT" \
    --logger.path "$RUN_LOGGER_DIR" \
    --logger.tmpdir "$RUN_TMP_DIR" \
    "$@"
