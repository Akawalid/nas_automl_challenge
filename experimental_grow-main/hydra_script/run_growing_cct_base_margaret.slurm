#!/bin/bash

#SBATCH --job-name=growing_cct
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/%u/growing_cct_logs/%x-%j.out
#SBATCH --error=/scratch/%u/growing_cct_logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/StageFrugal}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-$PROJECT_DIR/experimental_grow}"
DATA_DIR="${DATA_DIR:-$HOME/datasets}"
CONFIG_NAME="${CONFIG_NAME:-base_growing_cct_cifar10_margaret}"
EXPERIMENT="${EXPERIMENT:-growing_cct7_3x1_cifar10_${SLURM_JOB_ID:-manual}}"
CONDA_ENV="${CONDA_ENV:-gromo-vit}"
VENV_PATH="${VENV_PATH:-}"
MODULES="${MODULES:-}"
PURGE_MODULES="${PURGE_MODULES:-1}"
USE_UV="${USE_UV:-0}"

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
mkdir -p "$DATA_DIR"
mkdir -p "$HOME/tau_frugal/$USER/hydra_outputs"
mkdir -p "$HOME/tau_frugal/$USER/mlflow"
touch "$HOME/tau_frugal/$USER/mlflow/mlflow.db"

export PYTHONPATH="$PROJECT_DIR/gromo/src:$EXPERIMENT_DIR:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

echo "Job id: ${SLURM_JOB_ID:-manual}"
echo "Host: $(hostname)"
echo "Project: $PROJECT_DIR"
echo "Experiment dir: $EXPERIMENT_DIR"
echo "Config: $CONFIG_NAME"
echo "Dataset: $DATA_DIR"
echo "Experiment: $EXPERIMENT"
echo "Python: $(command -v python || true)"
echo "Use uv: $USE_UV"

nvidia-smi

if [ "$USE_UV" = "1" ]; then
    if command -v uv >/dev/null 2>&1; then
        uv sync
        PYTHON_RUN=(uv run python)
    else
        echo "USE_UV=1 but uv is not available in the job environment."
        exit 1
    fi
else
    PYTHON_RUN=(python)
fi

"${PYTHON_RUN[@]}" -c "import torch, torchvision; print('torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('cuda:', torch.cuda.is_available())"

"${PYTHON_RUN[@]}" -m hydra_script.train_and_grow \
    --config-name="$CONFIG_NAME" \
    dataset_config.path="$DATA_DIR" \
    mlflow.run_name="$EXPERIMENT" \
    "$@"
