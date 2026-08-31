#!/bin/bash

#SBATCH --output=/home/tau/%u/tau_frugal/%u/logs/slurm-%x-%j.out
#SBATCH --error=/home/tau/%u/tau_frugal/%u/logs/slurm-%x-%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=6
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=tau
#SBATCH --exclude=margpu0[18,21,24-27,29]

nvidia-smi
uv sync

uv run python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

# Create logs directory if it doesn't exist
COMMAND="mkdir -p /home/tau/${USER}/tau_frugal/${USER}/logs"
echo ${COMMAND}
eval ${COMMAND}

# Get git commit and check for uncommitted changes in current workdir
echo "=== Current workdir git status ==="
git log -1 --oneline
git status --short

# Get git commit and check for uncommitted changes in gromo module
GROMO_PATH=$(uv run python -c "import gromo; import os; print(os.path.dirname(os.path.dirname(os.path.dirname(gromo.__file__))))" 2>/dev/null)
if [ -n "$GROMO_PATH" ] && [ -d "$GROMO_PATH/.git" ]; then
    echo ""
    echo "=== gromo module git status (at $GROMO_PATH) ==="
    git -C "$GROMO_PATH" log -1 --oneline
    git -C "$GROMO_PATH" status --short
fi

echo ""
echo "Starting job ${SLURM_JOB_ID} on hostname: $(hostname) / SLURMD_NODENAME: ${SLURMD_NODENAME} at $(date)"

# Source Conda's initialization script
. "${HOME}/miniconda3/etc/profile.d/conda.sh"  # Adjust path if using Anaconda or another location

conda activate $ENV_NAME

COMMAND="uv run python -m hydra_script.train_and_grow $@"
echo ${COMMAND}
eval ${COMMAND}
