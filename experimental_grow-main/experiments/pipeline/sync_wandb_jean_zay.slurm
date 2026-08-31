#!/bin/bash

# Sync offline W&B runs after the Jean Zay training array finishes. This job
# runs on prepost because Jean Zay GPU compute nodes do not have Internet.

set -euo pipefail

CONDA_ENV="${CONDA_ENV:?CONDA_ENV is required}"
RUN_DIR="${RUN_DIR:?RUN_DIR is required}"
WANDB_ROOT="$RUN_DIR/wandb"

module purge
module load miniforge/24.9.0
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

if ! command -v wandb >/dev/null 2>&1; then
    echo "wandb is not installed in $CONDA_ENV"
    exit 1
fi
if [ ! -d "$WANDB_ROOT" ]; then
    echo "W&B directory does not exist: $WANDB_ROOT"
    exit 1
fi

mapfile -d '' OFFLINE_RUNS < <(
    find "$WANDB_ROOT" -type d -name 'offline-run-*' -print0
)

if [ "${#OFFLINE_RUNS[@]}" -eq 0 ]; then
    echo "No offline W&B runs found under $WANDB_ROOT"
    exit 0
fi

echo "Found ${#OFFLINE_RUNS[@]} offline W&B run(s)."
wandb status

for run_dir in "${OFFLINE_RUNS[@]}"; do
    echo "Syncing $run_dir"
    wandb sync "$run_dir"
done

echo "W&B synchronization completed."
