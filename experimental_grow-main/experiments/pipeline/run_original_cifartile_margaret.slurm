#!/bin/bash
#
# Run Santiago Florido's ORIGINAL, unmodified pipeline.py on CifarTile (via his own
# run_pipeline.py entry point and experiments_config.yaml base config) -- no NAS-challenge
# restructuring, no submission_demeter code involved at all. Purpose: get a genuine ground-truth
# number to compare submission_demeter's own results against, and specifically to check whether
# the "growth.steps: 60" FIXED loop count (vs. our port's early-exit-on-freeze design) explains the
# accuracy/param gap already found.
#
# This is adapted from Florido's own run_init_scaling_ablation.slurm (same repo,
# experiments/pipeline/), simplified to a single run (no seed/init-strategy array) and with
# REQUIRE_DATA_READY dropped since margaret's compute nodes have internet access -- the dataset
# class (tools/datasets.py's NpyWebDataset) auto-downloads from the same figshare URLs already
# verified working elsewhere in this project.
#
# I (Claude) cannot submit or run this on margaret myself -- I have no access to that cluster.
#
# Usage:
#   1. Make sure this whole experimental_grow-main repo is on margaret (wherever PROJECT_DIR
#      points -- see below).
#   2. No separate gromo clone needed: this reuses submission_demeter's already-vendored copy
#      (nas-challenge-submission-main/submission_demeter/python_packages/gromo/). Checked directly
#      -- the upstream AblationDAG branch HEAD is the exact same commit already vendored there, so
#      a fresh clone would be byte-identical except for two narrowly-scoped local bugfixes (a
#      None-guard and a leaf-tensor .clone(), both documented inline in the vendored files) that
#      this pipeline's own CellArchitecture model would very likely hit too, since it's the same
#      4-DAG architecture. Point GROMO_PYTHONPATH at that vendored python_packages/ dir (default
#      below assumes the standard NAS/ layout; override if yours differs).
#   3. Reuse your existing submission_demeter venv rather than building a new one -- it already has
#      the slow part (torch/torchvision/numpy) installed. Just add this pipeline's other deps:
#        source <VENV_PATH>/bin/activate
#        pip install matplotlib pyyaml python-dotenv "hydra-core>=1.3.2" "mlflow>=3" pyvis psutil \
#                    "wandb>=0.18" requests
#   4. sbatch --export=ALL,VENV_PATH=/path/to/venv run_original_cifartile_margaret.slurm
#      (override GROMO_PYTHONPATH/DATA_DIR/SEED/SBATCH time the same way if needed)

#SBATCH --job-name=demeter_original_cifartile
#SBATCH --partition=tau
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

# ==================================================================================================
# Configuration -- override any of these with `sbatch --export=VAR=value ...`
# ==================================================================================================

# Where this experimental_grow-main checkout lives on margaret.
PROJECT_DIR="${PROJECT_DIR:-$HOME/nas/experimental_grow-main}"

# Reuses submission_demeter's already-vendored gromo (verified identical to upstream AblationDAG
# HEAD except for two narrowly-scoped local bugfixes this same CellArchitecture model would very
# likely need anyway -- see the usage comment above). No "src/" suffix: unlike a fresh clone, the
# vendored copy has gromo/ directly inside this directory.
GROMO_PYTHONPATH="${GROMO_PYTHONPATH:-$HOME/nas/nas-challenge-submission-main/submission_demeter/python_packages}"

# A venv with this pipeline's own dependencies installed (see step 3 in the usage comment above).
VENV_PATH="${VENV_PATH:?VENV_PATH is required}"

# Where the dataset gets auto-downloaded to/read from (figshare zip + extracted dir, NOT the
# challenge-format .npy layout submission_demeter uses -- this is intentional, it's Florido's own
# native dataset loader, unmodified).
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"

SEED="${SEED:-0}"

# ==================================================================================================

mkdir -p "$DATA_DIR"

cd "$PROJECT_DIR"

echo "Job id: ${SLURM_JOB_ID:-manual}"
echo "Host: $(hostname)"
echo "Project dir: $PROJECT_DIR"
echo "Gromo PYTHONPATH: $GROMO_PYTHONPATH"
echo "Data dir: $DATA_DIR"
echo "Seed: $SEED"
nvidia-smi

source "$VENV_PATH/bin/activate"

# Vendored gromo + this repo's own tools/experiments packages on PYTHONPATH -- same idea as
# run_init_scaling_ablation.slurm's own setup, just pointed at the vendored copy instead of a
# fresh clone (see the usage comment above for why that's fine here).
export PYTHONPATH="$GROMO_PYTHONPATH:$PROJECT_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
# Offline: runs the real logging code path unmodified without needing a wandb account/login.
export WANDB_MODE=offline

python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"
python -c "import gromo; print('gromo:', gromo.__file__)"

RUN_NAME="original_cifartile_seed${SEED}_${SLURM_JOB_ID:-manual}"
mkdir -p "$DATA_DIR/logs/$RUN_NAME" "$DATA_DIR/tmp/$RUN_NAME"

# This is the exact invocation pattern from Florido's own run_init_scaling_ablation.slurm's
# COMMON_ARGS + growth.initialization_strategy override -- nothing added or removed beyond
# pointing --logger.enabled false (skip wandb entirely, redundant with WANDB_MODE=offline above
# but explicit) and using this run's own name/paths instead of the ablation-array ones.
python experiments/pipeline/run_and_report.py \
    --config experiments/pipeline/experiments_config.yaml \
    --experiment.seed "$SEED" \
    --experiment.name "$RUN_NAME" \
    --dataset.name cifartile \
    --dataset.path "$DATA_DIR" \
    --growth.strategy restricted \
    --growth.initialization_strategy local \
    --training.es_abs_delta 0.00122 \
    --logger.enabled false \
    --logger.path "$DATA_DIR/logs/$RUN_NAME" \
    --logger.tmpdir "$DATA_DIR/tmp/$RUN_NAME"

echo "Done."
