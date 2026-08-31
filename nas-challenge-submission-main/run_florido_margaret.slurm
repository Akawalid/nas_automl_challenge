#!/bin/bash
#
# Run submission_demeter_florido on margaret (SLURM). Unlike run_demeter_margaret.slurm, this
# submission has no fast/approximate profile -- it vendors Santiago Florido's own real pipeline
# code verbatim and drives it through his own execute_pipeline() with the REAL, untouched
# experiments_config.yaml (training.epochs=100, growth.steps=60, growth.neuron_epochs=100). There
# is nothing to toggle here: this always runs at full fidelity. That's the entire point of this
# submission -- see submission_demeter_florido/nas.py's own docstring.
#
# IMPORTANT: submission_demeter_florido/nas.py never looks at the time budget at all (no
# self.clock usage) -- it just calls Florido's real execute_pipeline() and lets it run to
# completion. evaluation/main.py's TIME_LIMIT countdown thread can't actually stop it mid-training
# (Python threads aren't preemptible), it only unblocks the harness's main thread while the real
# run keeps going in the background -- so the ONLY hard stop is SLURM's own --time allocation
# below. No completed run's actual elapsed time is recorded anywhere in experimental_grow-main
# (checked -- no .out/.err/sacct output committed, only his own README telling YOU to check
# `sacct -j JOB_ID --format=Elapsed` yourself after running). What IS real evidence: his own
# ablation scripts for this exact full config (run_init_scaling_ablation.slurm,
# run_cifar_init_noisy_ablation.slurm, run_variance_transfer_ablation.slurm, etc., all single
# dataset/single seed like this run) all request --time=48:00:00, not 24h -- matched here rather
# than guessed.
#
# Defaults to CIFARTile only (one dataset) -- matching the earlier full-fidelity sanity check done
# against submission_demeter, so this run is directly comparable to that one and to the paper's
# Table 3 (55.7% +/- 2.7% for CifarTile). Override DATASETS_TO_RUN to add more once you've seen how
# long one takes.
#
# I (Claude) cannot submit or run this on margaret myself -- I have no access to that cluster.
#
# Usage:
#   1. git clone (or pull) this repo onto margaret -- see PROJECT_DIR below. This needs
#      submission_demeter_florido/ to actually be present in the clone (it's tracked in git,
#      not gitignored -- verify with `git ls-files submission_demeter_florido | head`).
#   2. sbatch --export=VENV_DIR=/path/to/your/venv run_florido_margaret.slurm
#      -- datasets/ is populated automatically on first run, same as run_demeter_margaret.slurm.
#      -- override any variable below the same way, e.g.
#         sbatch --export=VENV_DIR=...,DATASETS_TO_RUN="CIFARTile Gutenberg" run_florido_margaret.slurm

#SBATCH --job-name=florido_benchmarks
#SBATCH --partition=tau
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

# ==================================================================================================
# Configuration -- override any of these with `sbatch --export=VAR=value run_florido_margaret.slurm`
# ==================================================================================================

# Where this repo (specifically nas-challenge-submission-main/) lives on margaret.
PROJECT_DIR="${PROJECT_DIR:-$HOME/nas}"

SUBMISSION="${SUBMISSION:-submission_demeter_florido}"

# Path to a plain Python venv (venv/virtualenv/uv venv) -- e.g. the same
# /data/tau/iceberg_1/shared/ochabane/venvs/demeter used for submission_demeter. Its own
# torch/torchvision/numpy are reused; this submission's extra deps (GitPython, networkx, timm,
# tqdm, etc. -- see requirements.txt) get installed into it below if missing.
VENV_DIR="${VENV_DIR:?VENV_DIR is required -- e.g. sbatch --export=VENV_DIR=$HOME/.venvs/demeter run_florido_margaret.slurm}"

RESULTS_NAME="${RESULTS_NAME:-${SLURM_JOB_NAME:-florido}_${SLURM_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S)}"
RESULTS_DIR="results/${RESULTS_NAME}"

# Which dataset folders (under datasets/) main.py should actually process this run -- space
# separated, matching the folder names in DATASET_IDS below. Defaults to CIFARTile only: the point
# of this run is a real, single-dataset comparison against the paper's Table 3/Table 2 numbers, not
# a full 5-dataset sweep (which would multiply the already-unknown per-dataset wall-clock cost by 5).
DATASETS_TO_RUN="${DATASETS_TO_RUN:-CIFARTile}"

export PYTHONUNBUFFERED=1

# ==================================================================================================

cd "$PROJECT_DIR/nas-challenge-submission-main"

echo "Job id: ${SLURM_JOB_ID:-manual}"
echo "Host: $(hostname)"
echo "Submission: $SUBMISSION"
echo "Datasets to run: $DATASETS_TO_RUN"
echo "Results will be left at: nas-challenge-submission-main/$RESULTS_DIR"
nvidia-smi

# ---- environment ----
source "$VENV_DIR/bin/activate"

python -c "import torch, torchvision; print('torch', torch.__version__, 'torchvision', torchvision.__version__, 'cuda available:', torch.cuda.is_available())"

# gromo itself is vendored in $SUBMISSION/python_packages/, no separate install needed for that --
# only the plain pip deps (requirements.txt pins a cu113 PyTorch build; check that matches
# margaret's actual CUDA/driver before relying on this -- comment out the -f line / adjust the
# torch/torchvision pins if not, same caveat as run_demeter_margaret.slurm).
pip install --no-cache-dir -r "$SUBMISSION/requirements.txt"

# ---- datasets (identical to run_demeter_margaret.slurm -- see there for provenance notes) ----
declare -A DATASET_IDS=(
    [MultNIST]="24574678:1"
    [CIFARTile]="24551539:1"
    [Gutenberg]="24574753:1"
    [GeoClassing]="24050256:3"
    [Chesseract]="24118743:2"
)
WANTED_FILES="metadata train_x.npy train_y.npy valid_x.npy valid_y.npy test_x.npy test_y.npy"

download_dataset () {
    local name="$1" id="${2%%:*}" ver="${2#*:}" dir="datasets/$1"
    if [ -f "$dir/metadata" ] && [ -f "$dir/train_x.npy" ]; then
        echo "  $name: already present, skipping"
        return
    fi
    echo "  $name: downloading (figshare article $id v$ver)..."
    mkdir -p "$dir"
    curl -sf --max-time 60 "https://api.figshare.com/v2/articles/${id}/versions/${ver}" \
        -o "$dir/.filelist.json"
    python3 - "$dir" "$WANTED_FILES" <<'PYEOF'
import json, sys, os
d, wanted = sys.argv[1], set(sys.argv[2].split())
with open(os.path.join(d, ".filelist.json")) as f:
    files = json.load(f)["files"]
manifest = [(e["name"], e["download_url"]) for e in files if e["name"] in wanted]
missing = wanted - {n for n, _ in manifest}
if missing:
    sys.exit(f"figshare listing for {d} is missing expected file(s): {missing}")
with open(os.path.join(d, ".manifest.tsv"), "w") as f:
    for n, u in manifest:
        f.write(f"{n}\t{u}\n")
PYEOF
    while IFS=$'\t' read -r fname url; do
        curl -sL --max-time 300 --retry 3 "$url" -o "$dir/$fname"
        [ -s "$dir/$fname" ] || { echo "download failed or empty: $dir/$fname" >&2; exit 1; }
    done < "$dir/.manifest.tsv"
    rm -f "$dir/.filelist.json" "$dir/.manifest.tsv"
    echo "  $name: done ($(du -sh "$dir" | cut -f1))"
}

echo "Checking datasets/ (downloading anything missing)..."
mkdir -p datasets
for name in "${!DATASET_IDS[@]}"; do
    download_dataset "$name" "${DATASET_IDS[$name]}"
done
echo "Datasets ready: $(ls datasets)"

# ---- run (Makefile: clean -> build, then prune to DATASETS_TO_RUN, then run -> score) ----
make submission="$SUBMISSION" clean build

echo "Pruning package/datasets/ to just: $DATASETS_TO_RUN"
for d in package/datasets/*/; do
    name="$(basename "$d")"
    if [[ ! " $DATASETS_TO_RUN " == *" $name "* ]]; then
        rm -rf "$d"
    fi
done
echo "Will actually run: $(ls package/datasets)"

make submission="$SUBMISSION" run

# ---- score (inlined so labels/ can be pruned the same way as package/datasets/ above) ----
rm -Rf scoring
mkdir scoring
mkdir scoring/labels
mkdir scoring/predictions
rsync -avr --exclude='**/*x.npy' --exclude='**/train*.npy' --exclude='**/valid*.npy' --include='**/test_y.npy' datasets/* scoring/labels/
for d in scoring/labels/*/; do
    name="$(basename "$d")"
    if [[ ! " $DATASETS_TO_RUN " == *" $name "* ]]; then
        rm -rf "$d"
    fi
done
cp -R package/predictions scoring
cp evaluation/score.py scoring/score.py
cd scoring; python3 score.py; cd ..

# ---- collect results ----
mkdir -p "$RESULTS_DIR"
rsync -av package/predictions/ "$RESULTS_DIR/predictions/"
[ -d scoring ] && rsync -av scoring/ "$RESULTS_DIR/scoring/"
cp "slurm-${SLURM_JOB_NAME:-florido_benchmarks}-${SLURM_JOB_ID:-manual}.out" \
   "$RESULTS_DIR/run.log" 2>/dev/null || true

echo "Done. Results left at nas-challenge-submission-main/$RESULTS_DIR"
