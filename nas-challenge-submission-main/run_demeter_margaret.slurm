#!/bin/bash
#
# Run submission_demeter on the challenge benchmark datasets on margaret (SLURM), leaving the
# results under nas-challenge-submission-main/results/<name>/ on margaret's own filesystem. This
# does NOT run anything locally -- it's meant to be submitted with `sbatch` from margaret itself
# (or from a login node with a git clone of this repo).
#
# This is the FAST/APPROXIMATE run (DEMETER_FAST_RUN=1, ~4h total across all 5 datasets instead of
# the paper's own ~50.6 GPU-hours -- see nas.py's set_metadata() for exactly what's cut). Purpose:
# sanity-check that the growth trajectory / validation-accuracy curve looks reasonable, not to
# reproduce paper-quality final numbers. Set DEMETER_FAST_RUN=0 for the real, full-fidelity run.
#
# I (Claude) cannot submit or run this on margaret myself -- I have no access to that cluster.
# This script follows the same --partition=tau convention as the existing *.slurm files in
# experimental_grow-main/{hydra_script,experiments/pipeline}/, but writes its own SLURM
# stdout/stderr next to the script itself (slurm-<jobname>-<jobid>.{out,err}, relative to wherever
# you run `sbatch` from) rather than to their /tau_frugal/ logs convention, and activates a plain
# Python venv (not conda) -- point VENV_DIR at one with torch/torchvision/scikit-learn/numpy/Deprecated
# installed.
#
# Usage:
#   1. git clone (or pull) this repo onto margaret -- see PROJECT_DIR below.
#   2. sbatch --export=VENV_DIR=/path/to/your/venv run_demeter_margaret.slurm
#      -- datasets/ is populated automatically on first run (margaret's compute nodes have
#      internet access): the job fetches all 5 benchmark datasets straight from figshare if
#      they're not already present. See the DATASETS block below for details / to skip this.
#      (override any of the variables below the same way, e.g. `--export=VENV_DIR=...,SUBMISSION=...`)
#      -- to skip datasets a previous run already completed, add e.g.
#      `DATASETS_TO_RUN="GeoClassing Gutenberg MultNIST"` (folder names, not codenames)

#SBATCH --job-name=demeter_benchmarks
#SBATCH --partition=tau
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

# ==================================================================================================
# Configuration -- override any of these with `sbatch --export=VAR=value run_demeter_margaret.slurm`
# ==================================================================================================

# Where this repo (specifically nas-challenge-submission-main/) lives on margaret.
PROJECT_DIR="${PROJECT_DIR:-$HOME/nas}"

# Which submission folder to run (must exist under $PROJECT_DIR/nas-challenge-submission-main/).
SUBMISSION="${SUBMISSION:-submission_demeter}"

# Path to a plain Python venv (venv/virtualenv/uv venv -- anything with bin/activate) with
# torch/torchvision/scikit-learn/numpy/Deprecated installed, e.g. ~/.venvs/demeter. See the
# "install requirements" step below, which installs them into this env if missing.
VENV_DIR="${VENV_DIR:?VENV_DIR is required -- e.g. sbatch --export=VENV_DIR=$HOME/.venvs/demeter run_demeter_margaret.slurm}"

# Results (predictions/ + scoring output + the run log) are left under
# nas-challenge-submission-main/results/<RESULTS_NAME>/ on margaret's own filesystem.
RESULTS_NAME="${RESULTS_NAME:-${SLURM_JOB_NAME:-demeter}_${SLURM_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S)}"
RESULTS_DIR="results/${RESULTS_NAME}"

# Which dataset folders (under datasets/) main.py should actually process this run -- space
# separated, matching the folder names in DATASET_IDS below (not the internal 'codename' field
# printed in logs, e.g. use "GeoClassing" not "Sadie"). Defaults to all 5. Narrow this to just the
# ones you still need after a previous run already completed some of them -- main.py has no notion
# of "already done", it just reprocesses whatever it finds in datasets/ from scratch every time.
DATASETS_TO_RUN="${DATASETS_TO_RUN:-MultNIST CIFARTile Gutenberg GeoClassing Chesseract}"

# Turns on nas.py's fast/approximate hyperparameter profile (~10x fewer epochs on the 3 knobs that
# dominate wall-clock, plus a tighter round cap) -- see nas.py's set_metadata() for exactly what
# this changes and why. Set to 0 (or unset and override via sbatch --export) for the real run.
export DEMETER_FAST_RUN="${DEMETER_FAST_RUN:-1}"

# evaluation/main.py hardcodes a 12h wall-clock budget (TIME_LIMIT) shared across ALL datasets in
# datasets/ in a single run -- meant to mirror the real competition's per-submission budget. With
# DEMETER_FAST_RUN's ~12x speedup, the ~4h target for all 5 datasets fits well under that anyway;
# NAS_TIME_LIMIT_HOURS is kept as a safety net a bit above the target (not the primary mechanism --
# nas.py doesn't watch the clock mid-run, so this only protects against the approximation running
# long), comfortably under the --time=06:00:00 SBATCH allocation above to leave room for
# setup/downloads/scoring/copy-out.
export NAS_TIME_LIMIT_HOURS="${NAS_TIME_LIMIT_HOURS:-5}"

# Without this, Python fully-buffers stdout when it's not an interactive terminal (i.e. always,
# under SLURM) -- print() output (including the per-round progress lines and any crash traceback)
# only reaches the log file once the buffer fills or the process exits cleanly. If it crashes or
# hangs instead, that output can be lost for good. This makes every print() flush immediately.
export PYTHONUNBUFFERED=1

# ==================================================================================================

cd "$PROJECT_DIR/nas-challenge-submission-main"

echo "Job id: ${SLURM_JOB_ID:-manual}"
echo "Host: $(hostname)"
echo "Submission: $SUBMISSION"
echo "Results will be left at: nas-challenge-submission-main/$RESULTS_DIR"
nvidia-smi

# ---- environment ----
source "$VENV_DIR/bin/activate"

python -c "import torch, torchvision; print('torch', torch.__version__, 'torchvision', torchvision.__version__, 'cuda available:', torch.cuda.is_available())"

# Install the submission's own requirements (gromo itself is vendored in
# $SUBMISSION/python_packages/, no separate install needed for that -- only the plain pip deps).
# requirements.txt pins a cu113 PyTorch build; check that matches margaret's actual CUDA/driver
# before relying on this -- comment out the -f line / adjust the torch/torchvision pins if not.
pip install -r "$SUBMISSION/requirements.txt"

# ---- datasets ----
# The competition's actual grading datasets are hidden ("Unseen Data" challenge); these are the
# same 5 datasets the Demeter paper itself benchmarks against, and a good local stand-in. Each is
# hosted on Newcastle's figshare-backed repository (data.ncl.ac.uk) and already split into exactly
# the files this submission's DataProcessor expects: train_x.npy, train_y.npy, valid_x.npy,
# valid_y.npy, test_x.npy, test_y.npy, and a 'metadata' JSON file -- no unpacking needed, just
# fetch each file into datasets/<name>/.
#
# NOTE: data.ncl.ac.uk itself sits behind an AWS WAF bot challenge that blocks plain curl/wget
# (verified: even a browser User-Agent gets a 403/challenge response on its own download links) --
# so this goes through the public figshare API instead (api.figshare.com/v2/articles/<id>), which
# is not behind that wall and returns direct ndownloader.figshare.com links that a plain `curl -sL`
# follows fine, no auth needed. Verified working end-to-end against all 5 datasets below.
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

# ---- score (Makefile's `score` target, inlined so labels/ can be pruned the same way as
# package/datasets/ above -- otherwise score.py hits a dataset with labels but no matching
# predictions [pruned out] and crashes ("list index out of range") before scoring any of the
# datasets that actually ran, since it never skips missing ones gracefully) ----
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
cp "slurm-${SLURM_JOB_NAME:-demeter_benchmarks}-${SLURM_JOB_ID:-manual}.out" \
   "$RESULTS_DIR/run.log" 2>/dev/null || true

echo "Done. Results left at nas-challenge-submission-main/$RESULTS_DIR"
