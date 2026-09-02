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
# below. His own ablation scripts for this exact full config (run_init_scaling_ablation.slurm,
# run_cifar_init_noisy_ablation.slurm, run_variance_transfer_ablation.slurm, etc., all single
# dataset/single seed like this run) all request --time=48:00:00 -- matched here.
#
# PARTITION: back to --partition=tau. Checked against the actual margaret hardware docs
# (clusters-saclay.gitlabpages.inria.fr) -- margpu008 (3x A100 40GB, the node Florido's own real
# benchmark launcher launch_local_base_multi_dataset.sh pins to for the runs that produced the
# paper's numbers) is Tau-team-owned hardware, so --partition=tau is plausibly the correct,
# priority path to it, not a weaker fallback as an earlier version of this comment assumed. The
# GeoClassing OOM (10.90 GiB total GPU capacity at crash time) matches margpu017-023's 2080Ti
# cards (11GB each) instead -- i.e. this run simply never requested margpu008 specifically and
# landed on a small card by chance, on either partition name. For GeoClassing specifically, use
# run_florido_geoclassing_margaret.slurm (pins --nodelist=margpu008) rather than this script.
#
# NO PERSISTED OUTPUT: this deliberately does NOT save predictions/scoring/results anywhere --
# score.py's own Raw/Adj Score/params/runtime printout (visible directly in this job's own
# logs/slurm-%x-%j.out) is the only thing kept. Each run builds and scores inside an ephemeral
# per-job directory (see JOB_WORKDIR below), auto-deleted on exit.
#
# PARALLEL-SAFE: earlier versions of this script cd'd into the shared
# nas-challenge-submission-main/ and let `make` build package/ and scoring/ THERE -- fine for one
# job at a time, but multiple concurrent jobs would race on the same rm -Rf package/mkdir package
# (Makefile's build/score targets are destructive). Fixed by giving each job its own JOB_WORKDIR
# (symlinks to the shared datasets/evaluation/$SUBMISSION -- cheap, no multi-GB copy -- plus its
# own Makefile copy) so `make`'s package/scoring live in a job-private directory. The dataset
# download loop is also narrowed to just $DATASETS_TO_RUN (was: all 5 unconditionally) so two
# jobs targeting different datasets never touch the same not-yet-downloaded files. pip install
# still targets the one shared venv, so that one step is flock-guarded to serialize it across
# concurrent jobs (harmless once deps are already installed -- the lock is held only briefly).
#
# I (Claude) cannot submit or run this on margaret myself -- I have no access to that cluster.
#
# Usage:
#   1. git clone (or pull) this repo onto margaret -- see PROJECT_DIR below. This needs
#      submission_demeter_florido/ to actually be present in the clone (it's tracked in git,
#      not gitignored -- verify with `git ls-files submission_demeter_florido | head`).
#   2. sbatch --export=VENV_DIR=/path/to/your/venv,DATASETS_TO_RUN=CIFARTile run_florido_margaret.slurm
#      -- safe to submit several of these concurrently for different DATASETS_TO_RUN values, e.g.
#         for d in MultNIST Gutenberg GeoClassing Chesseract; do
#             sbatch --job-name=florido_$d --export=VENV_DIR=...,DATASETS_TO_RUN=$d run_florido_margaret.slurm
#         done

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

# Which dataset folders (under datasets/) main.py should actually process this run -- space
# separated, matching the folder names in DATASET_IDS below. Defaults to CIFARTile only.
DATASETS_TO_RUN="${DATASETS_TO_RUN:-CIFARTile}"

export PYTHONUNBUFFERED=1

# ==================================================================================================

cd "$PROJECT_DIR/nas-challenge-submission-main"

echo "Job id: ${SLURM_JOB_ID:-manual}"
echo "Host: $(hostname)"
echo "Submission: $SUBMISSION"
echo "Datasets to run: $DATASETS_TO_RUN"
nvidia-smi

# ---- environment ----
source "$VENV_DIR/bin/activate"

python -c "import torch, torchvision; print('torch', torch.__version__, 'torchvision', torchvision.__version__, 'cuda available:', torch.cuda.is_available())"

# gromo itself is vendored in $SUBMISSION/python_packages/, no separate install needed for that --
# only the plain pip deps. flock-guarded: safe if another concurrent job is installing into the
# same shared venv at the same time (see header comment).
(
    flock -x 200
    pip install --no-cache-dir -r "$SUBMISSION/requirements.txt"
) 200>"$VENV_DIR/.pip_install.lock"

# ---- datasets (identical provenance to run_demeter_margaret.slurm) -- only downloads what THIS
# job actually needs, so concurrent jobs targeting different datasets never touch the same files.
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

echo "Checking datasets/ for: $DATASETS_TO_RUN"
mkdir -p datasets
for name in $DATASETS_TO_RUN; do
    download_dataset "$name" "${DATASET_IDS[$name]}"
done

# ---- ephemeral per-job build dir -- see header comment. Symlinks (not copies) to the shared,
# read-only-during-build datasets/evaluation/$SUBMISSION; only package/ and scoring/ (what make
# actually writes into) end up job-private. Auto-removed on exit, including on failure.
JOB_WORKDIR="/tmp/${USER}_florido_${SLURM_JOB_ID:-manual}"
mkdir -p "$JOB_WORKDIR"
trap 'rm -rf "$JOB_WORKDIR"' EXIT

REPO_DIR="$(pwd)"
ln -sfn "$REPO_DIR/datasets" "$JOB_WORKDIR/datasets"
ln -sfn "$REPO_DIR/evaluation" "$JOB_WORKDIR/evaluation"
ln -sfn "$REPO_DIR/$SUBMISSION" "$JOB_WORKDIR/$SUBMISSION"
cp "$REPO_DIR/Makefile" "$JOB_WORKDIR/Makefile"
cd "$JOB_WORKDIR"

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

echo "Done (nothing persisted -- scores are above / in this job's own log)."
