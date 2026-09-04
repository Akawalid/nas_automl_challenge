#!/bin/bash
#
# Run submission_demeter_florido on ONE dataset, N independent times, to directly measure
# run-to-run variance (the "threshold-cascade" hypothesis from the GeoClassing investigation).
#
# IMPORTANT -- read before using this for a "different seeds" experiment: "seed" is currently
# functionally inert. Checked directly: set_random_seeds() (the actual RNG-seeding function) is
# only ever called from pipeline.py's set_random_seed(cfg, **_) step, and that step is NOT in
# experiments_config.yaml's own `pipeline:` list on either config variant (grepped both, zero
# matches) -- so even Florido's own real ablation scripts, which pass --experiment.seed on the
# CLI, never actually apply it to any RNG. This script therefore does NOT try to control the seed
# (nothing to control) -- each array task is just an independent, naturally-unseeded repetition,
# which is what actually answers "how much does this vary run to run". The SEED_LABEL below is
# recorded into metadata/logging for bookkeeping only -- it has zero effect on behavior today.
#
# Each array task gets its own JOB_WORKDIR (same isolation as run_florido_margaret.slurm) so N
# repeats run fully in parallel with no shared-directory races.
#
# I (Claude) cannot submit or run this on margaret myself -- I have no access to that cluster.
#
# Usage:
#   sbatch --array=0-4 --export=VENV_DIR=/path/to/your/venv,DATASETS_TO_RUN=GeoClassing run_florido_repeats_margaret.slurm
#   -- --array=0-4 means 5 independent repeats (task ids 0..4). Pin a node if you also want to
#      hold hardware constant across repeats: add --nodelist=margpu008 to the sbatch command.

#SBATCH --job-name=florido_repeats
#SBATCH --partition=tau
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm-%x-%A_%a.out
#SBATCH --error=logs/slurm-%x-%A_%a.err

set -euo pipefail

# ==================================================================================================
PROJECT_DIR="${PROJECT_DIR:-$HOME/nas}"
SUBMISSION="${SUBMISSION:-submission_demeter_florido}"
VENV_DIR="${VENV_DIR:?VENV_DIR is required -- e.g. sbatch --array=0-4 --export=VENV_DIR=$HOME/.venvs/demeter,DATASETS_TO_RUN=GeoClassing run_florido_repeats_margaret.slurm}"
DATASETS_TO_RUN="${DATASETS_TO_RUN:-GeoClassing}"
export NAS_TIME_LIMIT_HOURS="${NAS_TIME_LIMIT_HOURS:-46}"
export PYTHONUNBUFFERED=1

# Array-task bookkeeping. SEED_LABEL is currently cosmetic (see header) -- kept anyway so results
# are traceable back to a specific repeat, and so this script needs no changes the day
# set_random_seed actually gets wired back into the pipeline.
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
SEED_LABEL="$TASK_ID"
# ==================================================================================================

cd "$PROJECT_DIR/nas-challenge-submission-main"

echo "Job id: ${SLURM_ARRAY_JOB_ID:-manual}  Array task: $TASK_ID  (repeat #$TASK_ID)"
echo "Host: $(hostname)"
echo "Submission: $SUBMISSION"
echo "Dataset: $DATASETS_TO_RUN"
echo "SEED_LABEL (cosmetic only -- see header comment): $SEED_LABEL"
nvidia-smi

source "$VENV_DIR/bin/activate"
python -c "import torch, torchvision; print('torch', torch.__version__, 'torchvision', torchvision.__version__, 'cuda available:', torch.cuda.is_available())"

(
    flock -x 200
    pip install --no-cache-dir -r "$SUBMISSION/requirements.txt"
) 200>"$VENV_DIR/.pip_install.lock"

# ---- datasets ----
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
    curl -sf --max-time 60 "https://api.figshare.com/v2/articles/${id}/versions/${ver}" -o "$dir/.filelist.json"
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

# ---- ephemeral per-ARRAY-TASK build dir -- keyed by job+task id so concurrent repeats never collide ----
JOB_WORKDIR="/tmp/${USER}_florido_${SLURM_ARRAY_JOB_ID:-manual}_${TASK_ID}"
mkdir -p "$JOB_WORKDIR"
trap 'rm -rf "$JOB_WORKDIR"' EXIT

REPO_DIR="$(pwd)"
ln -sfn "$REPO_DIR/datasets" "$JOB_WORKDIR/datasets"
ln -sfn "$REPO_DIR/evaluation" "$JOB_WORKDIR/evaluation"
ln -sfn "$REPO_DIR/$SUBMISSION" "$JOB_WORKDIR/$SUBMISSION"
cp "$REPO_DIR/Makefile" "$JOB_WORKDIR/Makefile"
cd "$JOB_WORKDIR"

make submission="$SUBMISSION" clean build

echo "Pruning package/datasets/ to just: $DATASETS_TO_RUN"
for d in package/datasets/*/; do
    name="$(basename "$d")"
    if [[ ! " $DATASETS_TO_RUN " == *" $name "* ]]; then
        rm -rf "$d"
    fi
done

make submission="$SUBMISSION" run

# ---- score ----
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

echo "=== Repeat #$TASK_ID done (dataset: $DATASETS_TO_RUN) ==="
