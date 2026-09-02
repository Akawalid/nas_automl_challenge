#!/bin/bash
#
# Dedicated GeoClassing run of submission_demeter_florido on margaret. Same as
# run_florido_margaret.slurm in every other respect -- see that script's own header comment for
# what's not repeated here -- except this one pins the node.
#
# WHY A SEPARATE SCRIPT: a real run of GeoClassing (both through this harness AND through the
# standalone run_original_cifartile_margaret.slurm -- Florido's own unmodified pipeline, no glue
# code involved) crashed with a CUDA OOM inside expand_node()'s neuron-selection step
# (gromo/containers/growing_graph_network.py, the conv_transpose2d call around line 716), on a GPU
# reporting only 10.90 GiB total capacity. Checked against the actual margaret hardware docs
# (clusters-saclay.gitlabpages.inria.fr/clusters-docs/docs/margaret/hardware/): 10.90 GiB matches
# margpu017-023's 2080Ti cards (11GB each), NOT the 40GB A100s the run actually needed -- neither
# run had requested a specific node, so the scheduler put it on whatever GPU was free. Florido's
# own real multi-dataset benchmark launcher (launch_local_base_multi_dataset.sh, which produced
# the paper's own numbers) pins its whole sweep to one specific node for exactly this reason:
#     BENCHMARK_NODE="${BENCHMARK_NODE:-margpu008}"
#     SBATCH_ARGS+=(--nodelist="$BENCHMARK_NODE")
# margpu008 = 3x NVIDIA A100 40GB (also confirmed Tau-team-owned hardware, matching --partition=tau
# below). Replicated here. This intentionally is NOT folded into run_florido_margaret.slurm's own
# DATASETS_TO_RUN loop -- pinning every dataset to one node would serialize the other 3 concurrent
# jobs on it for no reason (they aren't hitting this memory ceiling).
#
# I (Claude) cannot submit or run this on margaret myself -- I have no access to that cluster.
#
# Usage:
#   sbatch --export=VENV_DIR=/path/to/your/venv run_florido_geoclassing_margaret.slurm

#SBATCH --job-name=florido_geoclassing
#SBATCH --partition=tau
#SBATCH --nodelist=margpu008
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

# ==================================================================================================
# Configuration -- override any of these with `sbatch --export=VAR=value run_florido_geoclassing_margaret.slurm`
# ==================================================================================================

PROJECT_DIR="${PROJECT_DIR:-$HOME/nas}"
SUBMISSION="${SUBMISSION:-submission_demeter_florido}"
VENV_DIR="${VENV_DIR:?VENV_DIR is required -- e.g. sbatch --export=VENV_DIR=$HOME/.venvs/demeter run_florido_geoclassing_margaret.slurm}"
DATASETS_TO_RUN="${DATASETS_TO_RUN:-GeoClassing}"

# evaluation/main.py's own internal time budget (separate from --time above -- see
# run_florido_margaret.slurm's header for why this matters: both worker threads in main() are
# daemon threads, so whichever fires first between real completion and this countdown kills
# whatever's still running, mid-execution, no partial save). GeoClassing's own crash was memory,
# not time (it OOM'd ~6h20m in, well under the 12h default), but set generously anyway as a
# consistent safety margin now that the actual blocker (memory) is fixed.
export NAS_TIME_LIMIT_HOURS="${NAS_TIME_LIMIT_HOURS:-46}"

export PYTHONUNBUFFERED=1

# ==================================================================================================

cd "$PROJECT_DIR/nas-challenge-submission-main"

echo "Job id: ${SLURM_JOB_ID:-manual}"
echo "Host: $(hostname)"
echo "Submission: $SUBMISSION"
echo "Datasets to run: $DATASETS_TO_RUN"
echo "NAS_TIME_LIMIT_HOURS: $NAS_TIME_LIMIT_HOURS"
nvidia-smi

# ---- environment ----
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

# ---- ephemeral per-job build dir (see run_florido_margaret.slurm's header for why) ----
JOB_WORKDIR="/tmp/${USER}_florido_${SLURM_JOB_ID:-manual}"
mkdir -p "$JOB_WORKDIR"
trap 'rm -rf "$JOB_WORKDIR"' EXIT

REPO_DIR="$(pwd)"
ln -sfn "$REPO_DIR/datasets" "$JOB_WORKDIR/datasets"
ln -sfn "$REPO_DIR/evaluation" "$JOB_WORKDIR/evaluation"
ln -sfn "$REPO_DIR/$SUBMISSION" "$JOB_WORKDIR/$SUBMISSION"
cp "$REPO_DIR/Makefile" "$JOB_WORKDIR/Makefile"
cd "$JOB_WORKDIR"

# ---- run ----
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

echo "Done (nothing persisted -- scores are above / in this job's own log)."
