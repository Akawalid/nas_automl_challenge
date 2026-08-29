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
# This script follows the same --partition=tau / /home/tau/$USER/tau_frugal/$USER/logs conventions
# as the existing *.slurm files in experimental_grow-main/{hydra_script,experiments/pipeline}/, but
# activates a plain Python venv (not conda) -- point VENV_DIR at one with
# torch/torchvision/scikit-learn/numpy/Deprecated installed.
#
# Usage:
#   1. git clone (or pull) this repo onto margaret -- see PROJECT_DIR below.
#   2. sbatch --export=VENV_DIR=/path/to/your/venv run_demeter_margaret.slurm
#      -- datasets/ is populated automatically on first run (margaret's compute nodes have
#      internet access): the job fetches all 5 benchmark datasets straight from figshare if
#      they're not already present. See the DATASETS block below for details / to skip this.
#      (override any of the variables below the same way, e.g. `--export=VENV_DIR=...,SUBMISSION=...`)

#SBATCH --job-name=demeter_benchmarks
#SBATCH --partition=tau
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH --output=/home/tau/%u/tau_frugal/%u/logs/slurm-%x-%j.out
#SBATCH --error=/home/tau/%u/tau_frugal/%u/logs/slurm-%x-%j.err



make all