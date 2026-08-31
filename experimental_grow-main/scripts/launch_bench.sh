#!/bin/bash
# Submit one array job per (dataset, model) combination.
# Usage: bash scripts/launch_bench.sh
# Tweak DATASETS, MODELS, and SLEEP_BETWEEN below.

#SBATCH --output=slurm/launch_bench.out

SLEEP_BETWEEN=10   # seconds between sbatch calls to avoid GPU context clashes

# dataset_name:num_classes
DATASETS=(
    "mnist:10"
    "addnist:20"
    "cifar10:10"
    "cifar100:100"
    "cifartile:4"
    "language:10"
    "gutenberg:6"
    "geoclassing:10"
    "chesseract:3"
    "gameoflife:25"
)

MODELS=(
    mobilenet_v2
    mobilenet_v3_small
    mobilenet_v3_large
    efficientnet_b0
    efficientnet_b1
    resnet18
    resnet34
    resnet50
    mnasnet1_0
    convnext_tiny
)

for ds_entry in "${DATASETS[@]}"; do
    DATASET="${ds_entry%%:*}"
    CLASSES="${ds_entry##*:}"

    for MODEL in "${MODELS[@]}"; do
        echo "Submitting: dataset=${DATASET} classes=${CLASSES} model=${MODEL}"

        sbatch \
            --job-name="${MODEL}_${DATASET}" \
            --export=ALL,BENCH_DATASET="${DATASET}",BENCH_CLASSES="${CLASSES}",BENCH_MODEL="${MODEL}" \
            scripts/bench_run.sh

        sleep "${SLEEP_BETWEEN}"
    done
done

echo "All jobs submitted."
