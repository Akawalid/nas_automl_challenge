#!/bin/bash

# Run an interactive session launching jupyter notebook
OPTIONS=""
OPTIONS="$OPTIONS --job-name=jupyter"
OPTIONS="$OPTIONS --output=slurm/slurm-%x-%j.out"
OPTIONS="$OPTIONS -p gpu-best"
OPTIONS="$OPTIONS --ntasks=1"
OPTIONS="$OPTIONS --gres=gpu:1"
OPTIONS="$OPTIONS -t 8:00:00"

echo "srun $OPTIONS scripts/node_jupyter.sh" 
srun $OPTIONS scripts/node_jupyter.sh