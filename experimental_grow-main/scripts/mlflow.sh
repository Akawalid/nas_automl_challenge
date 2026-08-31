#!/bin/bash

# Run an interactive session launching mlflow server
OPTIONS=""
OPTIONS="$OPTIONS --job-name=mlflow"
OPTIONS="$OPTIONS --output=slurm/slurm-%x-%j.out"
OPTIONS="$OPTIONS -p tau"
OPTIONS="$OPTIONS --ntasks=1"

echo "srun $OPTIONS scripts/node_mlflow.sh"
srun $OPTIONS scripts/node_mlflow.sh
