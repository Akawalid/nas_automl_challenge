#!/bin/bash

#SBATCH --job-name=optimize_smac
#SBATCH --output=slurm/slurm-%x-%A_%a.out
#SBATCH --time=5-00:10:00
#SBATCH --ntasks=1
#SBATCH -p tau

echo "NODE: $(hostname)"

python experiments/pipeline/optimize.py

rm -r temp/slurm*.sh
#rm -r temp/results/*.json
