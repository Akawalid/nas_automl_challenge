#!/bin/bash
#SBATCH --job-name=test_job
#SBATCH --output=res_test_job_%A_%a.txt
#SBATCH --error=error_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --partition=tau
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --array=0-2

A_VALUES=(1 2 3)
B_VALUES=(4 5 6)

srun python job.py --A ${A_VALUES[$SLURM_ARRAY_TASK_ID]} --B ${B_VALUES[$SLURM_ARRAY_TASK_ID]}