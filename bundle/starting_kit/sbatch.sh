#!/bin/bash
#SBATCH --account=tau
#SBATCH --partition=gpu-best
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=my_job
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# your commands here
make all submission=nas-challenge-submission-main/submission_gromo