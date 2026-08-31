#!/bin/bash

#SBATCH --job-name=test_initialisation
#SBATCH --output=slurm/slurm-%x-%j.out
#SBATCH --error=slurm/slurm-%x-%j.err
#SBATCH --time=10:00:00
#SBATCH --cpus=2
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1


python -m experiments.benchmark_layer_initialisation.benchmark_conv_growth
