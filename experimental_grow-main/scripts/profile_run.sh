#!/bin/bash

#SBATCH --job-name=profile_addnist
#SBATCH --output=slurm/slurm-%x-%A_%a.out
#SBATCH --time=1-00:10:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH -p besteffort
#SBATCH --nodelist=titanic-1
# SBATCH --array=[1-4]


if [ "${SLURM_ARRAY_JOB_ID}" ] ; then
    JOB_ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
else
    JOB_ID="${SLURM_JOB_ID}"
fi
echo -e "JOB ID = ${JOB_ID}"
echo -e "NODE NAME = ${SLURMD_NODENAME}"

OUTFILE="temp/profile_${JOB_ID}.scalene.html"

scalene --outfile $OUTFILE --json --html --no-browser experiments/main_graph_network.py --exp_name Debug --neurons 10 --iters 2 --random_growth --train_epochs 10 --job_id "${JOB_ID}" --node_name "${SLURMD_NODENAME}"
