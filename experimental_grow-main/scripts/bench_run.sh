#!/bin/bash

#SBATCH --output=slurm/slurm-%x-%A_%a.out
#SBATCH --time=1-00:10:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH -p gpu-best
#SBATCH --exclude=margpu018,margpu021
#SBATCH --array=[1-5]


STAGGER_SECONDS=2
SLEEP_TIME=$(( (SLURM_ARRAY_TASK_ID - 1) * STAGGER_SECONDS ))
echo "Array task ${SLURM_ARRAY_TASK_ID}: sleeping ${SLEEP_TIME}s before starting"
sleep "${SLEEP_TIME}"

echo "${CUDA_VISIBLE_DEVICES}"
nvidia-smi

if [ "${SLURM_ARRAY_JOB_ID}" ] ; then
    JOB_ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
else
    JOB_ID="${SLURM_JOB_ID}"
fi

echo "JOB ID = ${JOB_ID}"
echo "NODE NAME = ${SLURMD_NODENAME}"
echo "DATASET = ${BENCH_DATASET}"
echo "MODEL = ${BENCH_MODEL}"
echo "CLASSES = ${BENCH_CLASSES}"


python experiments/vanilla/small_net.py \
    --dataset  "${BENCH_DATASET}" \
    --classes  "${BENCH_CLASSES}" \
    --model    "${BENCH_MODEL}" \
    --epochs 150 \
    --augment
