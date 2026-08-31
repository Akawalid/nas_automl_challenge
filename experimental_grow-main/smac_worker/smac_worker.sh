#!/bin/bash
#SBATCH --job-name=smac-job
#SBATCH --output=slurm/slurm-%x-%A_%a.out
#SBATCH --time=2-00:10:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH -p tau
#SBATCH --exclude=margpu018,margpu021


# Load your env
source ~/.bashrc
conda activate gromo

nvidia-smi


if [ "${SLURM_ARRAY_JOB_ID}" ] ; then
    JOB_ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
else
    JOB_ID="${SLURM_JOB_ID}"
fi
echo -e "JOB ID = ${JOB_ID}"
echo -e "NODE NAME = ${SLURMD_NODENAME}"


python smac_worker/smac_worker.py --config='{CONFIG}' --result='{RESULT}' --job_id "${JOB_ID}" --node_name "${SLURMD_NODENAME}"
