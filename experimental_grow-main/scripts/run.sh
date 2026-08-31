#!/bin/bash

#SBATCH --job-name=adam
#SBATCH --output=slurm/slurm-%x-%A_%a.out
# SBATCH --error=slurm/slurm-%x-%A_%a.err
#SBATCH --time=3-00:10:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH -p tau
#SBATCH --exclude=margpu018,margpu021

# SBATCH -w margpu024
# SBATCH --nodelist=titanic-1
# SBATCH --array=[1-4]


sleep 5

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

if [ "${SLURM_ARRAY_JOB_ID}" ] ; then
    JOB_ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
else
    JOB_ID="${SLURM_JOB_ID}"
fi
echo -e "JOB ID = ${JOB_ID}"
echo -e "NODE NAME = ${SLURMD_NODENAME}"

# python experiments/main_graph_network_conv.py --exp_name Cifar10-Conv --neurons 20 --iters 5 --train_epochs 10 --job_id "${JOB_ID}" --node_name "${SLURMD_NODENAME}"
# python experiments/vanilla.py --exp_name Vanilla --job_id "${JOB_ID}" --node_name "${SLURMD_NODENAME}"

python experiments/pipeline/run_pipeline.py \
	--growth.neuron_selection_threshold 0.00028 \
	--training.es_abs_delta 0.00122 \
	# # --training.lrate 0.01 --training.momentum 0.99 \
	# # --training.weight_decay 0.0001 --training.optimizer torch.optim.SGD \
	--job_id "${JOB_ID}" --node_name "${SLURMD_NODENAME}"

# wait
