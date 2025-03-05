#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --constraint=volta32gb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --no-requeue
#SBATCH --exclusive

#SBATCH --job-name=torchtune
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

# /\ Customize SBATCH directives to custommize your hardware

# \/ Customize the virtual env/module load - this assumes a virtual env in root of torchtune
source ../../.venv/bin/activate

srun tune run \
--nnodes 1 \
--nproc_per_node 8 \
full_finetune_distributed --config dev/3B_sft_for_grpo "$@"
