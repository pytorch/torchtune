#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --constraint=volta32gb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# #SBATCH --gpus=8
#SBATCH --gpus-per-node=8
#SBATCH --partition=learnlab
#SBATCH --no-requeue
#SBATCH --exclusive

#SBATCH --job-name=torchtune
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

source /private/home/kwiat/projects/torchtune/.venv/bin/activate

srun torchrun \
--nnodes 1 \
--nproc_per_node 8 \
recipes/r1_full_finetune_distributed.py --config recipes/configs/llama3_2/8B_full_rl_cuda.yaml "$@"
