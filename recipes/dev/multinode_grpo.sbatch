#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# ---------- SBATCH commands ---------- #
#SBATCH --time=72:00:00
#SBATCH --job-name=torchtune-multi-node
#SBATCH --constraint=volta32gb
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=80
#SBATCH --output=slurm_logs/%j/%N.out
#SBATCH --error=slurm_logs/%j/%N.err


# ---------- Set env variables ---------- #
# Grab the IP for head node:
# You may need to set this to the fully qualified domain name of your head node
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo Node IP: $head_node_ip

# You might need to explicitly set the network interface for distributed backends:
# export NCCL_SOCKET_IFNAME=...
# export GLOO_SOCKET_IFNAME=...

export TORCH_DIST_INIT_BARRIER=1
export LOGLEVEL=INFO

# ---------- Launch training ---------- #
# You probably want to load in a virtual env w/ conda...
# module load conda
# conda activate torchtune
# ...or venv
# source torchtune/bin/activate

source ../../.venv/bin/activate

# Adjust sbatch --ntasks and sbatch --nodes above and --nnodes below to your specific node count
srun --export=ALL,OMP_NUM_THREADS=8 tune run --nnodes 2 --nproc_per_node 8 --rdzv_id 101 --rdzv_backend c10d --rdzv_endpoint "$head_node_ip:29500" \
    dev/grpo_full_finetune_distributed --config dev/3B_full_grpo "$@"
