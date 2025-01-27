#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# ---------- SBATCH commands ---------- #
#SBATCH --job-name=torchtune-multi-node
#SBATCH --ntasks=2
#SBATCH --nodes=2
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=96
#SBATCH --partition=train

# ---------- Set env variables ---------- #
# Grab the IP for head node:
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo Node IP: $head_node_ip
export LOGLEVEL=INFO

# You might need to explicitly set the network interface:
# export NCCL_SOCKET_IFNAME=...
# export GLOO_SOCKET_IFNAME=...

export TORCH_DIST_INIT_BARRIER=1

# ---------- Launch training ---------- #
# Adjust sbatch --ntasks and sbatch --nodes above and --nnodes below to your specific node count
srun tune run --nnodes 2 --nproc_per_node 8 --rdzv_id 101 --rdzv_backend c10d --rdzv_endpoint "$head_node_ip:29500" \
    full_finetune_distributed --config ${CONFIG_FILE}
