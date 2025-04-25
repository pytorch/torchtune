# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

import ray
import torch
from readerwriterlock import rwlock
from torchtune import utils
from torchtune.dev.rl.utils import stateless_init_process_group

log = utils.get_logger("DEBUG")


@ray.remote(num_cpus=4, num_gpus=1)
class VLLMParameterServer:
    def __init__(self, cfg, vllm_master_addresses, vllm_master_ports, env_vars):
        super().__init__()
        self.cfg = cfg
        self.vllm_master_addresses = vllm_master_addresses
        self.vllm_master_ports = vllm_master_ports
        self.vllm_comm_groups = dict()
        self.vllm_weight_versions = dict()
        self.vllm_worker_handles = dict()

        torch.cuda.set_device(torch.device("cuda", 0))

        for var in env_vars:
            os.environ[var] = str(env_vars[var])

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend="nccl", device_id=torch.device("cuda:0")
            )

        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        assert self.rank == self.world_size - 1

    def register_model_metadata(self, model_metadata):
        self.model_metadata = model_metadata
        self.state_dict = dict()
        for k, (dtype, shape) in model_metadata.items():
            self.state_dict[k] = torch.zeros(shape, dtype=dtype, device="cuda")
        self.state_dict_lock = rwlock.RWLockFairD()
        self.version = 0
        self.version_tensor = torch.tensor([0], device="cuda")

    def acquire_state_dict_lock(self):
        self.write_lock = self.state_dict_lock.gen_wlock()
        self.write_lock.acquire()

    def release_state_dict_lock(self):
        self.version += 1
        self.version_tensor += 1
        torch.cuda.synchronize()
        self.write_lock.release()

    def _get_server_weights(self):
        return self.state_dict

    def _maybe_map_weights(self, server_weights):
        return server_weights

    def _skip_update(self, worker_id):
        if self.version == 0:
            return True
        if worker_id not in self.vllm_weight_versions:
            return False
        if self.vllm_weight_versions[worker_id] == self.version:
            log.info(
                f"Skipping update for {worker_id=}, {self.version=}, {self.vllm_weight_versions[worker_id]=}"
            )
            return True
        return False

    def _init_model_update_group(self, worker_id):
        vllm_tp_size = self.cfg.inference.tensor_parallel_dim
        weight_sync_world_size = vllm_tp_size + 1
        model_update_group = stateless_init_process_group(
            self.vllm_master_addresses[worker_id],
            self.vllm_master_ports[worker_id],
            0,
            weight_sync_world_size,
            torch.device("cuda:0"),
        )
        self.vllm_comm_groups[worker_id] = model_update_group

    def _sync_weights_with_worker(self, worker_id: int):
        server_weights = self._maybe_map_weights(self._get_server_weights())
        if worker_id not in self.vllm_comm_groups:
            self._init_model_update_group(worker_id)
        read_lock = self.state_dict_lock.gen_rlock()
        read_lock.acquire()
        for i, k in enumerate(server_weights.keys()):
            self.vllm_comm_groups[worker_id].broadcast(
                server_weights[k], src=0, stream=torch.cuda.current_stream()
            )
        self.vllm_comm_groups[worker_id].broadcast(
            self.version_tensor, src=0, stream=torch.cuda.current_stream()
        )
        torch.cuda.synchronize()
        self.vllm_weight_versions[worker_id] = self.version
        read_lock.release()

    def receive_from_trainer(self):
        for _, v in self.state_dict.items():
            torch.distributed.recv(v, src=0)
