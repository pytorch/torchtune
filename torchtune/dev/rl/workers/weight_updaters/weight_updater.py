# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import ray
from torchrl.collectors import WeightUpdateReceiverBase


class VLLMHFWeightUpdateReceiver(WeightUpdateReceiverBase):
    def __init__(
        self, master_address, master_port, model_metadata, param_server, worker_idx
    ):
        self.master_address = master_address
        self.master_port = master_port
        self.model_metadata = model_metadata
        self.initialized_group = None
        self.param_server = param_server
        self.worker_idx = worker_idx

    def _get_server_weights(self):
        return None

    def _get_local_weights(self):
        # We don't implement this because we let vLLM's update_weights API handle everything for now
        return None

    def _maybe_map_weights(self, server_weights, local_weights):
        # vLLM update_weights function handles the mapping from huggingface
        # so we don't implement this for now
        return None

    def _update_local_weights(self, local_weights, mapped_weights):
        should_update = not ray.get(
            self.param_server._skip_update.remote(self.worker_idx)
        )
        if should_update:
            self.param_server._sync_weights_with_worker.remote(self.worker_idx)
            inference_server = self.collector.inference_server
            if self.initialized_group is None:
                weight_sync_world_size = (
                    inference_server.llm_engine.parallel_config.tensor_parallel_size + 1
                )
                inference_server.collective_rpc(
                    "init_weight_update_group",
                    args=(
                        self.master_address,
                        self.master_port,
                        1,
                        weight_sync_world_size,
                    ),
                )
                self.initialized_group = True

            for k, (dtype, shape) in self.model_metadata.items():
                inference_server.collective_rpc("update_weight", args=(k, dtype, shape))

            inference_server.collective_rpc("update_policy_version")
