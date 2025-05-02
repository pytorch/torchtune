# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib

_has_ray = importlib.util.find_spec("ray") is not None
_has_vllm = importlib.util.find_spec("vllm") is not None

# Do not import anything unless Python >= 3.10
if _has_ray:
    import ray
    from ray import remote
else:
    # dummy decorator - never used bc tests are skipped
    def remote(*args, **kwargs):
        return lambda cls: cls


import os
from typing import Dict

import pytest
import torch
from omegaconf import OmegaConf
from tests.test_utils import gpu_test, skip_if_lt_python_310


@remote(num_cpus=1, num_gpus=1)
class DummyTrainer:
    def __init__(self, env_vars):
        from transformers import AutoModel

        for k, v in env_vars.items():
            os.environ[k] = str(v)

        torch.cuda.set_device(torch.device("cuda", 0))

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend="nccl", device_id=torch.device("cuda:0")
            )

        self.rank = int(os.environ["RANK"])
        self.model = AutoModel.from_pretrained("gpt2").cuda().bfloat16()

        self.model._apply(lambda t: t.fill_(1.0))

    def get_model_metadata(self):
        return {k: (v.dtype, v.shape) for k, v in self.model.state_dict().items()}

    def register_parameter_server(self, parameter_server):
        self.parameter_server = parameter_server

    def sync_weights(self):
        ray.get(self.parameter_server.acquire_state_dict_lock.remote())
        self.parameter_server.receive_from_trainer.remote()
        for k, v in self.model.state_dict().items():
            torch.distributed.send(v, 1)
        ray.get(self.parameter_server.release_state_dict_lock.remote())


@remote(num_cpus=1, num_gpus=1)
class DummyCollector:
    def __init__(self, weight_receiver, tp_size):
        from torchtune.dev.rl.workers.datacollectors import VLLMWorkerWrapper
        from vllm import LLM
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            VocabParallelEmbedding,
        )

        os.environ["VLLM_USE_V1"] = "0"

        class TestVLLMWorkerWrapper(VLLMWorkerWrapper):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def _check_weights_changed(self, fill_val) -> bool:
                """
                Check if the weights are updated to fill_val.
                """
                weights_updated = True
                for name, p in self.model_runner.model.named_parameters():
                    mod, child_name = name.rsplit(".", 1)
                    parent_module = self.model_runner.model.get_submodule(mod)
                    # TODO: vLLM pads embeddings with zeros, so not the full embedding will be filled with fill_val
                    if not isinstance(parent_module, VocabParallelEmbedding):
                        weights_updated = weights_updated and torch.allclose(
                            p, torch.empty_like(p).fill_(fill_val)
                        )
                return weights_updated

        self.inference_server = LLM(
            "gpt2",
            enforce_eager=True,
            dtype="bfloat16",
            worker_cls=TestVLLMWorkerWrapper,
            tensor_parallel_size=tp_size,
        )

        self.weight_update_receiver = weight_receiver
        self.weight_update_receiver.register_collector(self)

    def update_policy_weights_(self) -> None:
        self.weight_update_receiver()

    def check_weights_changed(self, fill_val) -> bool:
        return self.inference_server.collective_rpc(
            "_check_weights_changed", args=(fill_val,)
        )


@pytest.mark.skipif(not _has_ray or not _has_vllm, reason="requires ray and vllm")
@skip_if_lt_python_310()
class TestParamServer:
    def _get_env_vars(self, rank, world_size) -> Dict[str, str]:
        env_vars = {
            "RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": "29500",
        }
        return env_vars

    def _get_config(self, tp_size: int = 1):
        config_str = f"""
inference:
    tensor_parallel_dim: {tp_size}
        """
        cfg = OmegaConf.create(config_str)
        return cfg

    @pytest.mark.integration_test
    @gpu_test(gpu_count=2)
    def test_receive_from_trainer(self) -> None:
        import ray
        from torchtune.dev.rl.workers import VLLMParameterServer

        ray.init(num_cpus=10, num_gpus=2)

        try:
            trainer_handle = DummyTrainer.remote(self._get_env_vars(0, 2))
            cfg = self._get_config()
            ps_handle = VLLMParameterServer.remote(
                cfg, None, None, self._get_env_vars(1, 2)
            )
            trainer_handle.register_parameter_server.remote(ps_handle)
            model_metadata = ray.get(trainer_handle.get_model_metadata.remote())
            ps_handle.register_model_metadata.remote(model_metadata)
            ps_weights = ray.get(ps_handle._get_server_weights.remote())

            for k, v in model_metadata.items():
                assert ps_weights[k].dtype == v[0]
                assert ps_weights[k].shape == v[1]
                assert torch.equal(ps_weights[k], torch.zeros_like(ps_weights[k]))

            ray.get(trainer_handle.sync_weights.remote())

            updated_ps_weights = ray.get(ps_handle._get_server_weights.remote())
            for k, v in model_metadata.items():
                assert torch.equal(
                    updated_ps_weights[k], torch.ones_like(updated_ps_weights[k])
                )
        finally:
            ray.shutdown()

    @pytest.mark.integration_test
    @gpu_test(gpu_count=4)
    @pytest.mark.parametrize("tp_size", (1, 2))
    def test_send_to_generator(self, tp_size) -> None:
        import ray

        ray.init(num_cpus=10, num_gpus=2 + tp_size)

        try:
            from torchtune.dev.rl.workers import (
                VLLMHFWeightUpdateReceiver,
                VLLMParameterServer,
            )
            from transformers import AutoModel
            from vllm.utils import get_ip, get_open_port

            model = AutoModel.from_pretrained("gpt2").cuda().to(torch.bfloat16)
            model_metadata = {
                k: (v.dtype, v.shape) for k, v in model.state_dict().items()
            }

            trainer_handle = DummyTrainer.remote(self._get_env_vars(0, 2))
            ip, port, worker_idx = get_ip(), get_open_port(), 0
            cfg = self._get_config(tp_size=tp_size)
            ps_handle = VLLMParameterServer.remote(
                cfg, [ip], [port], self._get_env_vars(1, 2)
            )
            weight_receiver = VLLMHFWeightUpdateReceiver(
                ip, port, model_metadata, ps_handle, worker_idx
            )
            collector_handle = DummyCollector.options(num_gpus=tp_size).remote(
                weight_receiver, tp_size
            )

            trainer_handle.register_parameter_server.remote(ps_handle)
            ray.get(ps_handle.register_model_metadata.remote(model_metadata))
            ray.get(trainer_handle.sync_weights.remote())

            ray.get(collector_handle.update_policy_weights_.remote())
            assert all(ray.get(collector_handle.check_weights_changed.remote(1.0)))
        finally:
            ray.shutdown()
