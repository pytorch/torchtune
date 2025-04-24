# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib

SKIP_FILE = not (
    importlib.util.find_spec("ray") is not None
    and importlib.util.find_spec("transformers") is not None
    and importlib.util.find_spec("vllm") is not None
)


if not SKIP_FILE:

    import os

    import pytest
    import ray
    import torch
    from omegaconf import OmegaConf
    from tests.test_utils import gpu_test

    from torchtune.dev.rl.workers import VLLMHFWeightUpdateReceiver, VLLMParameterServer
    from torchtune.dev.rl.workers.datacollectors import VLLMWorkerWrapper
    from transformers import AutoModel
    from vllm import LLM

    @ray.remote(num_cpus=1, num_gpus=1)
    class DummyTrainer:
        def __init__(self, env_vars):
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

    @ray.remote(num_cpus=1, num_gpus=1)
    class DummyCollector:
        def __init__(self, weight_receiver, tp_size):
            os.environ["VLLM_USE_V1"] = "0"

            self.inference_server = LLM(
                "gpt2",
                enforce_eager=True,
                dtype="bfloat16",
                worker_cls=VLLMWorkerWrapper,
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

    class TestParamServer:
        def _get_env_vars(self, rank):
            env_vars = {
                "RANK": rank,
                "WORLD_SIZE": 2,
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": "29500",
            }
            return env_vars

        def _get_config(self, tp_size: int = 1):
            config_str = f"""
vllm:
    tp_size: {tp_size}
            """
            cfg = OmegaConf.create(config_str)
            return cfg

        @gpu_test(gpu_count=2)
        def test_receive_from_trainer(self) -> None:
            ray.init(num_cpus=10, num_gpus=2)

            try:
                trainer_handle = DummyTrainer.remote(self._get_env_vars(0))
                cfg = self._get_config()
                ps_handle = VLLMParameterServer.remote(
                    cfg, None, None, self._get_env_vars(1)
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

        @gpu_test(gpu_count=4)
        @pytest.mark.parametrize("tp_size", (1, 2))
        def test_send_to_generator(self, tp_size) -> None:
            ray.init(num_cpus=10, num_gpus=4)

            try:
                from vllm.utils import get_ip, get_open_port

                model = AutoModel.from_pretrained("gpt2").cuda().to(torch.bfloat16)
                model_metadata = {
                    k: (v.dtype, v.shape) for k, v in model.state_dict().items()
                }

                trainer_handle = DummyTrainer.remote(self._get_env_vars(0))
                ip, port, worker_idx = get_ip(), get_open_port(), 0
                cfg = self._get_config(tp_size=tp_size)
                ps_handle = VLLMParameterServer.remote(
                    cfg, [ip], [port], self._get_env_vars(1)
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
