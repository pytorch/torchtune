# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import time

import pytest

_has_ray = importlib.util.find_spec("ray") is not None

# Do not import anything if not running on Python >= 3.10
if _has_ray:
    import ray
    from ray import remote
    from ray.util.queue import Queue
    from tensordict import NonTensorData
    from torchtune.dev.rl.datatypes.trajectory import Trajectory
    from torchtune.dev.rl.workers.postprocessing import PostProcessingWorker
else:
    # dummy decorator - never used bc tests are skipped
    def remote(*args, **kwargs):
        return lambda cls: cls


import torch
from omegaconf import OmegaConf
from tests.test_utils import gen_log_file_name, gpu_test, skip_if_lt_python_310

grpo_samples = 4
max_generated_tokens = 32


@remote(num_cpus=1, num_gpus=0)
class DummyTrainer:
    def __init__(self, rollout_queue):
        self.rollout_queue = rollout_queue

    def train(self):
        while True:
            size = self.rollout_queue.qsize()
            if size == 0:
                # TODO: this is a very bad thing
                time.sleep(15.0)
                return
            time.sleep(0.1)


class TestPostProcessingWorker:
    @pytest.fixture
    def log_file(self, tmpdir):
        return gen_log_file_name(tmpdir)

    @pytest.fixture
    def cfg(self, tmpdir, log_file):
        cfg = {
            "device": "cuda",
            "tokenizer": {
                "_component_": "torchtune.models.llama3.llama3_tokenizer",
                "path": "/tmp/test-artifacts/tokenizer_llama3.model",
            },
            "model": {
                "_component_": "torchtune.models.llama3.llama3",
                "vocab_size": 128_256,
                "num_layers": 2,
                "num_heads": 4,
                "embed_dim": 512,
                "max_seq_len": 1024,
                "norm_eps": 1e-5,
                "num_kv_heads": 2,
            },
            "dtype": "fp32",
            "postprocessing": {
                "ref_checkpointer": {
                    "_component_": "torchtune.training.FullModelHFCheckpointer",
                    "checkpoint_dir": "/tmp/test-artifacts/llama3-hf-04232025",
                    "checkpoint_files": ["model.safetensors"],
                    "output_dir": str(tmpdir),
                    "model_type": "LLAMA3",
                },
            },
            "metric_logger": {
                "_component_": "torchtune.training.metric_logging.DiskLogger",
                "log_dir": str(tmpdir),
                "filename": log_file,
            },
            "output_dir": str(tmpdir),
            "inference": {
                "group_size": grpo_samples,
                "batch_size": 1,
                "temperature": 1.0,
            },
            "num_steps": 3,
        }
        return OmegaConf.create(cfg)

    @pytest.mark.integration_test
    @gpu_test(gpu_count=1)
    @pytest.mark.skipif(not _has_ray, reason="requires ray")
    @skip_if_lt_python_310()
    def test_run(self, cfg, log_file):
        ray.init(num_cpus=19, num_gpus=1)

        torch.cuda.set_device(torch.device("cuda", 0))

        try:
            # create queue
            rollout_queue = Queue(
                actor_options={"num_cpus": 10, "num_gpus": 0},
                maxsize=2,
            )
            for i in range(2):
                rollout_queue.put(
                    Trajectory(
                        query_responses=torch.randint(
                            0,
                            128000,
                            (grpo_samples, max_generated_tokens + 20 * (i + 1)),
                        ).to(dtype=torch.long),
                        responses=torch.randint(
                            0, 128000, (grpo_samples, max_generated_tokens)
                        ).to(dtype=torch.long),
                        logprobs=torch.randn(grpo_samples, max_generated_tokens),
                        ref_logprobs=None,
                        query_response_padding_masks=torch.randint(
                            0, 2, (grpo_samples, max_generated_tokens + 20 * (i + 1))
                        ).to(dtype=torch.bool),
                        seq_lens=torch.randint(0, 100, (grpo_samples,)),
                        answers=NonTensorData(["42"] * grpo_samples),
                        policy_version=None,
                        rewards=None,
                        advantages=None,
                        successes=None,
                        reward_metadata=None,
                        sequence_ids=None,
                    )
                )
            replay_buffer = []

            # Ref actor handle
            postprocessing_worker = PostProcessingWorker.remote(
                cfg=cfg,
                rollout_queue=rollout_queue,
                replay_buffer=replay_buffer,
            )
            actor_handles = [postprocessing_worker.run.remote()]

            # dummy trainer to wait for queue to empty
            dummy_trainer = DummyTrainer.remote(rollout_queue=rollout_queue)
            dummy_trainer_handles = [dummy_trainer.train.remote()]
            ray.get(dummy_trainer_handles)

        finally:
            ray.shutdown()
