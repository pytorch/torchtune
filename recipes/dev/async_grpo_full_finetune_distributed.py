# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
import os
import time
from typing import Any, Dict

import ray
import torch
from omegaconf import DictConfig, OmegaConf
from ray.util.queue import Queue
from tensordict import TensorDict, TensorDictBase
from tensordict.utils import expand_as_right
from torchrl.data import LazyStackStorage, RayReplayBuffer
from torchtune import config, utils
from torchtune.dev.rl.datatypes import RequestOutput, Trajectory
from torchtune.dev.rl.workers import (
    MetricLoggerWorker,
    PostProcessingWorker,
    SyncLLMCollector,
    TrainingWorker,
    VLLMHFWeightUpdateReceiver,
    VLLMParameterServer,
)
from torchtune.recipe_interfaces import OrchestrationRecipeInterface
from vllm import SamplingParams
from vllm.utils import get_ip, get_open_port

log = utils.get_logger("DEBUG")


def vllm_generate(
    tensordict: TensorDictBase | Trajectory,
    vllm_instance,  # The LLM object
    generate_kwargs: Dict[str, Any] | None = None,
    tokenizer=None,
    text_key: str = "text",
    token_key: str = "tokens",
    token_response_key: str = "tokens_response",
    text_response_key: str = "text_response",
    log_prob_key: str = "log_probs",
    attention_mask_key: str = "attention_mask",
    pad_output: bool = True,
    padding_value: int = -1,
) -> TensorDict:

    args = ()

    if generate_kwargs is None:
        generate_kwargs = {}
    generate_kwargs.setdefault("detokenize", True)
    generate_kwargs.setdefault("prompt_logprobs", False)
    generate_kwargs.setdefault("logprobs", True)
    # Create SamplingParams from generate_kwargs
    sampling_params = SamplingParams(**generate_kwargs)
    kwargs = {"sampling_params": sampling_params, "use_tqdm": False}

    txt = tensordict.get(text_key)
    if not isinstance(txt, (list, str)):
        txt = txt.tolist()
    args = (txt,)

    time_generate_start = time.perf_counter()
    tokens_out = vllm_instance.generate(*args, **kwargs)
    time_generate = time.perf_counter() - time_generate_start
    out = _get_output_tokens_and_log_probs(
        tokens_out, tokenizer, log_prob_key, token_response_key, text_response_key
    )

    if pad_output:
        out = _pad_output_tokens_and_log_probs(
            out, token_response_key, log_prob_key, padding_value
        )

    assert set(out.keys()) == set([token_response_key, log_prob_key, text_response_key])

    # prevent stateless transforms from breaking
    td = tensordict.clone()
    td.update(out, keys_to_update=list(out.keys()))

    # return a td instead of Trajectory here otherwise within torchrl calls like traj["bla"]
    # will be broken
    return td, time_generate


def _pad_output_tokens_and_log_probs(
    tokens_response_td, token_response_key, log_prob_key, padding_value
):
    tokens_response_td = tokens_response_td.densify(
        layout=torch.strided
    ).to_padded_tensor(padding=padding_value)

    padded_values = tokens_response_td[token_response_key] == padding_value
    if padded_values.any():
        lps = tokens_response_td[log_prob_key]
        lps = torch.where(expand_as_right(~padded_values, lps), lps, 1.0)
        tokens_response_td[log_prob_key] = lps

    return tokens_response_td


def _get_output_tokens_and_log_probs(
    tokens_out, tokenizer, log_prob_key, token_response_key, text_response_key
):
    tokens_out = RequestOutput.from_request_output(tokens_out)
    tokens_response_td = tokens_out.outputs._tensordict.select(
        "text", "token_ids", "logprobs", strict=False
    )
    tokens_response_td.rename_key_("token_ids", token_response_key)
    tokens_response_td.rename_key_("text", text_response_key)
    tokens_response_td.rename_key_("logprobs", log_prob_key)
    return tokens_response_td


class RayGRPORecipe(OrchestrationRecipeInterface):
    def setup(self, cfg: DictConfig):
        self.cfg = cfg

        # Store worker counts as instance variables
        self.num_inference_workers = cfg.orchestration.num_inference_workers
        self.num_postprocessing_workers = cfg.orchestration.num_postprocessing_workers
        self.num_training_workers = cfg.orchestration.num_training_workers

        self.vllm_tp_size = cfg.inference.tensor_parallel_dim
        # Initialize queues
        self.rollout_queue = Queue(
            actor_options={"num_cpus": 10, "num_gpus": 0},
            maxsize=self.cfg.inference.queue_maxsize,
        )
        self.replay_buffer = RayReplayBuffer(
            storage=functools.partial(
                LazyStackStorage, max_size=cfg.orchestration.replay_buffer_size
            ),
            batch_size=cfg.training.batch_size,
            remote_config={"num_cpus": 10, "num_gpus": 0},
        )

        # Create workers using config values directly
        self.ref_workers = self._create_ref_workers()
        self.actor_workers = self._create_fsdp_group(
            worker_cls=TrainingWorker,
            fsdp_world_size=self.num_training_workers,
        )
        self.param_server = self._create_param_server(
            parameter_server_cls=VLLMParameterServer,
            fsdp_world_size=self.num_training_workers,
            num_inference_workers=self.num_inference_workers,
        )
        self.rollout_workers = self._create_data_collectors()

        # needs to happens after workers are created
        # or there are conflicts with the placement group
        self._set_metric_logger_to_actors()

    def start_ray(self):
        total_gpus = (
            self.num_inference_workers * self.vllm_tp_size
            + self.num_postprocessing_workers
            + self.num_training_workers
        )
        total_cpus = 32 * total_gpus + 2
        ray.init(num_cpus=total_cpus, num_gpus=total_gpus)
        print("Cluster resources:", ray.cluster_resources())

    def _set_metric_logger_to_actors(self):
        self.metric_logger = MetricLoggerWorker.remote(self.cfg)

        # Collect object references for all remote calls
        set_logger_handles = []
        for worker in self.rollout_workers:
            handle = worker.set_metric_logger.remote(self.metric_logger)
            set_logger_handles.append(handle)
        for worker in self.ref_workers:
            handle = worker.set_metric_logger.remote(self.metric_logger)
            set_logger_handles.append(handle)
        for worker in self.actor_workers:
            handle = worker.set_metric_logger.remote(self.metric_logger)
            set_logger_handles.append(handle)

        # Wait for all set_metric_logger calls to complete
        ray.get(set_logger_handles)
        log.info("Set metric logger to all actors.")

    def _create_fsdp_group(
        self,
        worker_cls,
        fsdp_world_size: int,
    ):
        self.addr, self.port = get_ip(), get_open_port()
        fsdp_workers = []
        world_size = fsdp_world_size + 1
        for i in range(fsdp_world_size):
            env_vars = {
                "RANK": str(i),
                "WORLD_SIZE": world_size,
                "MASTER_ADDR": self.addr,
                "MASTER_PORT": self.port,
            }
            worker = worker_cls.remote(
                self.cfg,
                env_vars,
                self.replay_buffer,
            )
            fsdp_workers.append(worker)

        return fsdp_workers

    def _create_param_server(
        self,
        parameter_server_cls,
        fsdp_world_size: int,
        num_inference_workers: int,
    ):
        world_size = fsdp_world_size + 1
        self.vllm_addresses = [get_ip()] * num_inference_workers
        self.vllm_ports = [get_open_port() for i in range(num_inference_workers)]

        env_vars = {
            "RANK": str(fsdp_world_size),
            "WORLD_SIZE": world_size,
            "MASTER_ADDR": self.addr,
            "MASTER_PORT": self.port,
        }

        parameter_server = parameter_server_cls.options(max_concurrency=5).remote(
            self.cfg, self.vllm_addresses, self.vllm_ports, env_vars
        )

        self.actor_workers[0].register_parameter_server.remote(parameter_server)
        self.model_metadata = ray.get(self.actor_workers[0].get_model_metadata.remote())
        ray.get(parameter_server.register_model_metadata.remote(self.model_metadata))

        return parameter_server

    def _create_ref_worker(self):
        worker = PostProcessingWorker.remote(
            rollout_queue=self.rollout_queue,
            replay_buffer=self.replay_buffer,
            cfg=self.cfg,
        )
        return worker

    def _create_data_collectors(self):
        data_collectors = []

        # This seems to need to be done to prevent some internal checks in torchrl
        vllm_generate.out_keys = [
            "prompt",
            "prompt_tokens",
            "response",
            "response_tokens",
            "prompt_attention_mask",
            "log_probs",
        ]

        weight_update_receivers = [
            VLLMHFWeightUpdateReceiver(
                vllm_master_address,
                vllm_update_port,
                self.model_metadata,
                self.param_server,
                idx,
            )
            for idx, (vllm_master_address, vllm_update_port) in enumerate(
                zip(self.vllm_addresses, self.vllm_ports)
            )
        ]

        for i in range(self.num_inference_workers):
            collector = (
                ray.remote(
                    num_cpus=0,
                    num_gpus=self.cfg.inference.tensor_parallel_dim,
                )(SyncLLMCollector)
                .options(max_concurrency=5)
                .remote(
                    cfg=self.cfg,
                    llm=self.cfg.inference.model,
                    policy=vllm_generate,
                    worker_id=i,
                    dialog_turns_per_batch=1,
                    total_dialog_turns=-1,
                    reset_at_each_iter=True,
                    queue=self.rollout_queue,
                    weight_update_receiver=weight_update_receivers[i],
                )
            )
            data_collectors.append(collector)
        return data_collectors

    def _create_ref_workers(self):
        workers = []
        for i in range(self.num_postprocessing_workers):
            worker = PostProcessingWorker.remote(
                rollout_queue=self.rollout_queue,
                replay_buffer=self.replay_buffer,
                cfg=self.cfg,
                actor_id=i,
            )
            workers.append(worker)
        return workers

    def run(self):
        # Start the workers
        rollout_handles = [worker.run.remote() for worker in self.rollout_workers]
        ref_handles = [worker.run.remote() for worker in self.ref_workers]
        worker_handles = [worker.train.remote() for worker in self.actor_workers]
        ray.get(worker_handles)
        [ray.kill(w) for w in self.rollout_workers + self.ref_workers]
        ray.get(self.actor_workers[0].cleanup.remote())

    def cleanup(self):
        ray.shutdown()


@config.parse
def recipe_main(cfg: DictConfig) -> None:
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)

    if cfg.get("enable_expandable_segments", True):
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    recipe = RayGRPORecipe()
    recipe.setup(cfg)
    recipe.run()
    recipe.cleanup()


if __name__ == "__main__":
    recipe_main()
