#!/usr/bin/env python3

"""
README!! What's going on in this script?

At a high level, this script is grpo_full_finetune_distributed.py but it
1. Uses vLLM for generation instead of torchtune generate
2. Uses ray for orchestrating data parallel actors and vllm actors + unsharded ref actor
3. The dataloader is now owned by the vllm worker (rather than 1 dataloader per FSDP worker)
4. Uses a ray.util.Queue (this wraps a remote actor with a queue) as a "replay buffer"
   for the vllm workers to put their generated tokens into, and the FSDP workers to get them from
5. Of the items in GRPOTrajectory
        a. query_responses: vllm worker computes and puts into queue
        b. logprobs vllm worker puts into queue
        c. ref_logprobs: computed by unsharded RefActor which contains torchtune model
        d. rewards: fsdp worker computes
        e. sucecesses: fsdp worker computes
        f. advantages: fsdp worker computes
        g. masks: fsdp worker computes
        h. position_ids: fsdp worker computes
6. In config, we set ``steps_before_sync``. After ``steps_before_sync * num_data_parallel_worker`` steps
   the vllm worker will "sleep" and spin in a while loop until the FSDP workers are done syncing their weights
7. Weight sync currently blocks the train loop and is done by each fsdp workers .full_tensor (allgather) on each DTensor,
   calling tune_to_hf and then rank 0 broadcasting (+ also calling the vllm collective rpc to make it also issues
   the broadcast and then load weights call)

With this script, I can observe successes + rewards increasing over training steps, which is
a good sign. (See screenshot in Cabernet sprint notes.) But there are several issues with this script:
1. Peak memory usage for the fsdp worker is significantly higher than the original recipe in a fresh conda environmnet.
   This could be because
    a. I turned compile off (it seems to be broken with torch 2.5.1 that vllm requires)
    b. [FIXED] I turned activation checkpointing off (there wasn't an explcit reason for this it just got omitted accidentally)
2. I have an assert that num_vllm_workers == 1 and vllm_tp_size == 1 for now. There's no real reason for this,
   I expect the code to mostly generalize, just need to go over parts where I might have hardcoded this assumption of 1
   vllm worker.
3. Epochs is definitely not handled correctly right now :P
4. There's many FIXMEs peppered through this script

The run command is

    python runnable_recipe_ray_vllm_weight_sync.py --config ../recipes/configs/dev/qwen3B_full_grpo.yaml


"""

import functools
import os
import time
from functools import partial
from typing import Any, Dict, List, Optional, Union

from warnings import warn

import ray
import torch
import torch.distributed
import torch.nn as nn
import torchtune.training as training

from omegaconf import DictConfig, ListConfig, OmegaConf
from ray.util.placement_group import placement_group

from ray.util.queue import Full as QueueFull, Queue
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from readerwriterlock import rwlock
from tensordict import TensorClass, TensorDict

from torch.optim import Optimizer

from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler
from torchrl.collectors import (
    LocalWeightUpdaterBase,
    RemoteWeightUpdaterBase,
    SyncDataCollector,
)
from torchrl.data import LazyStackStorage, RayReplayBuffer
from torchtune import config, generation, modules, rlhf, utils
from torchtune.dev.grpo.rewards import batched_rewards
from torchtune.dev.grpo.types import GRPOStats, GRPOTrajectory
from torchtune.dev.rl.datatypes import RequestOutput, Trajectory
from torchtune.dev.rl.utils import stateless_init_process_group
from torchtune.dev.rl.workers import (
    MetricLoggerActor,
    PyTorchActorModel,
    RefActor,
    SyncLLMCollector,
    VLLMParameterServer,
)
from torchtune.models.qwen2._convert_weights import qwen2_tune_to_hf

from torchtune.training import DummyProfiler, PROFILER_KEY

from vllm.utils import get_ip, get_open_port
from vllm.worker.worker import Worker

log = utils.get_logger("DEBUG")


# =============================================================================================

from typing import Any, Dict

import torch
import vllm
from tensordict import from_dataclass, lazy_stack, TensorClass, TensorDictBase
from tensordict.utils import _zip_strict, expand_as_right
from vllm import LLM, SamplingParams


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


# =============================================================================================


class VLLMHFLocalWeightUpdater(LocalWeightUpdaterBase):
    def __init__(self, master_address, master_port, model_metadata):
        self.master_address = master_address
        self.master_port = master_port
        self.model_metadata = model_metadata
        self.initialized_group = None

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
        inference_server = self.collector.inference_server
        if self.initialized_group is None:
            weight_sync_world_size = (
                inference_server.llm_engine.parallel_config.tensor_parallel_size + 1
            )
            inference_server.collective_rpc(
                "init_weight_update_group",
                args=(self.master_address, self.master_port, 1, weight_sync_world_size),
            )
            self.initialized_group = True

        for k, (dtype, shape) in self.model_metadata.items():
            inference_server.collective_rpc("update_weight", args=(k, dtype, shape))

        inference_server.collective_rpc("update_policy_version")


class RayGRPORecipe:
    def setup(self, cfg):
        self.cfg = cfg

        # Store worker counts as instance variables
        self.num_vllm_workers = cfg.vllm.num_workers
        self.vllm_tp_size = cfg.vllm.tp_size
        self.num_ref_workers = cfg.num_ref_workers
        self.num_fsdp_workers = cfg.num_fsdp_workers

        # Initialize queues
        self.rollout_queue = Queue(
            actor_options={"num_cpus": 10, "num_gpus": 0},
            maxsize=self.cfg.vllm.queue_maxsize,
        )
        self.replay_buffer = RayReplayBuffer(
            storage=functools.partial(
                LazyStackStorage, max_size=cfg.replay_buffer_size
            ),
            batch_size=cfg.batch_size,
            remote_config={"num_cpus": 10, "num_gpus": 0},
        )

        # Create workers using config values directly
        self.ref_workers = self._create_ref_workers()
        self.actor_workers = self._create_fsdp_group(
            worker_cls=PyTorchActorModel,
            fsdp_world_size=self.num_fsdp_workers,
        )
        self.param_server = self._create_param_server(
            parameter_server_cls=VLLMParameterServer,
            fsdp_world_size=self.num_fsdp_workers,
            num_vllm_workers=self.num_vllm_workers,
        )
        self.rollout_workers = self._create_data_collectors()

        # needs to happens after workers are created
        # or there are conflicts with the placement group
        self._set_metric_logger_to_actors()

    def start_ray(self):
        total_gpus = (
            self.num_vllm_workers * self.vllm_tp_size
            + self.num_ref_workers
            + self.num_fsdp_workers
        )
        total_cpus = 32 * total_gpus + 2
        ray.init(num_cpus=total_cpus, num_gpus=total_gpus)
        print("Cluster resources:", ray.cluster_resources())

    def _set_metric_logger_to_actors(self):
        self.metric_logger = MetricLoggerActor.remote(self.cfg)
        # Pass the logger handle to each worker
        for worker in self.rollout_workers:
            worker.set_metric_logger.remote(self.metric_logger)
        for worker in self.ref_workers:
            worker.set_metric_logger.remote(self.metric_logger)
        for worker in self.actor_workers:
            worker.set_metric_logger.remote(self.metric_logger)

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
        num_vllm_workers: int,
    ):
        world_size = fsdp_world_size + 1
        self.vllm_addresses = [get_ip()] * num_vllm_workers
        self.vllm_ports = [get_open_port() for i in range(num_vllm_workers)]

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
        worker = RefActor.remote(
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

        vllm_addresses = self.vllm_addresses
        vllm_ports = self.vllm_ports

        local_weight_updaters = [
            VLLMHFLocalWeightUpdater(
                vllm_master_address, vllm_update_port, self.model_metadata
            )
            for vllm_master_address, vllm_update_port in zip(vllm_addresses, vllm_ports)
        ]

        for i in range(self.num_vllm_workers):

            pg_inference = placement_group(
                [{"GPU": 1, "CPU": 0}] * self.cfg.vllm.tp_size
            )
            ray.get(pg_inference.ready())
            scheduling_inference = PlacementGroupSchedulingStrategy(
                placement_group=pg_inference,
                placement_group_capture_child_tasks=True,
            )

            collector = (
                ray.remote(
                    num_cpus=0, num_gpus=0, scheduling_strategy=scheduling_inference
                )(SyncLLMCollector)
                .options(max_concurrency=5)
                .remote(
                    cfg=self.cfg,
                    llm="Qwen/Qwen2.5-3B",
                    policy=vllm_generate,
                    worker_id=i,
                    dialog_turns_per_batch=1,
                    total_dialog_turns=1000,
                    reset_at_each_iter=True,
                    queue=self.rollout_queue,
                    local_weight_updater=local_weight_updaters[i],
                    remote_weight_updater=self.param_server,
                )
            )
            # TODO: Currently we register a handle to the collector to the parameter server
            # this will be cleaned up when we make the local_weight_updater remotely call
            # the param server
            ray.get(self.param_server.register_collector.remote(i, collector))
            data_collectors.append(collector)
        return data_collectors

    def _create_ref_workers(self):
        workers = []
        for i in range(self.num_ref_workers):
            worker = RefActor.remote(
                rollout_queue=self.rollout_queue,
                replay_buffer=self.replay_buffer,
                cfg=self.cfg,
                actor_id=i,
            )
            workers.append(worker)
        return workers

    def train(self):
        rollout_handles = [worker.run.remote() for worker in self.rollout_workers]
        ref_handles = [worker.run.remote() for worker in self.ref_workers]
        worker_handles = [worker.train.remote() for worker in self.actor_workers]
        ray.get(rollout_handles + ref_handles + worker_handles)
        ray.get(self.actor_workers[0].cleanup.remote())

    def stop_ray(self):
        ray.shutdown()


@config.parse
def recipe_main(cfg: DictConfig) -> None:
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)

    if cfg.get("enable_expandable_segments", True):
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    recipe = RayGRPORecipe()
    recipe.setup(cfg)
    recipe.train()
    recipe.stop_ray()


if __name__ == "__main__":
    recipe_main()
