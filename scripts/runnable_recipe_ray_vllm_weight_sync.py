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
from torchtune.models.qwen2._convert_weights import qwen2_tune_to_hf

from torchtune.training import DummyProfiler, PROFILER_KEY

from vllm.utils import get_ip, get_open_port
from vllm.worker.worker import Worker

log = utils.get_logger("DEBUG")


class Trajectory(TensorClass["nocast"]):
    query_responses: torch.Tensor
    responses: torch.Tensor
    logprobs: torch.Tensor
    ref_logprobs: torch.Tensor
    query_response_padding_masks: torch.Tensor
    seq_lens: torch.Tensor
    answers: torch.Tensor
    policy_version: int
    rewards: torch.Tensor
    advantages: torch.Tensor
    successes: torch.Tensor
    reward_metadata: Dict[str, List[str]]


def stateless_init_process_group(
    master_address: str,
    master_port: int,
    rank: int,
    world_size: int,
    device: torch.device,
):
    """
    vLLM provides `StatelessProcessGroup` to create a process group
    without considering the global process group in torch.distributed.
    It is recommended to create `StatelessProcessGroup`, and then initialize
    the data-plane communication (NCCL) between external (train processes)
    and vLLM workers.
    """
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup

    pg = StatelessProcessGroup.create(
        host=master_address, port=master_port, rank=rank, world_size=world_size
    )

    pynccl = PyNcclCommunicator(pg, device=device)
    return pynccl


@ray.remote(num_cpus=8, num_gpus=1)
class RefActor:
    def __init__(self, *args, **kwargs):
        assert "rollout_queue" in kwargs, "Must pass queue to vLLMRefActor"
        assert "replay_buffer" in kwargs, "Must pass replay_buffer to vLLMRefActor"
        assert "cfg" in kwargs, "Must pass cfg to vLLMRefActor"

        self.actor_id = kwargs.pop("actor_id", -1)
        self._is_actor_zero = self.actor_id == 0

        self.cfg = kwargs.pop("cfg")
        self.rollout_queue = kwargs.pop("rollout_queue")
        self.replay_buffer = kwargs.pop("replay_buffer")
        self._device = utils.get_device(device=self.cfg.device)
        self._tokenizer = config.instantiate(self.cfg.tokenizer)
        self._dtype = training.get_dtype(self.cfg.dtype, device=self._device)
        ref_checkpoint_dict = self.load_ref_checkpoint(
            cfg_ref_checkpointer=self.cfg.ref_checkpointer
        )
        self._ref_model = self._setup_model(
            self.cfg.model, ref_checkpoint_dict[training.MODEL_KEY]
        )
        self._temperature = self.cfg.temperature

        self.metric_logger = None  # Placeholder for the logger

        self.grpo_samples = self.cfg.grpo_samples
        self.vllm_batch_size = self.cfg.vllm.batch_size

        device_type = self.cfg.device
        self._log_peak_memory_stats = self.cfg.get("log_peak_memory_stats", True)
        if self._log_peak_memory_stats and device_type != "cuda":
            log.info(
                "log_peak_memory_stats was set to True, however, training does not use cuda. Setting log_peak_memory_stats=False."
            )
            self._log_peak_memory_stats = False

        if self._is_actor_zero:
            memory_stats = training.get_memory_stats(device=self._device)
            training.log_memory_stats(memory_stats)

        self.STOP_TOKENS_TENSOR = torch.tensor(
            self._tokenizer.stop_tokens, device=self._device
        )

    def set_metric_logger(self, logger):
        """Store the MetricLoggerActor handle."""
        if self._is_actor_zero:
            print(f"setting metric logger {logger} for actor id", self.actor_id)
            self._metric_logger = logger

    def load_ref_checkpoint(self, cfg_ref_checkpointer: DictConfig) -> Dict[str, Any]:
        """
        Extract the reference checkpoint state from file and validate.
        """
        self._ref_checkpointer = config.instantiate(
            cfg_ref_checkpointer, resume_from_checkpoint=False
        )

        ref_checkpoint_dict = self._ref_checkpointer.load_checkpoint()

        return ref_checkpoint_dict

    def _setup_model(self, cfg_model, ref_model_state_dict):
        from torchtune.training import disable_dropout

        with training.set_default_dtype(self._dtype), torch.device("meta"):
            ref_model = config.instantiate(cfg_model)

        with training.set_default_dtype(self._dtype), self._device:
            for m in ref_model.modules():
                if hasattr(m, "rope_init"):
                    m.rope_init()

        for k, v in ref_model_state_dict.items():
            ref_model_state_dict[k] = v.to(self._device)

        ref_model.load_state_dict(ref_model_state_dict, assign=True, strict=True)

        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

        # Ensure no params and buffers are on meta device
        training.validate_no_params_on_meta_device(ref_model)

        disable_dropout(ref_model)
        print("done setting up ref model")
        return ref_model

    def _log_metrics(
        self,
        step_idx,
        time_total_ref_step,
        time_model_running,
        time_waiting_buffer,
        # full_queue_data_discard,
        rollout_queue_size,
        rewards_mean,
        successes_mean,
        rewards_mean_per_func,
        successes_mean_per_func,
        reward_metadata,
    ):
        """Log metrics for the RefActor, only on actor zero."""
        if not self._is_actor_zero:
            return

        log_dict = {}
        if self._log_peak_memory_stats:
            memory_stats = training.get_memory_stats(device=self._device)
            log_dict.update(
                {
                    f"ref_actor_performance/memory/{k}": v
                    for k, v in memory_stats.items()
                }
            )

        pct_time_model_running = (
            (time_model_running / time_total_ref_step) * 100
            if time_total_ref_step > 0
            else 0
        )
        pct_time_waiting_buffer = (
            (time_waiting_buffer / time_total_ref_step) * 100
            if time_total_ref_step > 0
            else 0
        )

        log_dict.update(
            {
                "ref_actor_performance/time_total_ref_step (s)": time_total_ref_step,
                "ref_actor_performance/time_model_running (s)": time_model_running,
                "ref_actor_performance/pct_time_model_running (%)": pct_time_model_running,
                "ref_actor_performance/time_waiting_buffer (s)": time_waiting_buffer,
                "ref_actor_performance/pct_time_waiting_buffer (%)": pct_time_waiting_buffer,
                # "queues/ref_actor_full_queue_data_discard": full_queue_data_discard,
                "queues/rollout_queue_size": rollout_queue_size,
            }
        )

        log_dict.update(
            {
                "ref_actor_rewards/rewards_mean": rewards_mean.item(),
                "ref_actor_rewards/successes_mean": successes_mean.item(),
            }
        )

        # # TODO we should encode this in the dataclass instead of keeping a dict
        # # otherwise we end up with a list of identical dicts
        # assert all(
        #     metadata["func_names"] == reward_metadata[0]["func_names"]
        #     for metadata in reward_metadata
        # ), "Function names in reward_metadata are not consistent across all entries"
        # function_names = reward_metadata[0]["func_names"]

        function_names = reward_metadata["func_names"]

        # Per-function rewards and successes
        for func_name, func_mean in zip(function_names, rewards_mean_per_func):
            log_dict[
                f"ref_actor_rewards/rewards_func_{func_name}_mean"
            ] = func_mean.item()
        for func_name, func_mean in zip(function_names, successes_mean_per_func):
            log_dict[
                f"ref_actor_rewards/successes_func_{func_name}_mean"
            ] = func_mean.item()

        ray.get(self._metric_logger.log_dict.remote(log_dict, step=step_idx))

    def run(self):
        import time

        log.info("running ref actor")
        idx = 0
        while True:
            if idx == self.cfg.num_steps:
                break

            # Start measuring total step time
            time_step_start = time.perf_counter()
            trajectory = None
            if self._is_actor_zero:
                rollout_queue_size = self.rollout_queue.qsize()
            while trajectory is None:
                try:
                    if self._is_actor_zero:
                        log.info("Getting from rollout_queue queue.")
                    trajectory = self.rollout_queue.get(timeout=0.5)
                    trajectory = trajectory.to(self._device)
                except ray.util.queue.Empty:
                    trajectory = None
                    time.sleep(0.1)
            time_wait_end = time.perf_counter()
            time_waiting_buffer = time_wait_end - time_step_start

            context_length = (
                trajectory.query_responses.shape[1] - trajectory.responses.shape[1]
            )

            masks = generation.get_causal_mask_from_padding_mask(
                trajectory.query_response_padding_masks
            )
            position_ids = generation.get_position_ids_from_padding_mask(
                trajectory.query_response_padding_masks
            )

            # Reset GPU memory stats before model_running
            torch.cuda.reset_peak_memory_stats()

            time_grpo_steps_start = time.perf_counter()
            with torch.no_grad():
                ref_logits = self._ref_model(
                    trajectory.query_responses, input_pos=position_ids, mask=masks
                )
            time_model_running = time.perf_counter() - time_grpo_steps_start

            ref_logits = rlhf.truncate_sequence_for_logprobs(ref_logits, context_length)
            ref_logprobs = rlhf.batched_logits_to_logprobs(
                ref_logits, trajectory.responses, self._temperature
            )

            batch_size = self.cfg.vllm.batch_size  # B
            group_size = self.grpo_samples  # G

            del ref_logits, position_ids, masks
            # masking of ref_logprobs is done in grpo_step

            # Extract components from raw trajectory: these have size [B * G, T]
            print(f"Extracting components from raw trajectory: {trajectory}")
            query_responses = trajectory.query_responses
            responses = trajectory.responses
            query_response_padding_masks = trajectory.query_response_padding_masks
            seq_lens = trajectory.seq_lens
            answers = trajectory.answers  # list[str] of len (B * G)
            answers = [
                answers[i : i + self.grpo_samples]
                for i in range(0, len(answers), self.grpo_samples)
            ]  # list[list[str]] of len [B, G]. Basically a reshape

            # Compute padded tokens percentage
            total_tokens = query_responses.numel()
            padded_tokens = (query_responses == self._tokenizer.pad_id).sum().item()
            padded_tokens_percentage = (
                (padded_tokens / total_tokens) * 100 if total_tokens > 0 else 0
            )
            number_of_tokens = seq_lens.sum().item()

            # Truncate sequences at first stop token
            (
                response_padding_masks,
                responses,
            ) = rlhf.truncate_sequence_at_first_stop_token(
                responses,
                self.STOP_TOKENS_TENSOR.to(self._device),
                self._tokenizer.pad_id,
            )

            # Generate masks and position IDs
            masks = generation.get_causal_mask_from_padding_mask(
                query_response_padding_masks
            )
            position_ids = generation.get_position_ids_from_padding_mask(
                query_response_padding_masks
            )
            context_length = query_responses.shape[1] - responses.shape[1]
            del query_response_padding_masks

            # Compute rewards
            responses = responses.reshape(batch_size, group_size, -1)
            rewards_by_fn, successes_by_fn, reward_metadata = batched_rewards(
                self._tokenizer, responses, answers, device=self._device
            )  # These are (B, G, num_funcs)

            # Compute advantages: B, G, num_funcs -> B, G
            group_rewards = rewards_by_fn.sum(-1)
            group_successes = successes_by_fn.sum(-1)

            # To compute advantage, subtract the mean of the group rewards from each group reward
            group_advantages = (group_rewards - group_rewards.mean(1, keepdim=True)) / (
                group_rewards.std(1, keepdim=True) + 1e-4
            )  # (B, G)

            # Repack trajectory with policy_version

            trajectory = Trajectory(
                query_responses=trajectory.query_responses,
                responses=trajectory.responses,
                logprobs=trajectory.logprobs,
                ref_logprobs=ref_logprobs,
                query_response_padding_masks=trajectory.query_response_padding_masks,
                seq_lens=trajectory.seq_lens,
                answers=trajectory.answers,
                policy_version=trajectory.policy_version,
                rewards=rewards_by_fn.reshape(
                    batch_size * group_size, -1
                ),  # (B, G, num_funcs)
                advantages=group_advantages.reshape(batch_size * group_size),  # (B, G)
                successes=successes_by_fn.reshape(
                    batch_size * group_size, -1
                ),  # (B, G, num_funcs)
                reward_metadata=reward_metadata,
                batch_size=batch_size * group_size,
            )

            log.info(f"Constructed trajectory: {trajectory}")
            # Move tensors to CPU before putting into the queue
            trajectory = trajectory.cpu()

            # Update circular queue
            self.replay_buffer.extend(trajectory)

            # End of step timing
            time_total_ref_step = time.perf_counter() - time_step_start

            # Calculate mean rewards and successes for logging
            rewards_mean_per_func = rewards_by_fn.mean(dim=(0, 1)).cpu()
            successes_mean_per_func = successes_by_fn.mean(dim=(0, 1)).cpu()
            rewards_mean = rewards_mean_per_func.mean()
            successes_mean = successes_mean_per_func.mean()
            # log metrics
            if self._is_actor_zero:
                self._log_metrics(
                    step_idx=idx,
                    time_total_ref_step=time_total_ref_step,
                    time_model_running=time_model_running,
                    time_waiting_buffer=time_waiting_buffer,
                    # TODO: what should we do with this? We can log the total number of elements written in the buffer instead
                    # full_queue_data_discard=full_queue_data_discard,
                    rollout_queue_size=rollout_queue_size,
                    rewards_mean=rewards_mean,
                    successes_mean=successes_mean,
                    rewards_mean_per_func=rewards_mean_per_func,
                    successes_mean_per_func=successes_mean_per_func,
                    reward_metadata=reward_metadata,
                )

            torch.cuda.empty_cache()

            idx += 1


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


CompletionOutput_tc = from_dataclass(vllm.outputs.CompletionOutput)


class RequestOutput(TensorClass["nocast"]):
    request_id: str
    prompt: str
    prompt_token_ids: str
    prompt_logprobs: str
    outputs: str
    finished: str
    metrics: str
    lora_request: str
    encoder_prompt: str
    encoder_prompt_token_ids: str
    num_cached_tokens: str

    def __post_init__(self):
        def get_logprob(output):
            t = []
            for v, tid in zip(output.logprobs, output.token_ids):
                t.append(
                    v[tid]["logprob"] if v[tid].get("logprob") is not None else 1.0
                )
            return torch.tensor(t)

        def postproc(output):
            if output.logprobs:
                output.logprobs = get_logprob(output)
            output.token_ids = torch.tensor(output.token_ids)
            return output

        if isinstance(self.outputs, list):
            outputs = self.outputs
            outputs = [
                postproc(from_dataclass(output, dest_cls=CompletionOutput_tc))
                for output in outputs
            ]
            if len(outputs) == 1:
                self.outputs = outputs[0]
            else:
                self.outputs = maybe_dense_stack(outputs)
            if self.prompt_logprobs is not None:
                self.prompt_logprobs = torch.tensor(
                    [
                        v[tid].logprob if v is not None else 0.0
                        for v, tid in _zip_strict(
                            self.prompt_logprobs, self.prompt_token_ids
                        )
                    ]
                )
            self.prompt_token_ids = torch.tensor(self.prompt_token_ids)
            self.num_cached_tokens = torch.tensor(self.num_cached_tokens)

    @classmethod
    def from_request_output(cls, requests):
        out = lazy_stack(
            [
                cls(
                    request_id=request.request_id,
                    prompt=request.prompt,
                    prompt_token_ids=request.prompt_token_ids,
                    prompt_logprobs=request.prompt_logprobs,
                    outputs=request.outputs,
                    finished=request.finished,
                    metrics=request.metrics,
                    lora_request=request.lora_request,
                    encoder_prompt=request.encoder_prompt,
                    encoder_prompt_token_ids=request.encoder_prompt_token_ids,
                    num_cached_tokens=request.num_cached_tokens,
                )
                for request in requests
            ]
        )
        return out


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


# not decorating with @ray.remote here because num_gpus should vary based on tp_size
class LLMCollector(SyncDataCollector):
    """A simplified version of SyncDataCollector for LLM inference."""

    def __init__(
        self,
        cfg,
        llm,
        policy,
        queue,
        worker_id,
        *,
        dialog_turns_per_batch: int = -1,
        # -1 is never ending (until shutdown)
        total_dialog_turns: int = -1,
        async_envs: bool = False,
        reset_at_each_iter: bool = False,
        local_weight_updater: LocalWeightUpdaterBase | None = None,
        remote_weight_updater: RemoteWeightUpdaterBase | None = None,
    ):
        if async_envs:
            raise NotImplementedError

        self.cfg = cfg
        self.rollout_queue = queue
        self.worker_id = worker_id
        self._is_collector_zero = self.worker_id == 0

        self.tp_size = self.cfg.vllm.tp_size
        self.batch_size = self.cfg.vllm.batch_size

        self.inference_server = LLM(
            model="Qwen/Qwen2.5-3B",
            enforce_eager=True,
            enable_chunked_prefill=True,
            dtype="bfloat16",
            worker_cls=VLLMWorkerWrapper,
            tensor_parallel_size=self.tp_size,
            distributed_executor_backend="ray",
        )

        # local import below LLM call to avoid vLLM no CUDA GPUs available error
        from torchtune import config

        self._tokenizer = config.instantiate(self.cfg.tokenizer)

        policy_kwargs = {
            "generate_kwargs": dict(
                n=1,
                max_tokens=self.cfg.max_generated_tokens,
                temperature=self.cfg.temperature,
            ),
            "pad_output": True,
            "padding_value": self._tokenizer.pad_id,
        }
        self.policy_kwargs = policy_kwargs

        collate_name = self.cfg.get(
            "collate_fn", "torchtune.dev.grpo.data.padded_collate_rl"
        )
        dataloader = self._setup_data(
            self.cfg.dataset,
            self.cfg.get("shuffle", True),
            self.batch_size,
            collate_name,
            dataloader_state_dict=None,
        )

        # local import below LLM call to avoid vLLM no CUDA GPUs available error
        from torchrl.envs import LLMEnv

        env = LLMEnv.from_dataloader(
            dataloader=dataloader,
            tokenizer=None,
            str2str=True,
            batch_size=self.batch_size,
            repeats=self.cfg.grpo_samples,
        )

        super().__init__(
            create_env_fn=env,
            policy=policy,
            frames_per_batch=dialog_turns_per_batch,
            total_frames=total_dialog_turns,
            local_weight_updater=local_weight_updater,
            remote_weight_updater=remote_weight_updater,
            reset_at_each_iter=reset_at_each_iter,
            use_buffers=False,
            # This argument allows a non-TensorDictModule policy to be assumed
            # to be compatible with the collector
            trust_policy=True,
        )

        log.info("done init LLMCollector")

    @property
    def remote_weight_updater(self) -> RemoteWeightUpdaterBase:
        return self._remote_weight_updater

    @remote_weight_updater.setter
    def remote_weight_updater(self, value: RemoteWeightUpdaterBase | None):
        self._remote_weight_updater = value

    def _postprocess_for_queue(self, data):
        # local import to avoid vLLM no CUDA GPUs available error
        from torchtune import training

        data = data.squeeze()
        query_responses = torch.cat([data["tokens"], data["tokens_response"]], dim=-1)
        prompt_tokens = data["tokens"]
        response_tokens = data["tokens_response"]
        logprobs = data["log_probs"]
        query_response_padding_masks = torch.ne(query_responses, self._tokenizer.pad_id)
        answers = data["answers"]
        if hasattr(
            self.inference_server.llm_engine.model_executor.driver_worker.worker,
            "policy_version",
        ):
            policy_version = (
                self.inference_server.llm_engine.model_executor.driver_worker.worker.policy_version.item()
            )
        else:
            policy_version = 0

        response_padding_masks = torch.eq(response_tokens, self._tokenizer.pad_id)
        seq_lens = training.get_unmasked_sequence_lengths(response_padding_masks)
        del response_padding_masks

        postprocessed_results = Trajectory(
            query_responses=query_responses,
            responses=response_tokens,
            logprobs=logprobs,
            ref_logprobs=None,
            query_response_padding_masks=query_response_padding_masks,
            seq_lens=seq_lens,
            answers=answers,
            policy_version=policy_version,
            rewards=None,
            advantages=None,
            successes=None,
            reward_metadata=None,
        )

        total_generated_tokens = seq_lens.sum().item()
        return postprocessed_results, total_generated_tokens

    def update_policy_weights_(
        self,
        policy_weights: TensorDictBase | None = None,
        *,
        worker_ids: int | list[int] | torch.device | list[torch.device] | None = None,
        **kwargs,
    ) -> None:
        self.local_weight_updater(policy_weights, **kwargs)

    def set_metric_logger(self, logger):
        """Store the MetricLoggerActor handle."""
        if self._is_collector_zero:
            self._metric_logger = logger

    def _log_metrics(
        self,
        step_idx,
        time_total_rollout,
        time_generate,
        total_generated_tokens,
        # full_queue_data_discard,
        gpu_memory,
    ):
        """Log metrics for the LLMCollector, only on collector zero."""
        if not self._is_collector_zero:
            return

        pct_time_model_running = (
            (time_generate / time_total_rollout) * 100 if time_total_rollout > 0 else 0
        )
        tokens_per_second = (
            total_generated_tokens / time_generate if time_generate > 0 else 0
        )
        div_gib = 1024**3

        log_dict = {
            "vllm_actor_performance/total_rollout_time (s)": time_total_rollout,
            "vllm_actor_performance/pct_time_model_running (%)": pct_time_model_running,
            "vllm_actor_performance/tokens_per_second": tokens_per_second,
            "vllm_actor_performance/gpu_memory_peak_allocated (GiB)": gpu_memory[
                "allocated"
            ]
            / div_gib,
            "vllm_actor_performance/gpu_memory_peak_reserved (GiB)": gpu_memory[
                "reserved"
            ]
            / div_gib,
            "vllm_actor_performance/gpu_memory_peak_active (GiB)": gpu_memory["active"]
            / div_gib,
            # "queues/vllm_full_queue_data_discard": full_queue_data_discard,
            "queues/rollout_queue_size": self.rollout_queue.qsize(),
        }

        ray.get(self._metric_logger.log_dict.remote(log_dict, step=step_idx))

    def run(self):
        num_steps = (self.cfg.num_steps // self.cfg.vllm.num_workers) + 1
        for i in range(num_steps):
            self.rollout(i)
            if i % self.cfg.vllm.steps_before_sync == 0:
                log.info(f"{self.worker_id} about to update weights")
                ray.get(
                    self.remote_weight_updater.update_weights.remote(
                        weights=None, worker_ids=self.worker_id
                    )
                )

    def rollout(self, idx) -> TensorDictBase:
        if self.reset_at_each_iter or self._shuttle is None:
            data = self.env.reset()
        else:
            data = self._shuttle

        trajectories = []
        collected_frames = 0
        time_generate = 0
        time_step_start = time.perf_counter()

        while collected_frames < self.frames_per_batch:
            policy_input = data
            env_input, generation_time = self.policy(
                policy_input,
                self.inference_server,
                **self.policy_kwargs,
            )
            env_output, env_next_output = self.env.step_and_maybe_reset(env_input)

            time_generate += generation_time

            # carry over collector data without messing up devices
            collector_data = self._shuttle.get("collector").copy()
            env_next_output.set("collector", collector_data)
            self._shuttle = env_next_output
            self._shuttle.set("collector", collector_data)
            self._update_traj_ids(env_output)
            data = self._shuttle
            trajectories.append(data)
            collected_frames += data.numel()

        data = lazy_stack(trajectories, -1)

        if self.rollout_queue is not None:
            assert self.replay_buffer is None
            postprocessed_results, total_generated_tokens = self._postprocess_for_queue(
                data
            )

            while True:
                try:
                    self.rollout_queue.put_nowait(postprocessed_results)
                    break
                except QueueFull:
                    self.rollout_queue.get()  # Remove the oldest item to make space
                    log.warn("rollout queue full. Discarding data.")

        if self._is_collector_zero:
            # End timing the rollout step
            time_total_rollout = time.perf_counter() - time_step_start

            # TODO: training.get_memory_stats() crashes vLLM
            # Log metrics
            gpu_memory = {
                "allocated": torch.cuda.max_memory_allocated(device="cuda:0"),
                "reserved": torch.cuda.max_memory_reserved(device="cuda:0"),
                "active": torch.cuda.memory_stats(device="cuda:0").get(
                    "active_bytes.all.peak", 0
                ),
            }
            time_total_rollout = time.perf_counter() - time_step_start
            self._log_metrics(
                step_idx=idx,
                time_total_rollout=time_total_rollout,
                time_generate=time_generate,
                total_generated_tokens=total_generated_tokens,
                # full_queue_data_discard=full_queue_data_discard,
                gpu_memory=gpu_memory,
            )

        return data

    def _setup_data(
        self,
        cfg_dataset: DictConfig,
        shuffle: bool,
        batch_size: int,
        collate_fn: str,
        dataloader_state_dict: Optional[Dict[str, Any]] = None,
    ) -> StatefulDataLoader:
        """
        All data related setup happens here. Currently this recipe only supports the
        DistributedSamplers with Map-style Datasets which fit into memory. Other samplers,
        iterable datasets and streaming datasets are not supported.
        """
        # Not importing here and doing these imports globally will cause VLLM worker
        # to have no cuda devices during cuda lazy init for some reason?? Even when
        # this method is not actually called...
        from torchtune import config
        from torchtune.config._utils import _get_component_from_path
        from torchtune.datasets import ConcatDataset

        if isinstance(cfg_dataset, ListConfig):
            datasets = [
                config.instantiate(single_cfg_dataset, self._tokenizer)
                for single_cfg_dataset in cfg_dataset
            ]
            ds = ConcatDataset(datasets=datasets)
        else:
            ds = config.instantiate(cfg_dataset, self._tokenizer)

        collate_fn = _get_component_from_path(collate_fn)
        sampler = StatefulDistributedSampler(
            ds,
            # FIXME: hardcoding num_replicas and rank for now
            num_replicas=1,
            rank=0,
            shuffle=shuffle,
            # FIXME: set seed?
            # seed=self.seed,
        )
        dataloader = StatefulDataLoader(
            dataset=ds,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=(
                partial(
                    collate_fn,
                    padding_idx=self._tokenizer.pad_id,
                )
            ),
            # dropping last avoids shape issues with compile + flex attention
            drop_last=True,
        )
        if dataloader_state_dict is not None:
            raise AssertionError("Haven't handled dataloader_state_dict yet")
            dataloader.load_state_dict(dataloader_state_dict)
            # B/c we currently only save at epoch boundaries, if we cut the previous epoch short
            # we need to force the dataloader to finish the last iteration before it's actually used
            list(dataloader)
        return dataloader


class VLLMWorkerWrapper(Worker):
    """
    vLLM Rollout Model worker for Ray.

    vLLMParameterServer will always take rank 0 in the stateless process group
    initialized by this worker. And the tp ranks associated with the LLM class
    will be in the range [1, tp_size].
    """

    def __init__(self, *args, **kwargs):
        import os

        print(f"visible devices {os.environ['CUDA_VISIBLE_DEVICES']}")
        print(f"device count {torch.cuda.device_count()}")
        super().__init__(*args, **kwargs)

    def init_weight_update_group(
        self, master_address, master_port, rank_offset, world_size
    ):
        from vllm.distributed.parallel_state import get_world_group

        rank = get_world_group().rank + rank_offset

        self._model_update_group = stateless_init_process_group(
            master_address,
            master_port,
            rank,
            world_size,
            self.device,
        )

        self.version = torch.tensor([0], device="cuda")

    def update_weight(self, name, dtype, shape):
        weight = torch.empty(shape, dtype=dtype, device="cuda")
        # src=0 because fsdp worker 0 has been assigned as "0" in this process group
        self._model_update_group.broadcast(
            weight, src=0, stream=torch.cuda.current_stream()
        )
        self.model_runner.model.load_weights(weights=[(name, weight)])
        del weight

    def update_policy_version(self):
        self._model_update_group.broadcast(
            self.version, src=0, stream=torch.cuda.current_stream()
        )
        self.policy_version = self.version
        torch.cuda.synchronize()


@ray.remote(num_cpus=8, num_gpus=1)
class PyTorchActorModel:
    """
    A Ray actor responsible for training a model using the GRPO (Generalized Reward Policy Optimization)
    algorithm in a distributed setting.
    This class leverages PyTorch's distributed training capabilities with FSDP and interacts
    with a replay buffer and VLLM engines.

    Args:
        cfg (DictConfig): Configuration object containing training parameters.
        environment_variables (dict): Environment variables for distributed training setup.
        replay_buffer: Shared replay buffer for sampling trajectories.
    """

    def __init__(self, cfg, environment_variables, replay_buffer):
        import torch

        self.replay_buffer = replay_buffer
        self.cfg = cfg

        # Device and dtype setup
        self._device = utils.get_device(device=cfg.device)
        self._dtype = training.get_dtype(cfg.dtype, device=self._device)

        device_type = self.cfg.device
        self._log_peak_memory_stats = cfg.get("log_peak_memory_stats", True)
        if self._log_peak_memory_stats and device_type != "cuda":
            log.info(
                "log_peak_memory_stats was set to True, but training does not use cuda. Setting to False."
            )
            self._log_peak_memory_stats = False

        if self._dtype == torch.float16:
            raise ValueError(
                "Full fp16 training is not supported. Use bf16 or fp32 instead."
            )

        # Logging attributes
        self._output_dir = cfg.output_dir
        self._log_every_n_steps = cfg.get("log_every_n_steps", 1)

        self.fsdp_cpu_offload = cfg.get("fsdp_cpu_offload", False)
        self.distributed_backend = training.get_distributed_backend(
            device_type, offload_ops_to_cpu=self.fsdp_cpu_offload
        )

        # Distributed training setup: Simulate torchrun environment
        for var in environment_variables:
            os.environ[var] = str(environment_variables[var])

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")

        world_size = torch.distributed.get_world_size()
        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.fsdp_group = torch.distributed.new_group(
            ranks=list(range(self.world_size - 1)), use_local_synchronization=True
        )
        self.device_mesh = torch.distributed.device_mesh.DeviceMesh.from_group(
            self.fsdp_group, device_type="cuda"
        )

        self._is_rank_zero = self.rank == 0

        # Training configuration
        self._resume_from_checkpoint = cfg.resume_from_checkpoint
        self._clip_grad_norm = cfg.get("clip_grad_norm", None)

        # Activation checkpointing and offloading
        self._enable_activation_checkpointing = cfg.get(
            "enable_activation_checkpointing", False
        )
        self._enable_activation_offloading = cfg.get(
            "enable_activation_offloading", False
        )
        if self._enable_activation_offloading:
            if self._device.type != "cuda":
                raise RuntimeError(
                    "enable_activation_offloading should only be True when training on CUDA"
                )
            if not self._enable_activation_checkpointing:
                raise RuntimeError(
                    "enable_activation_offloading should only be True when enable_activation_checkpointing is True"
                )
        elif (
            self._enable_activation_checkpointing
            and cfg.checkpointer.model_type != "LLAMA3_VISION"
        ):
            utils.log_rank_zero(
                log,
                "Hint: enable_activation_checkpointing is True, but enable_activation_offloading isn't. "
                "Enabling activation offloading should reduce memory further.",
            )

        # Recipe state
        self.seed = training.set_seed(seed=cfg.seed)
        self.total_epochs = cfg.epochs
        self.global_step = 0
        self._steps_run = 0
        self._total_dialog_turns = cfg.num_steps
        self._epochs_run = 0
        self._rng = torch.Generator(self._device).manual_seed(
            self.seed
        )  # TODO: Verify if needed for GRPO

        # RL parameters
        self.grpo_samples = cfg.grpo_samples
        self._temperature = cfg.temperature
        self._top_k = cfg.top_k
        self._max_generated_tokens = cfg.max_generated_tokens
        self.batch_size = cfg.batch_size
        self._ppo_epochs = cfg.ppo_epochs
        self._save_every_n_epochs = cfg.save_every_n_epochs

        # Model and optimizer setup
        checkpoint_dict = self.load_checkpoint(cfg_checkpointer=cfg.checkpointer)
        self._compile = cfg.get("compile", False)
        self._model = self._setup_model(
            cfg_model=cfg.model,
            enable_activation_checkpointing=self._enable_activation_checkpointing,
            enable_activation_offloading=self._enable_activation_offloading,
            custom_sharded_layers=cfg.get("custom_sharded_layers", None),
            fsdp_cpu_offload=self.fsdp_cpu_offload,
            model_state_dict=checkpoint_dict[training.MODEL_KEY],
        )
        self._optimizer = self._setup_optimizer(cfg_optimizer=cfg.optimizer)
        self._loss_fn = config.instantiate(cfg.loss)

        if self._compile:
            training.compile_loss(self._loss_fn, verbose=self._is_rank_zero)

        # TODO: generalize this to any chunked loss
        # set num_output_chunks for model
        if self._loss_fn.__class__.__name__ == "GRPOWithChunkedOutputLoss":
            self._model.set_num_output_chunks(self._loss_fn.num_output_chunks)

        self._tokenizer = config.instantiate(self.cfg.tokenizer)

        # FIXME: need to get _steps_per_epoch when dataloader is no longer per fsdp worker but instead wrapped in vLLM
        # self._lr_scheduler = self._setup_lr_scheduler(
        #     cfg_lr_scheduler=cfg.get("lr_scheduler", None),
        #     num_training_steps=self.total_epochs * self._steps_per_epoch,
        #     last_epoch=self.global_step - 1,
        # )
        self._lr_scheduler = None

        # Set up profiler, returns DummyProfiler (nullcontext object with no-op `step` method)
        # if cfg is missing profiler key or if `cfg.profiler.enabled = False`
        self._profiler = self._setup_profiler(cfg.get(PROFILER_KEY, None))
        self._steps_before_sync = cfg.steps_before_sync

        # Initialize policy version for tracking age of trajectories
        self.policy_version = 0
        self.metric_logger = None  # Placeholder for the logger

        log.info("Done setup")

    def set_metric_logger(self, logger):
        """Store the MetricLoggerActor handle for logging metrics."""
        if self._is_rank_zero:
            log.info(f"Setting metric logger {logger} for rank {self.rank}")
            self._metric_logger = logger

    def _setup_profiler(
        self, cfg_profiler: Optional[DictConfig] = None
    ) -> Union[torch.profiler.profile, DummyProfiler]:
        """Set up the profiler based on the configuration. Returns DummyProfiler if not enabled."""
        if cfg_profiler is None:
            cfg_profiler = DictConfig({"enabled": False})

        if cfg_profiler.get("_component_", None) is None:
            cfg_profiler["_component_"] = "torchtune.training.setup_torch_profiler"
        else:
            assert (
                cfg_profiler.get("_component_")
                == "torchtune.training.setup_torch_profiler"
            ), "Only torch profiler supported currently: component must be `torchtune.training.setup_torch_profiler`"

        profiler, profiler_cfg = config.instantiate(cfg_profiler)
        utils.log_rank_zero(log, f"Profiler config after instantiation: {profiler_cfg}")
        if self._is_rank_zero:
            self.profiler_profile_memory = profiler_cfg.get("profile_memory", False)
            if profiler_cfg["enabled"]:
                self.profiler_wait_steps = profiler_cfg["wait_steps"]
                self.profiler_warmup_steps = profiler_cfg["warmup_steps"]
                self.profiler_active_steps = profiler_cfg["active_steps"]
        return profiler

    # FIXME: do we need this?
    def forward(self, *args, **kwargs):
        """Forward pass through the model."""
        return self._model(*args, **kwargs)

    def init_model_update_group(self, master_address, master_port, rank, world_size):
        """Initialize the model update group for weight synchronization."""
        self._model_update_group = stateless_init_process_group(
            master_address,
            master_port,
            rank,
            world_size,
            torch.device("cuda:0"),  # FIXME: Hardcoded device
        )

    def load_checkpoint(self, cfg_checkpointer: DictConfig) -> Dict[str, Any]:
        """
        Extract the checkpoint state from file and validate. If resume_from_checkpoint
        is True, this also includes the recipe state.
        """
        self._checkpointer = config.instantiate(
            cfg_checkpointer,
            resume_from_checkpoint=self._resume_from_checkpoint,
        )
        checkpoint_dict = self._checkpointer.load_checkpoint()
        if self._resume_from_checkpoint:
            self._update_recipe_state(checkpoint_dict)
        return checkpoint_dict

    def _update_recipe_state(self, ckpt_dict: Dict[str, Any]) -> None:
        """Update recipe state from checkpoint."""
        try:
            self._epochs_run = ckpt_dict[training.EPOCHS_KEY]
            self._rng.set_state(ckpt_dict[training.RNG_KEY])

            # on mismatch, warn the user and prevent the override
            if self.seed != ckpt_dict[training.SEED_KEY]:
                warn(
                    message=(
                        "Config value for seed does not match the checkpoint value, "
                        f"using the checkpoint value: {ckpt_dict[training.SEED_KEY]}"
                    )
                )
                self.seed = ckpt_dict[training.SEED_KEY]

            # on mismatch, warn the user but allow the override
            if self.total_epochs != ckpt_dict[training.TOTAL_EPOCHS_KEY]:
                warn(
                    message=(
                        "Config value for total_epochs does not match the checkpoint value, "
                        f"using the config value: {self.total_epochs}"
                    )
                )

        except KeyError as e:
            raise KeyError(
                "Checkpoint does not contain the required keys needed for updating recipe state. "
                "Are you sure you passed in the right recipe checkpoint?"
            ) from e

    def _setup_model(
        self,
        cfg_model: DictConfig,
        enable_activation_checkpointing: bool,
        enable_activation_offloading: bool,
        fsdp_cpu_offload: bool,
        model_state_dict: Dict[str, Any],
        custom_sharded_layers: Optional[List[str]] = None,
    ) -> nn.Module:
        """
        Model initialization has some important considerations:
           a. To minimize GPU peak memory, we initialize the model on meta device with
              the right dtype
           b. All ranks calls ``load_state_dict`` without peaking CPU RAMs since
              full state dicts are loaded with ``torch.load(mmap=True)``
        """
        from torchtune.training import disable_dropout

        utils.log_rank_zero(
            log,
            "FSDP is enabled. Instantiating model and loading checkpoint on Rank 0 ...",
        )
        time_setup_start = time.perf_counter()

        with training.set_default_dtype(self._dtype), torch.device("meta"):
            model = config.instantiate(cfg_model)

        if self._compile:
            training.compile_model(model, verbose=self._is_rank_zero)

        if enable_activation_checkpointing:
            training.set_activation_checkpointing(
                model, auto_wrap_policy={modules.TransformerSelfAttentionLayer}
            )

        fsdp_shard_conditions = [
            partial(training.get_shard_conditions, names_to_match=custom_sharded_layers)
        ]
        training.shard_model(
            model=model,
            shard_conditions=fsdp_shard_conditions,
            cpu_offload=fsdp_cpu_offload,
            reshard_after_forward=True,
            dp_mesh=self.device_mesh,
        )

        with training.set_default_dtype(self._dtype), self._device:
            for m in model.modules():
                # RoPE is not covered in state dict
                if hasattr(m, "rope_init"):
                    m.rope_init()

        # This method will convert the full model state dict into a sharded state
        # dict and load into the model
        training.load_from_full_model_state_dict(
            model,
            model_state_dict,
            self._device,
            strict=True,
            cpu_offload=fsdp_cpu_offload,
        )

        utils.log_rank_zero(
            log,
            f"Instantiating model and loading checkpoint took {time.perf_counter() - time_setup_start:.2f} secs",
        )

        self.activations_handling_ctx = training.get_act_offloading_ctx_manager(
            model, enable_activation_offloading
        )
        training.validate_no_params_on_meta_device(model)

        if self._is_rank_zero and self._log_peak_memory_stats:
            memory_stats = training.get_memory_stats(device=self._device)
            training.log_memory_stats(memory_stats)

        disable_dropout(model)

        # synchronize before training begins
        torch.distributed.barrier(group=self.fsdp_group)
        return model

    def _setup_optimizer(
        self, cfg_optimizer: DictConfig, opt_state_dict=None
    ) -> Optional[Optimizer]:
        """Initialize the optimizer."""
        optimizer = config.instantiate(cfg_optimizer, self._model.parameters())

        # TODO: does this work?
        if opt_state_dict:
            training.load_from_full_optimizer_state_dict(
                self._model,
                optimizer,
                opt_state_dict,
                self._device,
            )

        utils.log_rank_zero(log, "Optimizer is initialized.")
        return optimizer

    def grpo_step(
        self,
        trajectory: GRPOTrajectory,
        context_length: int,
    ) -> GRPOStats:
        """Perform a single GRPO optimization step over a batch of trajectories and corresponding advantages and returns.

        Args:
            trajectory (Trajectory): a batch of trajectories
            context_length (int): the length of the context window
            targets_mask (torch.Tensor): a boolean mask indicating which tokens in the trajectory are targets

        Returns:
            GRPOStats: An instance of :class:`~torchtune.rlhf.PPOStats`
        """

        # Create an output mask to avoid computing model.output on tokens we won't train
        # FIXME: when bsz>1, don't we have multiple context_length?
        # FIXME: because of chunked CE, the outout of pi_logits is a chunked list, so masking after the fact is
        # more annoying. Masking before the chunking is easier, but we have to figure out masking for bsz>1
        output_mask = torch.zeros_like(
            trajectory.query_responses, dtype=torch.bool, device=self._device
        )
        output_mask[:, context_length - 1 : -1] = True

        # call model
        with self.activations_handling_ctx:
            pi_logits = self._model(
                trajectory.query_responses,
                input_pos=trajectory.position_ids,
                mask=trajectory.masks,
                output_mask=output_mask,
            )

        # apply to targets
        targets = trajectory.query_responses[:, context_length:]

        # Compute GRPO loss
        loss, policy_loss, kl_loss, ratios, clipfrac, pi_logprobs = self._loss_fn(
            pi_logits=pi_logits,
            targets=targets,
            ref_logprobs=trajectory.ref_logprobs,
            advantages=trajectory.advantages,
            padding_masks=~trajectory.response_padding_masks,
        )

        with torch.no_grad():
            mask = ~trajectory.response_padding_masks  # True for non-padded tokens
            approx_policy_kls = (
                0.5 * ((pi_logprobs - trajectory.logprobs)[mask].pow(2)).mean()
            )

        del pi_logprobs, pi_logits
        torch.cuda.empty_cache()  # TODO: Test if this is needed
        loss.backward()

        return GRPOStats(
            loss=loss,
            policy_loss=policy_loss,
            kl_loss=kl_loss,
            ratios=ratios,
            clipfrac=clipfrac,
            approx_policy_kls=approx_policy_kls,
        )

    def set_vllm_engines(self, engines):
        """Set the vLLM engines for weight synchronization."""
        self._vllm_engines = engines

    def cleanup_after_step(
        self, trajectory: GRPOTrajectory, l_grpo_stats: List[GRPOStats]
    ) -> None:
        """Clean up memory after a training step."""
        for v in trajectory:
            del v
        del trajectory
        for g in l_grpo_stats:
            for v in g:
                del v
            del g
        del l_grpo_stats

    def _log_metrics(
        self,
        step_idx,
        trajectory,
        grpo_stats,
        total_step_time,
        time_grpo_steps,
        time_waiting_buffer,
        time_weight_sync,
        time_weight_gather,
        number_of_tokens,
        padded_tokens_percentage,
        policy_age,
        train_replay_buffer_size,
    ):
        """Log training metrics, only on rank zero."""
        if not self._is_rank_zero:
            return

        grpo_stats_stacked = GRPOStats(*map(torch.stack, zip(*grpo_stats)))
        log_dict = {}
        if self._log_peak_memory_stats:
            memory_stats = training.get_memory_stats(device=self._device)
            log_dict.update(
                {
                    f"train_actor_performance/memory/{k}": v
                    for k, v in memory_stats.items()
                }
            )

        log_dict.update(
            {
                "train_actor_training/loss": grpo_stats_stacked.loss.mean().item(),
                "train_actor_training/policy_loss": grpo_stats_stacked.policy_loss.mean().item(),
                "train_actor_training/num_stop_tokens": trajectory.response_padding_masks.any(
                    -1
                )
                .sum()
                .item(),
                "train_actor_training/kl_loss": grpo_stats_stacked.kl_loss.mean().item(),
                "train_actor_training/ratios": grpo_stats_stacked.ratios.mean().item(),
                "train_actor_training/clipfrac": grpo_stats_stacked.clipfrac.mean().item(),
                "train_actor_training/approx_policy_kls": grpo_stats_stacked.approx_policy_kls.mean().item(),
                "train_actor_training/response_lengths": trajectory.seq_lens.float()
                .mean()
                .item(),
            }
        )

        log_dict.update(
            {
                "train_actor_performance/total_step_time (s)": total_step_time,
                "train_actor_performance/time_grpo_steps (s)": time_grpo_steps,
                "train_actor_performance/pct_time_grpo_steps (%)": (
                    time_grpo_steps / total_step_time * 100
                    if total_step_time > 0
                    else 0
                ),
                "train_actor_performance/tokens_per_second": (
                    number_of_tokens / total_step_time if total_step_time > 0 else 0
                ),
                "train_actor_performance/time_weight_sync (s)": time_weight_sync,
                "train_actor_performance/pct_time_weight_sync (%)": (
                    time_weight_sync / total_step_time * 100
                    if total_step_time > 0
                    else 0
                ),
                "train_actor_performance/padded_tokens_percentage (%)": padded_tokens_percentage,
                "train_actor_performance/time_waiting_buffer (s)": time_waiting_buffer,
                "train_actor_performance/pct_time_waiting_buffer (%)": (
                    time_waiting_buffer / total_step_time * 100
                    if total_step_time > 0
                    else 0
                ),
                "train_actor_performance/time_weight_gather (s)": time_weight_gather,
                "train_actor_performance/pct_time_weight_gather (%)": (
                    time_weight_gather / total_step_time * 100
                    if total_step_time > 0
                    else 0
                ),
                "queues/train_actor_policy_age_mean": policy_age,
                "queues/train_replay_buffer_size": train_replay_buffer_size,
            }
        )

        ray.get(self._metric_logger.log_dict.remote(log_dict, step=step_idx))

    def train(self):
        """Execute the GRPO training loop."""
        training.cleanup_before_training()
        self._optimizer.zero_grad()
        self._profiler.start()

        while self._steps_run < self._total_dialog_turns:

            # Memory profiling start
            if (
                self._is_rank_zero
                and self.profiler_profile_memory
                and self._steps_run
                == self.profiler_wait_steps + self.profiler_warmup_steps
            ):
                torch.cuda.memory._record_memory_history()

            time_step_start = time.perf_counter()

            # Fetch trajectory from queue
            time_waiting_buffer_start = time.perf_counter()
            train_replay_buffer_size = None
            if self._is_rank_zero:
                train_replay_buffer_size = len(self.replay_buffer)

            while not len(self.replay_buffer):
                log.info("waiting for replay buffer")
                time.sleep(1.0)

            trajectory = self.replay_buffer.sample().to(self._device)
            time_waiting_buffer = time.perf_counter() - time_waiting_buffer_start
            if self._is_rank_zero:
                log.info(f"{self.rank=} got from queue traj {trajectory}")

            # Prepare trajectory for optimization
            trajectory, context_length, metadata = self._prepare_trajectory(trajectory)

            # Perform GRPO optimization
            time_grpo_steps_start = time.perf_counter()
            grpo_stats: list[GRPOStats] = []
            for _ in range(self._ppo_epochs):

                # step
                step_stats = self.grpo_step(
                    trajectory,
                    context_length,
                )
                grpo_stats.append(step_stats)

                # grad norm
                if self._clip_grad_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self._model.parameters(), max_norm=float(self._clip_grad_norm)
                    )

                # optimizer step
                torch.distributed.barrier(group=self.fsdp_group)
                self._optimizer.step()
                torch.distributed.barrier(group=self.fsdp_group)
                self._optimizer.zero_grad(set_to_none=True)

                # scheduler
                self.global_step += 1
                if self._lr_scheduler is not None:
                    self._lr_scheduler.step()

            log.info(f"{self.rank=} finished step {self._steps_run}")
            time_grpo_steps = time.perf_counter() - time_grpo_steps_start
            self._steps_run += 1

            # Synchronize weights
            time_weight_sync = time_weight_gather = 0
            if self._steps_run % self._steps_before_sync == 0:
                utils.log_rank_zero(log, "started weight gather")
                torch.distributed.barrier(group=self.fsdp_group)
                time_weight_gather_start = time.perf_counter()
                new_sd = {
                    k: v.full_tensor() for k, v in self._model.state_dict().items()
                }
                torch.cuda.synchronize()
                time_weight_gather = time.perf_counter() - time_weight_gather_start
                utils.log_rank_zero(log, f"Done gather in {time_weight_gather}")
                time_sync_start = time.perf_counter()
                utils.log_rank_zero(log, "started weight sync")
                self.sync_weights(new_sd)
                del new_sd
                time_weight_sync = time.perf_counter() - time_sync_start
                utils.log_rank_zero(log, f"Done sync in {time_weight_sync}")

            # Log metrics
            total_step_time = time.perf_counter() - time_step_start
            avg_policy_age = self.policy_version - (
                sum(metadata["policy_version"]) / len(metadata["policy_version"])
            )
            if self._is_rank_zero:
                log.info("logging metrics")
                self._log_metrics(
                    step_idx=self.global_step,
                    trajectory=trajectory,
                    grpo_stats=grpo_stats,
                    total_step_time=total_step_time,
                    time_grpo_steps=time_grpo_steps,
                    time_waiting_buffer=time_waiting_buffer,
                    time_weight_sync=time_weight_sync,
                    time_weight_gather=time_weight_gather,
                    number_of_tokens=metadata["number_of_tokens"],
                    padded_tokens_percentage=metadata["padded_tokens_percentage"],
                    policy_age=avg_policy_age,
                    train_replay_buffer_size=train_replay_buffer_size,
                )
                log.info("done logging metrics")

            self.cleanup_after_step(trajectory, grpo_stats)

            # Memory profiling stop
            self._profiler.step()
            if (
                self._is_rank_zero
                and self.profiler_profile_memory
                and self._steps_run
                == self.profiler_wait_steps
                + self.profiler_warmup_steps
                + self.profiler_active_steps
            ):
                torch.cuda.memory._record_memory_history(enabled=None)

            torch.distributed.barrier(group=self.fsdp_group)

        self._profiler.stop()

    def register_parameter_server(self, parameter_server):
        assert self._is_rank_zero
        self.parameter_server = parameter_server

    def get_model_metadata(self):
        from torch._subclasses.fake_tensor import FakeTensorMode

        fake_sd_metadata = {
            k: (v.shape, v.dtype) for k, v in self._model.state_dict().items()
        }
        fake_sd = dict()
        with FakeTensorMode():
            for k, (shape, dtype) in fake_sd_metadata.items():
                fake_sd[k] = torch.empty(shape, dtype=dtype, device="cuda")

        hf_fake_sd = qwen2_tune_to_hf(fake_sd, num_heads=16, num_kv_heads=2, dim=2048)

        return {k: (v.dtype, v.shape) for k, v in hf_fake_sd.items()}

    def sync_weights(self, new_sd):
        self.policy_version += 1
        if self._is_rank_zero:
            new_sd = qwen2_tune_to_hf(new_sd, num_heads=16, num_kv_heads=2, dim=2048)
            ray.get(self.parameter_server.acquire_state_dict_lock.remote())
            self.parameter_server.receive_from_trainer.remote()
            for i, (k, v) in enumerate(new_sd.items()):
                # dst is global rank, can switch to group_dst arg if not 2.5.1
                torch.distributed.send(v, dst=self.world_size - 1)

            ray.get(self.parameter_server.release_state_dict_lock.remote())

    def _prepare_trajectory(self, raw_trajectory):
        """Process raw trajectory, compute rewards, and prepare for optimization.

        Args:
            raw_trajectory: The trajectory sampled from the replay buffer.

        Returns:
            trajectory (GRPOTrajectory): Processed trajectory for GRPO optimization.
            context_length (int): Length of the context sequence.
            metadata (dict): Metadata for logging, including rewards and performance metrics.
        """
        # Extract components from raw trajectory
        query_responses = raw_trajectory.query_responses
        responses = raw_trajectory.responses
        logprobs = raw_trajectory.logprobs
        ref_logprobs = raw_trajectory.ref_logprobs
        query_response_padding_masks = raw_trajectory.query_response_padding_masks
        seq_lens = raw_trajectory.seq_lens
        answers = raw_trajectory.answers
        policy_version = raw_trajectory.policy_version
        rewards = raw_trajectory.rewards
        advantages = raw_trajectory.advantages
        successes = raw_trajectory.successes
        reward_metadata = raw_trajectory.reward_metadata

        # Compute padded tokens percentage
        total_tokens = query_responses.numel()
        padded_tokens = (query_responses == self._tokenizer.pad_id).sum().item()
        padded_tokens_percentage = (
            (padded_tokens / total_tokens) * 100 if total_tokens > 0 else 0
        )
        number_of_tokens = seq_lens.sum().item()

        # Truncate sequences at first stop token
        response_padding_masks, responses = rlhf.truncate_sequence_at_first_stop_token(
            responses,
            torch.tensor(self._tokenizer.stop_tokens, device=self._device),
            self._tokenizer.pad_id,
        )

        # Generate masks and position IDs
        masks = generation.get_causal_mask_from_padding_mask(
            query_response_padding_masks
        )
        position_ids = generation.get_position_ids_from_padding_mask(
            query_response_padding_masks
        )
        context_length = query_responses.shape[1] - responses.shape[1]
        del query_response_padding_masks

        # Create GRPOTrajectory
        trajectory = GRPOTrajectory(
            query_responses=query_responses,
            logprobs=logprobs,
            ref_logprobs=ref_logprobs,
            advantages=advantages,
            masks=masks,
            position_ids=position_ids,
            response_padding_masks=response_padding_masks,
            seq_lens=training.get_unmasked_sequence_lengths(response_padding_masks),
        )

        # Metadata for logging
        metadata = {
            "padded_tokens_percentage": padded_tokens_percentage,
            "number_of_tokens": number_of_tokens,
            "policy_version": policy_version,
            "reward_metadata": reward_metadata,
        }

        return trajectory, context_length, metadata

    def cleanup(self) -> None:
        """Close the metric logger on rank zero."""
        if self._is_rank_zero:
            self._metric_logger.close.remote()


@ray.remote(num_cpus=1, num_gpus=0)
class MetricLoggerActor:
    def __init__(self, cfg):
        self.logger = config.instantiate(cfg.metric_logger)
        self.logger.log_config(cfg)

    def log_dict(self, log_dict, step=None):
        # allowing actors to use their own step counters
        self.logger.log_dict(log_dict, step=step)

    def close(self):
        if hasattr(self.logger, "close"):
            self.logger.close()


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


@ray.remote(num_cpus=4, num_gpus=1)
class VLLMParameterServer(RemoteWeightUpdaterBase):
    def __init__(self, cfg, vllm_master_addresses, vllm_master_ports, env_vars):
        log.info("in param server init")
        super().__init__()
        self.cfg = cfg
        self.vllm_master_addresses = vllm_master_addresses
        self.vllm_master_ports = vllm_master_ports
        self.vllm_comm_groups = dict()
        self.vllm_weight_versions = dict()
        self.vllm_worker_handles = dict()

        import os

        import torch
        import torch.distributed

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

        # FIXME: why this hang even when I pass use_local_synchronization=False in the other one??
        # self.fsdp_group = torch.distributed.new_group(ranks=list(range(self.world_size - 1)))

    def register_collector(self, worker_id, handle):
        self.vllm_worker_handles[worker_id] = handle
        log.info(f"registered collector {worker_id=}")

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

    def all_worker_ids(self):
        return [i for i in range(len(self.collector._remote_collectors))]

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
                f"skipping update for {worker_id=}, {self.version=}, {self.vllm_weight_versions[worker_id]=}"
            )
            return True
        return False

    def _init_model_update_group(self, worker_id):
        worker_handle = self.vllm_worker_handles[worker_id]
        vllm_tp_size = self.cfg.vllm.tp_size
        weight_sync_world_size = vllm_tp_size + 1
        model_update_group = stateless_init_process_group(
            self.vllm_master_addresses[worker_id],
            self.vllm_master_ports[worker_id],
            0,
            weight_sync_world_size,
            torch.device("cuda:0"),
        )
        self.vllm_comm_groups[worker_id] = model_update_group

    def _sync_weights_with_worker(self, worker_id: int, server_weights):
        print(f"in _sync_weights_with_worker {worker_id}")
        worker_handle = self.vllm_worker_handles[worker_id]
        worker_handle.update_policy_weights_.remote()
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
        print(f"_sync_weights_with_worker done broadcast {worker_id} {self.version=}")
        self.vllm_weight_versions[worker_id] = self.version
        read_lock.release()

    def receive_from_trainer(self):
        for k, v in self.state_dict.items():
            torch.distributed.recv(v, src=0)


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
                )(LLMCollector)
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
