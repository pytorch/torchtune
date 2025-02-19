# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import json
import os
import sys
import time
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union
from warnings import warn

import torch
from omegaconf import DictConfig, ListConfig
from torch import nn
from torch.distributed import destroy_process_group, init_process_group
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler

from torchtune import config, generation, modules, rlhf, training, utils
from torchtune.config._utils import _get_component_from_path
from torchtune.datasets import ConcatDataset
from torchtune.modules import local_kv_cache
from torchtune.recipe_interfaces import FTRecipeInterface
from torchtune.rlhf import batch_shaped_correctness_reward
from torchtune.rlhf._types import GRPOStats, GRPOTrajectory
from torchtune.training import DummyProfiler, PROFILER_KEY
from torchtune.training.activations import apply_selective_activation_checkpointing
from torchtune.training.lr_schedulers import get_lr
from tqdm import tqdm

log = utils.get_logger("DEBUG")


class FullGRPOEval(FTRecipeInterface):

    def __init__(self, cfg: DictConfig) -> None:
        self._device = utils.get_device(device=cfg.device)
        self._dtype = training.get_dtype(cfg.dtype, device=self._device)
        device_type = cfg.device

        if self._dtype == torch.float16:
            raise ValueError(
                "full fp16 training is not supported with this recipe. Please use bf16 or fp32 instead."
            )

        self.cfg = cfg
        # logging attributes
        self._log_peak_memory_stats = cfg.get("log_peak_memory_stats", False)

        if self._log_peak_memory_stats and self._device.type != "cuda":
            log.info(
                "log_peak_memory_stats was set to True, however, training does not use cuda. Setting log_peak_memory_stats=False."
            )
            self._log_peak_memory_stats = False

        self.fsdp_cpu_offload = cfg.get("fsdp_cpu_offload", False)
        self._enable_async_checkpointing = cfg.get("enable_async_checkpointing", False)

        self.distributed_backend = training.get_distributed_backend(
            device_type,
            offload_ops_to_cpu=self.fsdp_cpu_offload
            or self._enable_async_checkpointing,
        )
        init_process_group(self.distributed_backend)

        world_size, rank = utils.get_world_size_and_rank()
        self.rank = rank
        self.world_size = world_size
        self._is_rank_zero = rank == 0

        # These are public properties which are updated by the checkpoint loader
        # when ``resume_from_checkpoint`` is `True` or validated in tests
        self.seed = training.set_seed(seed=cfg.seed)

        self._rng = torch.Generator(self._device).manual_seed(self.seed)

    def load_checkpoint(self, cfg_checkpointer: DictConfig) -> Dict[str, Any]:
        """
        Extract the checkpoint state from file and validate. If resume_from_checkpoint
        is True, this also includes the recipe state.
        """
        self._checkpointer = config.instantiate(
            cfg_checkpointer,
            resume_from_checkpoint=False,
        )
        checkpoint_dict = self._checkpointer.load_checkpoint()

        return checkpoint_dict

    def setup(self, cfg: DictConfig) -> None:
        """
        Setup the recipe. This includes training state (if resume_from_checkpoint is True),
        model, tokenizer, loss, optimizer, lr scheduler, sampler, and dataloader.
        """
        if self.fsdp_cpu_offload:
            training.set_torch_num_threads()

        checkpoint_dict = self.load_checkpoint(cfg_checkpointer=cfg.checkpointer)

        self._compile = cfg.get("compile", False)
        self._model = self._setup_model(
            cfg_model=cfg.model,
            enable_activation_checkpointing=False,
            custom_sharded_layers=cfg.get("custom_sharded_layers", None),
            fsdp_cpu_offload=self.fsdp_cpu_offload,
            model_state_dict=checkpoint_dict[training.MODEL_KEY],
        )
        self._tokenizer = config.instantiate(cfg.tokenizer)


        # sampler and dataloader depend on the tokenizer and loss_fn and should be
        # setup after both of these are initialized
        collate_name = cfg.get("collate_fn", "torchtune.data.padded_collate_rl")
        self._sampler, self._dataloader = self._setup_data(
            cfg_dataset=cfg.dataset,
            batch_size=cfg.batch_size,
            collate_fn=collate_name,
        )

        # Set up profiler, returns DummyProfiler (nullcontext object with no-op `step` method)
        # if cfg is missing profiler key or if `cfg.profiler.enabled = False`
        self._profiler = self._setup_profiler(cfg.get(PROFILER_KEY, None))


        if cfg.get("stop_token_ids", False):
            stop_token_ids = cfg.stop_token_ids
            if self._tokenizer.eos_id not in stop_token_ids:
                warn(
                    f"tokenizer eos_id ({self._tokenizer.eos_id}) is not in stop_token_ids ({stop_token_ids})."
                    "This may lead to unexpected behaviour."
                )
        else:
            if not hasattr(self._tokenizer, "stop_tokens"):
                warn(
                    "No stop tokens defined in tokenizer, and no stop_token_ids provided. This may lead to unexpected behaviour."
                )
                stop_token_ids = []
            else:
                stop_token_ids = self._tokenizer.stop_tokens
        self._stop_token_ids = torch.tensor(stop_token_ids, device=self._device)


        # Saving setup
        checkpoint_path = self.cfg.checkpointer.checkpoint_dir
        eval_name = self.cfg.eval_name

        self.eval_path = os.path.join(checkpoint_path, eval_name)
        self.answer_dump_path = os.path.join(self.eval_path, f"{eval_name}_{self.rank}.txt")

        if self._is_rank_zero:
            os.makedirs(self.eval_path)
            self.results_path = os.path.join(self.eval_path, "results.json")
        else:
            self.results_path = None

    def _setup_profiler(
        self, cfg_profiler: Optional[DictConfig] = None
    ) -> Union[torch.profiler.profile, DummyProfiler]:
        """
        Parses the `profiler` section of top-level `cfg` and sets up profiler

        Args:
            cfg_profiler (Optional[DictConfig]): ``profiler`` section of the top-level ``cfg`` (the main config passed to
                `recipe.main`). Default None.

        Returns:
            profiler: Union[torch.profiler.profile, DummyProfiler] - DummyProfiler is a nullcontext with no-op methods
            for `start`, `stop`, and `step` that can be used in place of `torch.profiler.profile` if profiler is not enabled such
            that the instrumented training loop does not need to be changed profiling is disabled.

        The profiler config can be provided in configs under the `profiler` key with the following layout:

        .. code-block:: yaml
            profiler:
                enabled: bool

                #Output directory of trace artifacts
                output_dir: str

            #`torch.profiler.ProfilerActivity` types to trace
            cpu: bool
            cuda: bool

                #Trace options
                profile_memory: bool
                with_stack: bool
                record_shapes: bool
                with_flops: bool

            # `torch.profiler.schedule` options:
            # wait_steps -> wait, warmup_steps -> warmup, active_steps -> active, num_cycles -> repeat
            wait_steps: int
            warmup_steps: int
            active_steps: int
            num_cycles: int
        """
        # Missing profiler section in config, assume disabled
        if cfg_profiler is None:
            cfg_profiler = DictConfig({"enabled": False})

        # Check that component is included and set correctly
        if cfg_profiler.get("_component_", None) is None:
            cfg_profiler["_component_"] = "torchtune.training.setup_torch_profiler"
        else:
            assert (
                cfg_profiler.get("_component_")
                == "torchtune.training.setup_torch_profiler"
            ), "Only torch profiler supported currently: component must be `torchtune.training.setup_torch_profiler`"

        profiler, profiler_cfg = config.instantiate(cfg_profiler)

        utils.log_rank_zero(
            log, f" Profiler config after instantiation: {profiler_cfg}"
        )
        if self._is_rank_zero:
            self.profiler_profile_memory = profiler_cfg.get("profile_memory", False)
            if profiler_cfg["enabled"]:
                self.profiler_wait_steps = profiler_cfg["wait_steps"]
                self.profiler_warmup_steps = profiler_cfg["warmup_steps"]
                self.profiler_active_steps = profiler_cfg["active_steps"]

        return profiler

    def _setup_model(
        self,
        cfg_model: DictConfig,
        enable_activation_checkpointing: bool,
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

        utils.log_rank_zero(
            log,
            "FSDP is enabled. Instantiating model and loading checkpoint on Rank 0 ...",
        )
        init_start = time.perf_counter()

        with training.set_default_dtype(self._dtype), torch.device("meta"):
            model = config.instantiate(cfg_model)

        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        if self._compile:
            training.compile_model(model, verbose=self._is_rank_zero)

        # original activation checkpointing (full) - flip the condition above
        if enable_activation_checkpointing:
            training.set_activation_checkpointing(
                model, auto_wrap_policy={modules.TransformerSelfAttentionLayer}
            )


        # For FSDP sharding
        # fsdp_shard_conditions = [
        #     partial(
        #         training.get_shard_conditions,
        #         names_to_match=custom_sharded_layers,
        #     )
        # ]
        # training.shard_model(
        #     model=model,
        #     shard_conditions=fsdp_shard_conditions,
        #     cpu_offload=fsdp_cpu_offload,
        #     reshard_after_forward=reshard_after_forward,
        # )

        with training.set_default_dtype(self._dtype), self._device:
            for m in model.modules():
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

        # Ensure no params and buffers are on meta device
        training.validate_no_params_on_meta_device(model)

        utils.log_rank_zero(
            log,
            f"Instantiating model and loading checkpoint took {time.perf_counter() - init_start:.2f} secs",
        )
        if self._is_rank_zero:
            memory_stats = training.get_memory_stats(device=self._device)
            training.log_memory_stats(memory_stats)

        # synchronize before training begins
        torch.distributed.barrier()

        return model

    def _setup_data(
        self,
        cfg_dataset: DictConfig,
        batch_size: int,
        collate_fn: str,
    ) -> Tuple[DistributedSampler, DataLoader]:
        """
        All data related setup happens here. Currently this recipe only supports the
        DistributedSamplers with Map-style Datasets which fit into memory. Other samplers,
        iterable datasets and streaming datasets are not supported.
        """

        if isinstance(cfg_dataset, ListConfig):
            datasets = [
                config.instantiate(single_cfg_dataset, self._tokenizer)
                for single_cfg_dataset in cfg_dataset
            ]
            ds = ConcatDataset(datasets=datasets)
        else:
            ds = config.instantiate(cfg_dataset, self._tokenizer)

        # Instantiate collate_fn
        collate_fn = _get_component_from_path(collate_fn)

        sampler = DistributedSampler(
            ds,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False,
            seed=self.seed,
        )
        dataloader = DataLoader(
            dataset=ds,
            batch_size=batch_size,
            sampler=sampler,
            # dropping last avoids shape issues with compile + flex attention
            drop_last=True,
            collate_fn=(
                partial(
                    collate_fn,
                    padding_idx=self._tokenizer.pad_id,
                )
            ),
        )

        utils.log_rank_zero(log, "Dataset and Sampler are initialized.")

        return sampler, dataloader

    def generate_trajectory(
        self, input_ids: torch.Tensor, answers: List[str]
    ) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        """
        Generates a trajectory given the current policy model, the reference policy model, the reward function,
        and batch of inputs. This is done over the following steps:

        1: Generate responses, and logits corresponding to the responses using the current policy,
            generating (query, response) pairs.
        2. Estimate logprobs of the generated responses using the current policy.
        3. Compute rewards and successes for the generated responses.
        4. Estimate advantages using GRPO.
        5. Replace any tokens in the response after the first stop token (usually EOS token) with padding,
            producing truncated responses.

        Args:
            input_ids (torch.Tensor): tensor of input token IDs with shape [b, seq_length]
            answers (List[str]): list of answers corresponding to the input_ids

        Returns:
            Trajectory: An instance of :class:`~torchtune.rlhf.GRPOTrajectory` comprising
                the current trajectory.
        """
        batch_size, context_length = input_ids.shape
        grpo_size = self.cfg.num_samples

        batch_input_ids = input_ids[:, None, :].expand(-1, grpo_size, -1)  # [B, G, L]
        batch_input_ids = batch_input_ids.reshape(batch_size * grpo_size, -1)

        # step 1: generate responses, and logits corresponding to the responses using the current policy

        with local_kv_cache(
            model=self._model,
            batch_size=batch_size * grpo_size,
            device=self._device,
            dtype=self._dtype,
            decoder_max_seq_len=context_length + self.cfg.max_generated_tokens,
        ):
            query_responses, logits = generation.generate(  # [B x G, L], [B x G, L, V]
                model=self._model,
                prompt=batch_input_ids,
                max_generated_tokens=self.cfg.max_generated_tokens,
                temperature=self.cfg.temperature,
                top_k=self.cfg.top_k,
                pad_id=self._tokenizer.pad_id,
                rng=self._rng,
                return_logits=False,
                stop_tokens=self._tokenizer.stop_tokens,
            )

        torch.distributed.barrier()
        torch.cuda.empty_cache()


        responses = query_responses[:, context_length:].clone()

        # step 4. replace any tokens in the responses after the first stop token (usually EOS token) with padding
        # resulting in truncated responses
        (
            response_padding_masks,
            responses,
        ) = rlhf.truncate_sequence_at_first_stop_token(  # [B x G, L]
            responses, self._stop_token_ids, self._tokenizer.pad_id
        )

        # responses :: [B x G, L]
        responses = responses.reshape(batch_size, grpo_size, -1)  # [B, G, L]

        rewards, successes = batch_shaped_correctness_reward(
            self._tokenizer, responses, answers
        )  # [B, G]
        rewards = rewards.to(self._device)
        successes = successes.to(self._device)

        all_text_responses = []

        for b in range(batch_size):
            for g in range(grpo_size):
                tokens = responses[b, g]
                text_response = self._tokenizer.decode(tokens.tolist())
                all_text_responses.append(text_response)

        return rewards, successes, all_text_responses

    def train(self) -> None:
        """
        The core training loop.
        """
        # clean up before training begins
        training.cleanup_before_training()

        self._profiler.start()

        all_rewards: list[torch.Tensor] = []
        all_successes: list[torch.Tensor] = []
        all_completions: list[str] = []

        pbar = tqdm(total=len(self._dataloader), disable=not self._is_rank_zero)
        for idx, batch in enumerate(self._dataloader):

            # Start tracking CUDA memory for active steps for just the first epoch
            if (
                self._is_rank_zero
                and self.profiler_profile_memory
                and idx == self.profiler_wait_steps + self.profiler_warmup_steps
            ):
                torch.cuda.memory._record_memory_history()

            tokens = batch["tokens"]  # type: ignore
            answers = batch["answers"]  # type: ignore
            tokens = tokens.to(self._device)  # [B, P]

            _, context_length = tokens.shape

            rewards, successes, completions = self.generate_trajectory(tokens, answers)  # [B, G]

            all_rewards.append(rewards.mean(1))
            all_successes.append(successes.mean(1))
            all_completions += completions

            pbar.update(1)

        torch.distributed.barrier()
        all_rewards = torch.cat(all_rewards).to(torch.float64)
        all_successes = torch.cat(all_successes).to(torch.float64)
        num_samples = all_rewards.shape[0]
        num_samples = torch.tensor(num_samples, device=self._device, dtype=torch.int64)

        reward_sum = all_rewards.sum()
        success_sum = all_successes.sum()

        torch.distributed.reduce(reward_sum, dst=0)
        torch.distributed.reduce(success_sum, dst=0)
        torch.distributed.reduce(num_samples, dst=0)

        if self._is_rank_zero:
            avg_reward = reward_sum / num_samples
            success_rate = success_sum / num_samples
            results = {"reward": avg_reward.item(), "success": success_rate.item()}

            with open(self.results_path, "w") as f:
                json.dump(results, f)

        with open(self.answer_dump_path, "w") as f:
            for completion in all_completions:
                f.write(completion + "\n\n")


    def cleanup(self) -> None:
        destroy_process_group()

    def cleanup_after_step(
        self,
        trajectory: GRPOTrajectory,
        l_grpo_stats: list[GRPOStats],
    ) -> None:
        for v in trajectory:
            del v
        del trajectory
        for g in l_grpo_stats:
            for v in g:
                del v
            del g
        del l_grpo_stats


@config.parse
def recipe_main(cfg: DictConfig) -> None:
    """
    Entry point for the recipe.

    Configurable parameters are read in the following order:
        - Parameters specified in config (see available configs through ``tune ls``)
        - Overwritten by arguments from the command-line
    """

    recipe = FullGRPOEval(cfg=cfg)
    config.log_config(recipe_name="FullGRPOEval", cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())
