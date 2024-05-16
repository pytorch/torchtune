# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys

from functools import partial
from typing import Any, Dict, Optional, Tuple
from warnings import warn

import numpy as np

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler
from torchtune import config, utils

from torchtune.datasets import ConcatDataset
from torchtune.models.mistral.modules import TransformerLMWithValueHead
from torchtune.models.mistral.utils import generate_next_token_with_value_head_model

from torchtune.modules.peft.peft_utils import (
    disable_adapter,
    get_adapter_params,
    get_merged_lora_ckpt,
    set_trainable_params,
    validate_state_dict_for_lora,
)
from torchtune.recipe_interfaces import FTRecipeInterface
from torchtune.utils.pooling import pool_sequence_logits
from torchtune.utils.ppo_utils import (
    AdaptiveKLController,
    estimate_advantages,
    get_rewards,
    whiten,
)
from tqdm import tqdm

log = utils.get_logger("DEBUG")

import scipy


def discounted_cumsum(x: np.ndarray, gamma: float) -> np.ndarray:
    """
    rllab method for exponentially discounted cumulative sum of vectors.
    Args:
        x: A vector of length n [x0, x1, ..., xn]
        gamma: discount factor in [0, 1]
    """
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


class LoRAPPORecipeSingleDevice(FTRecipeInterface):
    def __init__(self, cfg: DictConfig) -> None:

        self._device = utils.get_device(device=cfg.device)
        # Reduced precision logic
        self._dtype = utils.get_dtype(cfg.dtype, device=self._device)
        # fp16 precision is explicitly disabled as it is not supported in this
        # recipe (for example, no gradient scaling).
        if self._dtype == torch.float16:
            raise ValueError(
                "fp16 precision is not supported in this recipe. Please use fp32 or bf16."
            )
        # For CUDA devices, check if the HW supports bf16 if bf16 is specified.
        if (
            self._dtype == torch.bfloat16
            and self._device != torch.device("cpu")
            and not torch.cuda.is_bf16_supported()
        ):
            raise RuntimeError("Full bf16 training is not supported on this hardware.")
        # logging attributes
        self._output_dir = cfg.output_dir
        self._log_every_n_steps = cfg.get("log_every_n_steps", 1)
        self._log_peak_memory_stats = cfg.get("log_peak_memory_stats", False)

        # These are public properties which are updated by the checkpoint loader
        # when ``resume_from_checkpoint`` is `True` or validated in tests
        self.seed = utils.set_seed(seed=cfg.seed)
        self.epochs_run = 0
        self.total_epochs = 0
        self.global_step = 0

        self._resume_from_checkpoint = cfg.resume_from_checkpoint
        self._gradient_accumulation_steps = cfg.gradient_accumulation_steps

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
            if utils.ADAPTER_KEY not in checkpoint_dict:
                raise ValueError(
                    "Adapter weights not found. Please ensure a valid adapter checkpoint is provided."
                )
            # _update_recipe_state will throw an exception if the recipe state is not correctly loaded
            # no need to check here
            self._update_recipe_state(checkpoint_dict)
        return checkpoint_dict

    def save_checkpoint(self, epoch: int) -> None:
        """
        Checkpoint the state of the recipe. The constructed checkpoint state dict
        contains the following information:
        - Merged weights with key MODEL_KEY
        - Adapter weights with key ADAPTER_KEY
        - Relevant recipe state if training is not complete

        Checkpointer will save the merged weights, adapter weights and recipe state in
        different checkpoint files. To correctly resume from training, the adapter weights
        and recipe state must be provided along with the base model weights.
        """
        ckpt_dict = {}
        # if training is in-progress, checkpoint the optimizer state as well
        if epoch + 1 < self.total_epochs:
            ckpt_dict.update(
                {
                    utils.OPT_KEY: self._optimizer.state_dict(),
                    utils.SEED_KEY: self.seed,
                    utils.EPOCHS_KEY: self.epochs_run,
                    utils.TOTAL_EPOCHS_KEY: self.total_epochs,
                }
            )

        # Move to CPU to avoid a copy on GPU
        state_dict = {k: v.cpu() for k, v in self._model.state_dict().items()}

        # Construct the full state dict with LoRA weights merged into base LLM weights
        merged_state_dict = get_merged_lora_ckpt(
            state_dict,
            rank=self._lora_rank,
            alpha=self._lora_alpha,
        )
        ckpt_dict.update({utils.MODEL_KEY: merged_state_dict})

        # Construct the adapter weights
        adapter_key_filter = lambda x: x in self.adapter_params
        adapter_state_dict = {
            k: v for k, v in self._model.state_dict().items() if adapter_key_filter(k)
        }
        ckpt_dict.update({utils.ADAPTER_KEY: adapter_state_dict})
        self._checkpointer.save_checkpoint(
            ckpt_dict,
            epoch=epoch,
            intermediate_checkpoint=(epoch + 1 < self.total_epochs),
        )

    def _update_recipe_state(self, ckpt_dict: Dict[str, Any]) -> None:
        """
        Updates the recipe state from checkpoint.
        """
        # If seed or total_epoch,
        # warn the user and overwrite
        if (
            self.seed != ckpt_dict[utils.SEED_KEY]
            or self.total_epochs != ckpt_dict[utils.TOTAL_EPOCHS_KEY]
        ):
            warn(
                message="""Configured value for seed or epochs
                does not match the value stored in checkpoint."""
            )
        self.seed = utils.set_seed(seed=ckpt_dict[utils.SEED_KEY])
        self.epochs_run = ckpt_dict[utils.EPOCHS_KEY]
        self.total_epochs = ckpt_dict[utils.TOTAL_EPOCHS_KEY]

    def setup(self, cfg: DictConfig) -> None:
        """
        Setup the recipe state. This includes recipe state (if resume_from_checkpoint is True),
        model, tokenizer, loss, optimizer, learning rate scheduler, sampler, and dataloader.
        """
        self._metric_logger = config.instantiate(cfg.metric_logger)

        # log config with parameter override
        self._metric_logger.log_config(cfg)

        # load base model checkpoint
        model_checkpoint_dict = self.load_checkpoint(cfg_checkpointer=cfg.checkpointer)

        # load reward model checkpoint
        reward_model_checkpoint_dict = self.load_checkpoint(
            cfg_checkpointer=cfg.reward_checkpointer
        )

        # ``_setup_model`` handles initialization and loading the state dict. This method
        # should be called before ``_setup_optimizer`` since transforming the optimizer
        # state dict requires the model
        self._model, self._reward_model = self._setup_model(
            cfg_model=cfg.model,
            cfg_reward_model=cfg.reward_model,
            enable_activation_checkpointing=cfg.enable_activation_checkpointing,
            model_state_dict=model_checkpoint_dict[utils.MODEL_KEY],
            reward_model_state_dict=reward_model_checkpoint_dict[utils.MODEL_KEY],
            model_lora_weights_state_dict=(
                model_checkpoint_dict[utils.ADAPTER_KEY]
                if self._resume_from_checkpoint
                else None
            ),
        )
        # setup response length
        self.max_generated_tokens = cfg.max_generated_tokens
        self._model.max_seq_len = self.max_generated_tokens
        # setup tokenizer
        self._tokenizer = config.instantiate(cfg.tokenizer)
        log.info("Tokenizer is initialized from file.")

        # setup opt
        # _setup_optimizer should take in ckpt_dict only if training is resumed from
        # checkpoint. Transforming the opt state dict is handled by this method
        self._optimizer = self._setup_optimizer(
            cfg_optimizer=cfg.optimizer,
            opt_state_dict=(
                self._model.state_dict() if self._resume_from_checkpoint else None
            ),
        )

        # GAE hyperparameters
        self.gamma = cfg.loss.gamma
        self.lmbda = cfg.loss.lmbda
        self.whiten_rewards = cfg.whiten_rewards
        self._loss_fn = config.instantiate(cfg.loss)
        log.info("Loss is initialized.")

        # sampler and dataloader depends on the tokenizer and should be set
        # setup afterit is initialized
        self._sampler, self._dataloader = self._setup_data(
            cfg_dataset=cfg.dataset,
            shuffle=cfg.shuffle,
            batch_size=cfg.batch_size,
        )

        # setup training params:
        # num_steps - global number of optimisation steps (batches)
        # batch_size - number of samples in a batch
        # ppo_epochs - number of epochs to optimise the policy over a batch of episodes
        # ppo_batch_size - number of minibatches (sampled from a single batch) to optimise the policy over
        # max_generated_tokens - maximum number of tokens to generate in a single forward pass
        self.num_steps = cfg.num_steps
        self.batch_size = cfg.batch_size
        self.ppo_epochs = cfg.ppo_epochs
        self.ppo_batch_size = cfg.ppo_batch_size
        self.ppo_backward_batch_size = (
            cfg.ppo_batch_size // cfg.gradient_accumulation_steps
        )
        self._gradient_accumulation_steps = cfg.gradient_accumulation_steps

        # trajectory generation args
        self.temperature = cfg.temperature
        self.top_k = cfg.top_k
        self.top_p = cfg.top_p

        self.total_epochs = self.num_steps // self.batch_size
        self._steps_per_epoch = (
            len(self._dataloader) // self._gradient_accumulation_steps
        )
        self.global_step = self.epochs_run * self._steps_per_epoch

        # setup adaptive KL controller
        self.kl_controller = AdaptiveKLController(
            cfg.init_kl_coef, cfg.kl_target, cfg.kl_horizon
        )

        self._profiler_enabled = cfg.profiler.enabled
        self._profiler = config.instantiate(cfg.profiler)

    def _setup_model(
        self,
        cfg_model: DictConfig,
        cfg_reward_model: DictConfig,
        enable_activation_checkpointing: bool,
        model_state_dict: Dict[str, Any],
        reward_model_state_dict: Dict[str, Any],
        model_lora_weights_state_dict: Optional[Dict[str, Any]] = None,
    ) -> nn.Module:

        with utils.set_default_dtype(self._dtype), self._device:
            model = config.instantiate(cfg_model)
            reward_model = config.instantiate(cfg_reward_model)

        self._lora_rank = cfg_model.lora_rank
        self._lora_alpha = cfg_model.lora_alpha
        self.adapter_params = get_adapter_params(model)
        set_trainable_params(model, self.adapter_params)

        if enable_activation_checkpointing:
            utils.set_activation_checkpointing(
                model, auto_wrap_policy={TransformerLMWithValueHead}
            )

        validate_state_dict_for_lora(
            lora_attn_modules=cfg_model.lora_attn_modules,
            apply_lora_to_mlp=cfg_model.apply_lora_to_mlp,
            apply_lora_to_output=cfg_model.apply_lora_to_output,
            full_model_state_dict_keys=model.state_dict().keys(),
            lora_state_dict_keys=(
                model_lora_weights_state_dict.keys()
                if model_lora_weights_state_dict is not None
                else None
            ),
            base_model_state_dict_keys=model_state_dict.keys(),
        )
        value_head_state_dict = model.state_dict()
        for k, v in model_state_dict.items():
            if k in value_head_state_dict:
                value_head_state_dict[k] = v

        # load checkpoints
        model.load_state_dict(value_head_state_dict)
        reward_model.load_state_dict(reward_model_state_dict)

        # Validate model was loaded in with the expected dtype.
        utils.validate_expected_param_dtype(model.named_parameters(), dtype=self._dtype)
        log.info(f"Base model is initialized with precision {self._dtype}.")

        utils.validate_expected_param_dtype(
            reward_model.named_parameters(), dtype=self._dtype
        )
        log.info(f"Reward model is initialized with precision {self._dtype}.")

        return model, reward_model

    def _setup_optimizer(
        self, cfg_optimizer: DictConfig, opt_state_dict: Optional[Dict[str, Any]] = None
    ) -> Optimizer:
        optimizer = config.instantiate(cfg_optimizer, self._model.parameters())
        if opt_state_dict:
            optimizer.load_state_dict(opt_state_dict)

        log.info("Optimizer and loss are initialized.")
        return optimizer

    def _setup_lr_scheduler(
        self,
        cfg_lr_scheduler: DictConfig,
        num_training_steps: int,
        last_epoch: int,
    ) -> Optimizer:
        lr_scheduler = config.instantiate(
            cfg_lr_scheduler,
            self._optimizer,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch,
        )

        log.info("Learning rate scheduler is initialized.")
        return lr_scheduler

    def _setup_data(
        self,
        cfg_dataset: DictConfig,
        shuffle: bool,
        batch_size: int,
    ) -> Tuple[DistributedSampler, DataLoader]:
        """
        All data related setup happens here. Currently this recipe only supports
        Map-style Datasets which fit into memory and an option for random shuffling.
        Samplers, iterable datasets, and streaming datasets are not supported.
        """
        if isinstance(cfg_dataset, ListConfig):
            datasets = [
                config.instantiate(single_cfg_dataset, tokenizer=self._tokenizer)
                for single_cfg_dataset in cfg_dataset
            ]
            ds = ConcatDataset(datasets=datasets)
        else:
            ds = config.instantiate(cfg_dataset, tokenizer=self._tokenizer)

        sampler = DistributedSampler(
            ds,
            num_replicas=1,
            rank=0,
            shuffle=shuffle,
            seed=0,
        )
        dataloader = DataLoader(
            dataset=ds,
            sampler=sampler,
            batch_size=batch_size,
            collate_fn=partial(
                utils.padded_collate,
                padding_idx=self._tokenizer.pad_id,
                # TODO (SalmanMohammadi): Add support for other ignore indices
                ignore_idx=0,
            ),
        )

        log.info("Dataset and Sampler are initialized.")

        return sampler, dataloader

    def train(self) -> None:
        """
        The core training loop."""
        step_ = 0
        for curr_epoch in range(self.total_epochs):
            self._sampler.set_epoch(curr_epoch)

            with self._profiler:

                pbar = tqdm(total=self.num_steps)
                batch = next(iter(self._dataloader))
                if self._profiler_enabled:
                    self._profiler.step()

                # generating the current trajectory in inference mode
                with torch.no_grad(), disable_adapter(self._model):
                    # this should support any dataset format
                    input_ids, *_ = batch
                    input_ids = input_ids.to(self._device)

                    responses = utils.generate(
                        model=self._model,
                        prompt=input_ids,
                        max_generated_tokens=self.max_generated_tokens,
                        pad_id=self._tokenizer.pad_id,
                        temperature=self.temperature,
                        top_k=self.top_k,
                        stop_tokens=self._tokenizer.eos_id,
                        custom_generate_next_token=generate_next_token_with_value_head_model,
                    )

                    ref_responses = utils.generate(
                        model=self._model,
                        prompt=input_ids,
                        max_generated_tokens=self.max_generated_tokens,
                        pad_id=self._tokenizer.pad_id,
                        temperature=self.temperature,
                        top_k=self.top_k,
                        stop_tokens=self._tokenizer.eos_id,
                        custom_generate_next_token=generate_next_token_with_value_head_model,
                    )

                    # TODO pad out responses to max_generated_tokens

                    # ref policy and value estimates for the current trajectory
                    # pi_{theta_old] and V_{phi_old}
                    # [b, s, v], [b, s, 1]
                    context_length = input_ids.shape[1]
                    responses = torch.tensor(responses).to(self._device)
                    # TODO (SalmanMohammadi) implement minibatch model forward pass
                    logits, values = self._model(responses)

                    values = values[:, context_length - 1 : -1].squeeze(-1)
                    logits = logits[:, context_length - 1 : -1]
                    logits /= self.temperature
                    # shape [b, s]
                    logprobs = torch.gather(
                        F.log_softmax(logits, dim=-1),
                        2,
                        responses[:, context_length:].unsqueeze(-1),
                    ).squeeze(-1)

                    del logits

                    ref_responses = torch.tensor(ref_responses).to(self._device)
                    # TODO (SalmanMohammadi) implement minibatch model forward pass
                    ref_logits, _ = self._model(responses)

                    ref_logits = ref_logits[:, context_length - 1 : -1]
                    ref_logits /= self.temperature
                    # shape [b, s]
                    ref_logprobs = torch.gather(
                        F.log_softmax(ref_logits, dim=-1),
                        2,
                        responses[:, context_length:].unsqueeze(-1),
                    ).squeeze(-1)

                    del ref_logits

                    # run reward model on query-response sequences: shape [b, 1]
                    # TODO (SalmanMohammadi): Add support for _reward_model and _model using different tokenizers
                    # TODO (SalmanMohammadi): use a classifier signature here rather than value head
                    _, scores = self._reward_model(responses)
                    scores = pool_sequence_logits(
                        input_ids, scores, self._tokenizer.pad_id
                    ).squeeze()
                    rewards, kl, kl_rewards = get_rewards(
                        scores, logprobs, ref_logprobs, self.kl_controller.value
                    )

                    if self.whiten_rewards:
                        # shifting mean is disabled for rewards
                        # https://github.com/huggingface/trl/blob/d1aa0b6b2c8dfd78c0f771759d1ff2469c0e5ed2/trl/trainer/ppo_trainer.py#L1155
                        rewards = whiten(rewards)

                    advantages, returns = estimate_advantages(
                        values, rewards, self.gamma, self.lmbda
                    )

                for cur_ppo_epoch in range(self.ppo_epochs):
                    # TODO (SalmanMohammadi): Add support for early stopping
                    # shuffle batch indices every epoch
                    batch_idxs = torch.randperm(self.batch_size)
                    for i in range(0, self.batch_size, self.ppo_batch_size):
                        mini_batch_idxs = batch_idxs[i : i + self.ppo_batch_size]
                        for j in range(
                            0, self.ppo_batch_size, self.ppo_backward_batch_size
                        ):
                            backward_batch_idxs = mini_batch_idxs[
                                j : j + self.ppo_backward_batch_size
                            ]

                            backward_returns = returns[backward_batch_idxs]
                            backward_advantages = advantages[backward_batch_idxs]
                            backward_logprobs = logprobs[backward_batch_idxs]
                            backward_responses = responses[backward_batch_idxs]
                            backward_values = values[backward_batch_idxs]

                            # policy and value estimates for the current optimisation step
                            # pi_{theta] and V_{phi}
                            # TODO (SalmanMohammadi) implement minibatch model forward pass
                            pi_logits, phi_output = self._model(backward_responses)
                            pi_logits = pi_logits[:, context_length - 1 : -1]
                            pi_logits /= self.temperature
                            pi_logprobs = torch.gather(
                                F.log_softmax(pi_logits, dim=-1),
                                2,
                                backward_responses[:, context_length:].unsqueeze(-1),
                            ).squeeze(-1)

                            phi_output = phi_output[:, context_length - 1 : -1].squeeze(
                                -1
                            )

                            loss, policy_loss, value_loss = self._loss_fn(
                                backward_logprobs,
                                pi_logprobs,
                                backward_advantages,
                                backward_values,
                                backward_returns,
                            )

                            # grab some some stats for logging

                            if j % self._gradient_accumulation_steps == 0:
                                self._optimizer.step()
                                self._optimizer.zero_grad(set_to_none=True)
                                self.global_step += 1

            self.save_checkpoint(epoch=curr_epoch)
            pbar.update(1)
            pbar.set_description(
                f"{curr_epoch+1}|{self.global_step}|reward: {rewards.sum(1).mean()}| loss: {loss}"
            )

            kl = logprobs - ref_logprobs
            self.kl_controller.update(kl.sum(1).mean().item(), curr_epoch)

    def cleanup(self, **kwargs) -> None:
        self._metric_logger.close()


@config.parse
def recipe_main(cfg: DictConfig) -> None:
    """
    Entry point for the recipe.

    Configurable parameters are read in the following order:
        - Parameters specified in config (see available configs through ``tune ls``)
        - Overwritten by arguments from the command-line
    """
    config.log_config(recipe_name="LoRADPORecipeSingleDevice", cfg=cfg)
    print(cfg.checkpointer.output_dir)
    recipe = LoRAPPORecipeSingleDevice(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())
