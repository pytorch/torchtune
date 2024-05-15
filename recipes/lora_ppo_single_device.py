# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys

from copy import deepcopy

from functools import partial
from typing import Any, Dict, Optional, Tuple

import numpy as np

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler
from torchtune import config, modules, utils

# from torchtune.data import CROSS_ENTROPY_IGNORE_IDX

from torchtune.datasets import ConcatDataset
from torchtune.models.mistral.utils import generate_next_token_with_value_head_model

# from torchtune.modules.peft.peft_utils import (
#     disable_adapter,
#     get_adapter_params,
#     get_merged_lora_ckpt,
#     set_trainable_params,
#     validate_state_dict_for_lora,
# )
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
        self.total_training_steps = 0
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
            self._update_recipe_state(checkpoint_dict)
        return checkpoint_dict

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
        self._model_compile = cfg.compile
        self._model, self._reward_model = self._setup_model(
            cfg_model=cfg.model,
            cfg_reward_model=cfg.reward_model,
            enable_activation_checkpointing=cfg.enable_activation_checkpointing,
            compile_model=self._model_compile,
            model_state_dict=model_checkpoint_dict[utils.MODEL_KEY],
            reward_model_state_dict=reward_model_checkpoint_dict[utils.MODEL_KEY],
        )
        # setup response length
        self.max_generated_tokens = cfg.max_generated_tokens
        self._model.max_seq_len = self.max_generated_tokens
        # setup tokenizer
        self._tokenizer = config.instantiate(cfg.tokenizer)
        log.info("Tokenizer is initialized from file.")

        # create a ref copy of the base model and disable grad
        # TODO dont need this if we use lora
        self._ref_model = deepcopy(self._model)

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

        # Learning rate scheduler can only be set up after number of steps
        # has been computed
        self._lr_scheduler = self._setup_lr_scheduler(
            cfg_lr_scheduler=cfg.lr_scheduler,
            num_training_steps=self.num_steps,
            last_epoch=self.global_step - 1,
        )

        # setup adaptive KL controller
        self.kl_controller = AdaptiveKLController(
            cfg.kl_init, cfg.kl_target, cfg.kl_horizon
        )

        self._profiler_enabled = cfg.profiler.enabled
        self._profiler = config.instantiate(cfg.profiler)

    def _setup_model(
        self,
        cfg_model: DictConfig,
        cfg_reward_model: DictConfig,
        enable_activation_checkpointing: bool,
        compile_model: bool,
        model_state_dict: Dict[str, Any],
        reward_model_state_dict: Dict[str, Any],
    ) -> nn.Module:

        with utils.set_default_dtype(self._dtype), self._device:
            model = config.instantiate(cfg_model)
            reward_model = config.instantiate(cfg_reward_model)

        if enable_activation_checkpointing:
            utils.set_activation_checkpointing(
                model, auto_wrap_policy={modules.TransformerDecoderLayer}
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
        # Compile model, if enabled.
        if compile_model:
            log.info("Compiling model with torch.compile...")
            model = utils.wrap_compile(model)
        if self._device.type == "cuda":
            memory_stats = utils.get_memory_stats(device=self._device)
            utils.log_memory_stats(memory_stats)

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
                with torch.no_grad():
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
                        # generate until max_generated_tokens is reached
                        stop_tokens=None,
                        custom_generate_next_token=generate_next_token_with_value_head_model,
                    )

                    ref_responses = utils.generate(
                        model=self._ref_model,
                        prompt=input_ids,
                        max_generated_tokens=self.max_generated_tokens,
                        pad_id=self._tokenizer.pad_id,
                        temperature=self.temperature,
                        top_k=self.top_k,
                        # generate until max_generated_tokens is reached
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
                    ref_logits, _ = self._ref_model(responses)

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
            self.save_checkpoint(epoch=curr_epoch)
            pbar.update(1)
            pbar.set_description(f"{curr_epoch+1}|{self.global_step}|reward: {loss}")
            self.kl_controller.step(kl, curr_epoch)

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
