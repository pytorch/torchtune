# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys

# from copy import deepcopy

from functools import partial
from typing import Any, Dict, Optional, Tuple

import torch

from omegaconf import DictConfig, ListConfig
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler
from torchtune import config, modules, utils

# from torchtune.data import CROSS_ENTROPY_IGNORE_IDX

from torchtune.datasets import ConcatDataset

# from torchtune.modules.peft.peft_utils import (
#     disable_adapter,
#     get_adapter_params,
#     get_merged_lora_ckpt,
#     set_trainable_params,
#     validate_state_dict_for_lora,
# )
from torchtune.recipe_interfaces import FTRecipeInterface
from tqdm import tqdm

log = utils.get_logger("DEBUG")


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
        # setup tokenizer
        self._tokenizer = config.instantiate(cfg.tokenizer)
        log.info("Tokenizer is initialized from file.")

        # create a ref copy of the base model and disable grad
        # TODO dont need this if we use lora
        # self._ref_model = deepcopy(self._model)

        # setup opt
        # _setup_optimizer should take in ckpt_dict only if training is resumed from
        # checkpoint. Transforming the opt state dict is handled by this method
        self._optimizer = self._setup_optimizer(
            cfg_optimizer=cfg.optimizer,
            opt_state_dict=(
                self._model.state_dict() if self._resume_from_checkpoint else None
            ),
        )

        # setup lossfn

        # TODO

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
        self.num_steps = cfg.num_steps
        self.batch_size = cfg.batch_size
        self.ppo_epochs = cfg.ppo_epochs
        self.ppo_batch_size = cfg.ppo_batch_size
        self.ppo_backward_batch_size = (
            cfg.ppo_batch_size // cfg.gradient_accumulation_steps
        )
        self._gradient_accumulation_steps = cfg.gradient_accumulation_steps

        self.total_epochs = self.num_steps // self.batch_size

        # Learning rate scheduler can only be set up after number of steps
        # has been computed
        self._lr_scheduler = self._setup_lr_scheduler(
            cfg_lr_scheduler=cfg.lr_scheduler,
            num_training_steps=self.num_steps,
            last_epoch=self.global_step - 1,
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

    def estimate_advantages(
        self, rewards: torch.Tensor, values: torch.Tensor
    ) -> torch.Tensor:
        pass

    def train(self) -> None:
        """
        The core training loop."""
        for curr_epoch in range(self.total_epochs):
            self._sampler.set_epoch(curr_epoch)
            with self._profiler:
                pbar = tqdm(total=self.num_steps)
                for idx, batch in enumerate(self._dataloader):
                    if self._profiler_enabled:
                        self._profiler.step()

                    # this should support any dataset format
                    input_ids, *_ = batch
                    input_ids = input_ids.to(self._device)

                    # policy and value estimates for the current optimisation step
                    # pi_{theta_old] and V_{phi_old}
                    logits, v = self._model(input_ids)

                    # reference logits from reference model
                    ref_logits = self._ref_model(input_ids)

                    # return for episode
                    rewards = self._reward_model(input_ids)

                    advantages = self.estimate_advantages

                    for cur_ppo_epoch in self.ppo_epochs:
                        # TODO (SalmanMohammadi): Add support for early stopping
                        # shuffle batch indices
                        batch_idxs = torch.randperm(self.batch_size)
                        for i in range(0, self.batch_size, self.ppo_batch_size):
                            mini_batch_idxs = batch_idxs[i : i + self.ppo_batch_size]
                            for j in range(
                                0, self.ppo_batch_size, self.ppo_backward_batch_size
                            ):
                                # forward pass
                                # grab preds, value estimates from policy at current batch optimisation step
                                # loss function
                                # loss.backward
                                if j % self._gradient_accumulation_steps == 0:
                                    self._optimizer.step()

            self.save_checkpoint(epoch=curr_epoch)

            # sample queries and rewards from dataset
            # sample reference outputs
            # sample responses and value estimations from current policy
            # for ppo_epoch in ppo_eoppchs:
            #   break condition
            #   shuffle batch indiex
            #   for i in 0, batch_size, step=mini_batch_size:
            # 	    for j in 0, mini_batch_size, step=micro_batch_size:
            # 		    #forward pass - grab preds, value estimates
            # 		    #from policy at current batch optimisation step
            # 		    # loss fn
            # 		    # loss.backward
            # 		    if j % grad_accm_steps == 0:
            # 			    optimizer.step()

        # # grab queries, responses, scores
        # queries [seq_len], responses [response_len], scores [response_len]
        # maybe config can have a
        # reward:
        #   some specification of a reward source
        #   these will be something we pre-specify, like from a dataset,
        #   or even some custom metric
        #   or a reward model
        #   maybe in setup we can have a get_rewards(ppo_config)
        #   which will assume either a specific dataset format, or a reward model
        #
        pass

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
    # recipe.train()
    # recipe.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())
