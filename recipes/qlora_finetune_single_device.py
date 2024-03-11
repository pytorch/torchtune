# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import os
import sys

from functools import partial
from typing import Any, Dict, Optional, Tuple
from warnings import warn

import torch
from omegaconf import DictConfig

from torchtune.recipe_interfaces import FTRecipeInterface

from torch import nn
from torch.cuda.amp import GradScaler
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler
from torchtune import config, modules, utils
from torchtune.modules.peft.peft_utils import (
    get_adapter_params,
    set_trainable_params,
    validate_state_dict_for_lora,
)
from torchtune.utils.constants import (
    EPOCHS_KEY,
    MAX_STEPS_KEY,
    MODEL_KEY,
    OPT_KEY,
    SEED_KEY,
    TOTAL_EPOCHS_KEY,
)
from tqdm import tqdm

log = utils.get_logger("DEBUG")


class QLoRAFinetuneRecipeSingleDevice(FTRecipeInterface):
    """
    LoRA finetuning recipe for dense transformer-based LLMs such as Llama2 for
    single device training.

    This recipe supports:
        - Activation checkpointing. This is enabled by default but is configurable.
        - Mixed precision training via `torch.autocast` - fp32, fp16 and bf16 are supported.
        - Full bf16 training for supported HW architectures. We currently check bf16 support via
        the `torch.cuda.is_bf16_supported` API. This is disabled by default but can be enabled via
        the "full_bf16" configuration flag.
        - Checkpointing: of LoRA adapter parameters and their optimizer states. When resuming
            from a checkpoint, the adapter parameters are loaded from the checkpoint along
            with the base model weights. Note that intra-epoch resumption is not supported.
        - Logging to terminal, WandB, or TensorBoard.

    Assumptions:
        - Checkpoints are ONLY saved at epoch boundaries. In case of failure, work done
            in ongoing epoch is lost.
        - Datasets are Map-style and data fits in memory (not streamed).

    The following configs can be used to run this recipe:
        >>> tune ls
        RECIPE                          CONFIG
        lora_finetune_single_device        alpaca_llama2_lora_finetune_single_device

    Args:
        cfg (DictConfig): OmegaConf object parsed from yaml file

    """

    def __init__(self, cfg: DictConfig) -> None:

        self._device = utils.get_device(device=cfg.device)

        # logging attributes
        self._output_dir = cfg.output_dir
        self._log_every_n_steps = cfg.log_every_n_steps if cfg.log_every_n_steps else 1

        # These are public properties which are updated by the checkpoint loader
        # when ``resume_from_checkpoint`` is `True` or validated in tests
        self.seed = utils.set_seed(seed=cfg.seed)
        self.epochs_run = 0
        self.total_epochs = cfg.epochs
        self.max_steps_per_epoch = cfg.max_steps_per_epoch
        self.total_training_steps = 0

        self._resume_from_checkpoint = cfg.resume_from_checkpoint

    def setup(self, cfg: DictConfig) -> None:
        """
        Setup the recipe state. This includes recipe state (if resume_from_checkpoint is True),
        model, tokenizer, loss, optimizer, learning rate scheduler, sampler, and dataloader.
        """
        self._metric_logger = config.instantiate(cfg.metric_logger)

        # Load in base model weights
        # Note that we set resume_from_checkpoint=False when loading the base model.
        # This is because we only save LoRA weights during training, so only lora_checkpoint
        # will contain training state, while model_checkpoint contains model weights only.
        base_model_ckpt = self.load_checkpoint(
            ckpt_path=cfg.model_checkpoint, resume_from_checkpoint=False
        )

        # If we're resuming from checkpoint, the recipe's state should be updated before
        # initializing the training components. This ensures that the seed is correctly
        # propagated to the relevant components
        if self._resume_from_checkpoint:
            assert (
                cfg.lora_checkpoint is not None
            ), "Must pass lora_checkpoint when resuming training"
            lora_ckpt = self.load_checkpoint(
                ckpt_path=cfg.lora_checkpoint, resume_from_checkpoint=True
            )
            self._update_recipe_state(lora_ckpt)

        self._model = self._setup_model(
            cfg_model=cfg.model,
            full_bf16=cfg.full_bf16,
            enable_activation_checkpointing=cfg.enable_activation_checkpointing,
            base_model_state_dict=base_model_ckpt[MODEL_KEY],
            lora_weights_state_dict=(
                lora_ckpt[MODEL_KEY] if self._resume_from_checkpoint else None
            ),
        )

        self._tokenizer = config.instantiate(cfg.tokenizer)
        log.info("Tokenizer is initialized from file.")

        self._optimizer = self._setup_optimizer(
            cfg_optimizer=cfg.optimizer,
            opt_state_dict=lora_ckpt[OPT_KEY] if self._resume_from_checkpoint else None,
        )

        self._loss_fn = config.instantiate(cfg.loss)
        log.info("Loss is initialized.")

        # Dataloader depends on the tokenizer and loss_fn and should be
        # setup after all of these are setup
        self._sampler, self._dataloader = self._setup_data(
            cfg_dataset=cfg.dataset,
            shuffle=cfg.shuffle,
            batch_size=cfg.batch_size,
        )

        # Finally update the recipe state which can only be correctly set after all of the
        # other components have been initialized and updated.

        # Number of training steps in each epoch depends on the number of batches produced
        # by the dataloader and the max_steps_per_epoch param set by the user and is used
        # for logging and tracking training state. This should be computed after the dataloader
        # has been setup
        steps_per_epoch = len(self._dataloader)
        if self.max_steps_per_epoch is not None and self.max_steps_per_epoch < len(
            self._dataloader
        ):
            steps_per_epoch = self.max_steps_per_epoch
            self.total_training_steps = self.epochs_run * steps_per_epoch

        # Learning rate scheduler can only be set up after number of steps
        # has been computed
        self._lr_scheduler = self._setup_lr_scheduler(
            cfg_lr_scheduler=cfg.lr_scheduler,
            num_training_steps=self.total_epochs * steps_per_epoch,
            last_epoch=self.total_training_steps - 1,
        )

    def load_checkpoint(self, ckpt_path: str, resume_from_checkpoint: bool):
        """
        Extract the checkpoint state from file and validate.
        """
        ckpt_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        utils.validate_checkpoint(ckpt_dict, resume_from_checkpoint)
        return ckpt_dict

    def _update_recipe_state(self, ckpt_dict: Dict[str, Any]) -> None:
        """
        Updates the recipe state from checkpoint.
        """
        # If seed, total_epoch or max_steps_per_epoch don't match,
        # warn the user and overwrite
        if (
            self.seed != ckpt_dict[SEED_KEY]
            or self.total_epochs != ckpt_dict[TOTAL_EPOCHS_KEY]
            or self.max_steps_per_epoch != ckpt_dict[MAX_STEPS_KEY]
        ):
            warn(
                message="""Configured value for seed, epochs or max_steps_per_epoch
                does not match the value stored in checkpoint."""
            )
        self.seed = utils.set_seed(seed=ckpt_dict[SEED_KEY])
        self.epochs_run = ckpt_dict[EPOCHS_KEY]
        self.total_epochs = ckpt_dict[TOTAL_EPOCHS_KEY]
        self.max_steps_per_epoch = ckpt_dict[MAX_STEPS_KEY]

    def _setup_model(
        self,
        cfg_model: DictConfig,
        full_bf16: bool,
        enable_activation_checkpointing: bool,
        base_model_state_dict: Dict[str, Any],
        lora_weights_state_dict: Optional[Dict[str, Any]] = None,
    ) -> nn.Module:

        dtype_context = (
            utils.set_default_dtype(torch.bfloat16)
            if full_bf16
            else contextlib.nullcontext()
        )
        with dtype_context:
            with self._device:
                model = config.instantiate(cfg_model)

        # Note: this needs to be set before wrapping with FSDP
        self.adapter_params = get_adapter_params(model)
        set_trainable_params(model, self.adapter_params)

        if enable_activation_checkpointing:
            utils.set_activation_checkpointing(
                model, auto_wrap_policy={modules.TransformerDecoderLayer}
            )
        lora_modules = cfg_model.lora_attn_modules + (
            ["w1", "w2", "w3"] if cfg_model.apply_lora_to_mlp else []
        )
        validate_state_dict_for_lora(
            lora_modules=lora_modules,
            full_model_state_dict_keys=model.state_dict().keys(),
            lora_state_dict_keys=(
                lora_weights_state_dict.keys()
                if lora_weights_state_dict is not None
                else None
            ),
            base_model_state_dict_keys=base_model_state_dict.keys(),
        )
        model.load_state_dict(base_model_state_dict, strict=False)
        if lora_weights_state_dict:
            model.load_state_dict(lora_weights_state_dict, strict=False)

        # Validate model was loaded in with the expected dtype.
        expected_dtype = torch.bfloat16 if full_bf16 else torch.float32
        utils.validate_expected_param_dtype(
            model, dtype=expected_dtype
        )

        log.info(f"Model is initialized with precision {expected_dtype}.")
        return model

    def _setup_optimizer(
        self, cfg_optimizer: DictConfig, opt_state_dict: Optional[Dict[str, Any]] = None
    ) -> Optimizer:
        optimizer = config.instantiate(cfg_optimizer, self._model.parameters())
        if opt_state_dict:
            # Note: technically we should check _contains_fsdp for
            # just the state dict of the adapter cfg, but should be equivalent
            opt_state_dict = utils.transform_opt_state_dict(
                opt_state_dict, self._model, optimizer
            )
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
    ) -> DataLoader:
        """
        All data related setup happens here. Currently this recipe only supports
        Map-style Datasets which fit into memory and an option for random shuffling.
        Samplers, iterable datasets, and streaming datasets are not supported.
        """
        ds = config.instantiate(
            cfg_dataset,
            tokenizer=self._tokenizer,
        )
        sampler = DistributedSampler(
            ds,
            num_replicas=1,
            rank=0,
            shuffle=shuffle,
            seed=0,
        )
        dataloader = DataLoader(
            dataset=ds,
            # shuffle=shuffle,
            sampler=sampler,
            batch_size=batch_size,
            collate_fn=partial(
                utils.padded_collate,
                padding_idx=self._tokenizer.pad_id,
                ignore_idx=self._loss_fn.ignore_index,  # TODO support loss without ignore_index
            ),
        )

        log.info("Dataset and Sampler are initialized.")

        return sampler, dataloader

    def save_checkpoint(self, epoch: int) -> None:
        """
        Checkpoint the state of the recipe. Currently this only includes checkpointing
        model weights and optimizer state.
        """
        os.makedirs(self._output_dir, exist_ok=True)
        output_loc = f"{self._output_dir}/model_{epoch}.ckpt"
        ckpt_dict = {MODEL_KEY: self._model}
        # if training is in-progress, checkpoint the optimizer state as well
        if epoch + 1 < self.total_epochs:
            ckpt_dict.update(
                {
                    OPT_KEY: self._optimizer,
                    SEED_KEY: self.seed,
                    EPOCHS_KEY: self.epochs_run,
                    TOTAL_EPOCHS_KEY: self.total_epochs,
                    MAX_STEPS_KEY: self.max_steps_per_epoch,
                }
            )
        utils.save_checkpoint(
            ckpt_dict, output_loc, model_key_filter=lambda x: x in self.adapter_params
        )

        log.info(
            msg=f"Model checkpoint of size {os.path.getsize(output_loc) >> 20} MB saved to {output_loc}"
        )

    def train(self) -> None:
        """
        The core training loop.
        """

        # self.epochs_run should be non-zero when we're resuming from a checkpoint
        for curr_epoch in range(self.epochs_run, self.total_epochs):
            # Update the sampler to ensure data is correctly shuffled across epochs
            # in case shuffle is True
            self._sampler.set_epoch(curr_epoch)
            for idx, batch in enumerate(pbar := tqdm(self._dataloader)):
                if (
                    self.max_steps_per_epoch is not None
                    and idx == self.max_steps_per_epoch
                ):
                    break
                self.total_training_steps += 1
                self._optimizer.zero_grad()

                input_ids, labels = batch
                input_ids = input_ids.to(self._device)
                labels = labels.to(self._device)

                logits = self._model(input_ids)
                # Shift so that tokens < n predict n
                logits = logits[..., :-1, :].contiguous()
                labels = labels[..., 1:].contiguous()
                logits = logits.transpose(1, 2)
                # Compute loss
                loss = self._loss_fn(logits, labels)

                pbar.set_description(f"{curr_epoch+1}|{idx+1}|Loss: {loss.item()}")

                if self.total_training_steps % self._log_every_n_steps == 0:
                    self._metric_logger.log_dict(
                        {
                            "loss": loss.item(),
                            "lr": self._optimizer.param_groups[0]["lr"],
                            "gpu_resources": torch.cuda.memory_allocated(),
                        },
                        step=self.total_training_steps,  # Each step is unique, not limited to each epoch
                    )

                loss.backward()
                import pdb ; pdb.set_trace()
                self._optimizer.step()
                self._lr_scheduler.step()

            self.epochs_run += 1
            self.save_checkpoint(epoch=curr_epoch)

    def cleanup(self) -> None:
        self._metric_logger.close()


@config.parse
def recipe_main(cfg: DictConfig) -> None:
    """
    Entry point for the recipe.

    Configurable parameters are read in the following order:
        - Parameters specified in ``alpaca_llama2_lora_finetune_single_device.yaml``
        - Overwritten by arguments from the command-line using ``--override``
    """
    recipe = QLoRAFinetuneRecipeSingleDevice(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())
