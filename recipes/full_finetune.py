# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys

from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from warnings import warn

import torch
from omegaconf import DictConfig

from torch import nn
from torch.cuda.amp import GradScaler
from torch.distributed import init_process_group
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler

from torchtune import config, modules, utils

from torchtune.recipe_interfaces import FTRecipeInterface
from torchtune.utils import (
    CheckpointFormat,
    FullModelCheckpointer,
    is_torchtune_checkpoint,
    ModelType,
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


class FullFinetuneRecipe(FTRecipeInterface):
    """
    Full finetuning recipe for dense transformer-based LLMs such as Llama2.

    This recipe supports:
        - FSDP and activation checkpointing. This is enabled by default but can be
            configured using the ``enable_fsdp`` and ``enable_activation_checkpointing`` flags.
        - Mixed precision training - fp32, fp16 and bf16 are supported.
        - Checkpointing of model weights, optimizer state and the recipe state (epoch and seed).
        - Resuming from checkpoints saved using the ``save_checkpoint`` functionality.
        - Logging to terminal. WandB and TensorBoard are currently not supported.

    Assumptions:
        - Training is launched with the Tune CLI (recommended) which uses TorchRun under the
            hood. Setting up the env variables is handled by TorchRun.
        - Training happens on CUDA (CPU training is not supported)
        - Checkpoints are ONLY saved at epoch boundaries. Mid-epoch checkpointing is NOT supported.
        - Datasets are Map-style and data fits in memory (not streamed).

    The following configs can be used to run this recipe:
        >>> tune ls
        RECIPE               CONFIG
        full_finetune        alpaca_llama2_full_finetune

    Args:
        cfg (DictConfig): OmegaConf object parsed from yaml file
    """

    def __init__(self, cfg: DictConfig) -> None:

        self._device = utils.get_device(device=cfg.device)
        self._dtype = utils.get_dtype(dtype=cfg.dtype)

        # logging attributes
        self._output_dir = Path(cfg.output_dir)
        self._log_every_n_steps = cfg.log_every_n_steps if cfg.log_every_n_steps else 1

        # _is_rank_zero is used primarily for logging. In the future, the logger
        # should directly take care of this
        _, rank = utils.get_world_size_and_rank()
        self._is_rank_zero = rank == 0

        # Training cfg
        self._resume_from_checkpoint = cfg.resume_from_checkpoint
        self._enable_fsdp = cfg.enable_fsdp
        self._gradient_accumulation_steps = cfg.gradient_accumulation_steps

        # These are public properties which are updated by the checkpoint loader
        # when ``resume_from_checkpoint`` is `True` or validated in tests
        self.seed = utils.set_seed(seed=cfg.seed)
        self.epochs_run = 0
        self.total_epochs = cfg.epochs
        self.max_steps_per_epoch = cfg.max_steps_per_epoch
        self.total_training_steps = 0

    def load_checkpoint(self, cfg: DictConfig) -> Dict[str, Any]:
        """
        Extract and load state dict from checkpoint file.

        This recipe supports two scenarios:
            * New Training Run. Load a checkpoint from an external source such as Llama2
                from Meta or HuggingFace. In this scenario we need to conver the checkpoint
                into a format used by TorchTune. This can be easily done using the
                FullModelCheckpointer's ``convert_to_torchtune_format`` method.

            * Resuming Training. Load a checkpoint from a partially completed TorchTune
                training. In this scenario, the checkpoint is already in the TorchTune
                format. We simply extract the relevant recipe state and continue with recipe
                setup.
        """
        self._checkpoint_format = getattr(CheckpointFormat, cfg.checkpoint_format)
        self._model_type = getattr(
            ModelType,
            cfg.model_type,
        )
        self._checkpointer = FullModelCheckpointer(
            checkpoint_dir=Path(cfg.checkpoint_dir),
            checkpoint_files=[Path(ckpt_file) for ckpt_file in cfg.checkpoint_files],
            checkpoint_format=self._checkpoint_format,
            model_type=self._model_type,
            output_dir=self._output_dir,
            resume_from_checkpoint=self._resume_from_checkpoint,
        )
        self._is_torchtune_checkpoint = is_torchtune_checkpoint(self._checkpoint_format)

        checkpoint_dict = self._checkpointer.load_checkpoint()

        # If we're resuming training, update the recipe state
        if self._resume_from_checkpoint:
            self._update_recipe_state(checkpoint_dict)

        # If the checkpoint is not in a format compatible with TorchTune, then first
        # convert this
        if not self._is_torchtune_checkpoint:
            checkpoint_dict = self._checkpointer.convert_to_torchtune_format(
                checkpoint_dict
            )
        return checkpoint_dict

    def setup(self, cfg: DictConfig) -> None:
        """
        Sets up the recipe state correctly. This includes setting recipe attributes based
        on the ``resume_from_checkpoint`` flag.
        """
        self._metric_logger = config.instantiate(cfg.metric_logger)

        ckpt_dict = self.load_checkpoint(cfg.model_checkpoint)

        # ``_setup_model`` handles initialization and loading the state dict. This method
        # should be called before ``_setup_optimizer`` since transforming the optimizer
        # state dict requires the model
        self._model = self._setup_model(
            cfg_model=cfg.model,
            enable_fsdp=cfg.enable_fsdp,
            enable_activation_checkpointing=cfg.enable_activation_checkpointing,
            model_state_dict=ckpt_dict[MODEL_KEY]
            if self._resume_from_checkpoint
            else ckpt_dict,
        )

        self._tokenizer = config.instantiate(cfg.tokenizer)
        if self._is_rank_zero:
            log.info("Tokenizer is initialized from file.")

        # _setup_optimizer should take in ckpt_dict only if training is resumed from
        # checkpoint. Transforming the opt state dict is handled by this method
        self._optimizer = self._setup_optimizer(
            cfg_optimizer=cfg.optimizer,
            opt_state_dict=ckpt_dict[OPT_KEY] if self._resume_from_checkpoint else None,
        )

        self._loss_fn = config.instantiate(cfg.loss)
        if self._is_rank_zero:
            log.info("Loss is initialized.")

        # sampler and dataloader depend on the tokenizer and loss_fn and should be
        # setup after both of these are initialized
        self._sampler, self._dataloader = self._setup_data(
            cfg_dataset=cfg.dataset,
            shuffle=cfg.shuffle,
            batch_size=cfg.batch_size,
        )

        # training setup
        self._autocast = utils.get_autocast(self._dtype, self._device)
        self._grad_scaler = None
        if self._dtype == torch.float16:
            self._grad_scaler = utils.get_gradient_scaler(fsdp=cfg.enable_fsdp)
        else:
            self._grad_scaler = GradScaler(enabled=False)

        # Finally update the recipe state which can only be correctly set after all of the
        # other components have been initialized and updated.
        #
        # Number of training steps in each epoch depends on the number of batches produced
        # by the dataloader, the max_steps_per_epoch param set by the user and the
        # gradient_accumulation_steps param. This value is used for logging and tracking
        # training state. The computation should happen after the dataloader has been setup
        self._steps_per_epoch = (
            len(self._dataloader) // self._gradient_accumulation_steps
        )
        if (
            self.max_steps_per_epoch is not None
            and self.max_steps_per_epoch < self._steps_per_epoch
        ):
            self._steps_per_epoch = self.max_steps_per_epoch
        self.total_training_steps = self.epochs_run * self._steps_per_epoch

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
        enable_fsdp: bool,
        enable_activation_checkpointing: bool,
        model_state_dict: Dict[str, Any],
    ) -> nn.Module:
        """
        Set up the model including enabling FSDP and activation checkpointing. For this recipe,
        ``enable_fsdp`` should always be ``True``. This is currently a configurable flag for
        running tests on CPUs.
        """
        with self._device:
            model = config.instantiate(cfg_model)

        model = (
            utils.wrap_fsdp(
                model=model,
                device=self._device,
                dtype=self._dtype,
                strategy="FULL_SHARD",
                auto_wrap_policy={modules.TransformerDecoderLayer},
            )
            if enable_fsdp
            else model
        )
        if enable_activation_checkpointing:
            utils.set_activation_checkpointing(
                model, auto_wrap_policy={modules.TransformerDecoderLayer}
            )

        model.load_state_dict(model_state_dict)

        if self._is_rank_zero:
            log.info("Model is initialized.")
        return model

    def _setup_optimizer(
        self, cfg_optimizer: DictConfig, opt_state_dict: Optional[Dict[str, Any]] = None
    ) -> Optimizer:
        """
        Set up the optimizer. This method also handles transforing the state dict
        for FSDP.
        """
        optimizer = config.instantiate(cfg_optimizer, self._model.parameters())

        if opt_state_dict:
            opt_state_dict = utils.transform_opt_state_dict(
                opt_state_dict, self._model, optimizer
            )
            optimizer.load_state_dict(opt_state_dict)

        if self._is_rank_zero:
            log.info("Optimizer is initialized.")
        return optimizer

    def _setup_data(
        self,
        cfg_dataset: DictConfig,
        shuffle: bool,
        batch_size: int,
    ) -> Tuple[DistributedSampler, DataLoader]:
        """
        All data related setup happens here. Currently this recipe only supports the
        DistributedSamplers with Map-style Datasets which fit into memory. Other samplers,
        iterable datasets and streaming datasets are not supported.
        """
        world_size, rank = utils.get_world_size_and_rank()
        ds = config.instantiate(
            cfg_dataset,
            tokenizer=self._tokenizer,
        )
        sampler = DistributedSampler(
            ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=0,
        )
        dataloader = DataLoader(
            dataset=ds,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=partial(
                utils.padded_collate,
                padding_idx=self._tokenizer.pad_id,
                ignore_idx=self._loss_fn.ignore_index,  # TODO support loss without ignore_index
            ),
        )

        if self._is_rank_zero:
            log.info("Dataset and Sampler are initialized.")

        return sampler, dataloader

    def save_checkpoint(self, epoch: int) -> None:
        """
        Extract and load state dict from checkpoint file.

        This recipe supports two scenarios:
            * Mid-training checkpointing. In this scenario, the checkpoint contains more
                information than just model state. This includes the optimizer and recipe
                states. The recipe is responsible for correctly creating the state dict and
                passing this onto the checkpointers ``save_checkpoint`` method.

            * End-of-training checkpointing. In this scenario, the checkpoint contains only
                the model state. We first convert the state dict back to the original checkpoint
                format using the ``convert_from_torchtune_format`` method and then write this out
                to file using the ``save_checkpoint`` method.
        """
        # if training is in-progress, checkpoint the optimizer state as well
        if epoch + 1 < self.total_epochs:
            ckpt_dict = {MODEL_KEY: self._model.state_dict()}
            optimizer_state_dict = (
                FSDP.optim_state_dict(self._model, self._optimizer)
                if utils.contains_fsdp(self._model)
                else self._optimizer.state_dict()
            )
            ckpt_dict.update(
                {
                    OPT_KEY: optimizer_state_dict,
                    SEED_KEY: self.seed,
                    EPOCHS_KEY: self.epochs_run,
                    TOTAL_EPOCHS_KEY: self.total_epochs,
                    MAX_STEPS_KEY: self.max_steps_per_epoch,
                }
            )
            if self._is_rank_zero:
                self._checkpointer.save_checkpoint(
                    ckpt_dict,
                    intermediate_checkpoint=True,
                    intermediate_checkpoint_name=f"model_{epoch}.pt",
                )
        else:
            ckpt_dict = self._model.state_dict()

            # state dict conversion is not needed if we expect Torchtune compatible
            # checkpoints at the output
            ckpt_dict = self._checkpointer.convert_from_torchtune_format(ckpt_dict)
            if self._is_rank_zero:
                self._checkpointer.save_checkpoint(ckpt_dict)

    def _should_update_weights(self, current_iteration: int) -> bool:
        """
        Determines whether the weights should be updated on the current iteration or not.
        True is returned either if we've accumulated gradients for enough steps or if this
        is the last step in the epoch.
        """
        should_update_weights = (
            current_iteration + 1
        ) % self._gradient_accumulation_steps == 0
        return should_update_weights

    def train(self) -> None:
        """
        The core training loop. Supports training on subsets of the dataset using the
        ``max_steps_per_epoch``.
        """
        _, rank = utils.get_world_size_and_rank()

        # zero out the gradients before starting training
        self._optimizer.zero_grad()

        # self.epochs_run should be non-zero when we're resuming from a checkpoint
        for curr_epoch in range(self.epochs_run, self.total_epochs):

            # Update the sampler to ensure data is correctly shuffled across epochs
            # in case shuffle is True
            self._sampler.set_epoch(curr_epoch)

            for idx, batch in enumerate(
                pbar := tqdm(self._dataloader, disable=not (rank == 0))
            ):
                if (
                    self.max_steps_per_epoch is not None
                    and (idx // self._gradient_accumulation_steps)
                    == self.max_steps_per_epoch
                ):
                    break

                input_ids, labels = batch
                input_ids = input_ids.to(self._device)
                labels = labels.to(self._device)

                with self._autocast:
                    logits = self._model(input_ids)
                    # Shift so that tokens < n predict n
                    logits = logits[..., :-1, :].contiguous()
                    labels = labels[..., 1:].contiguous()
                    logits = logits.transpose(1, 2)
                    # Compute loss
                    loss = self._loss_fn(logits, labels)

                # Note: We're always logging the loss before normalizing it
                # Check if this is the norm or not
                pbar.set_description(f"{curr_epoch+1}|{idx+1}|Loss: {loss.item()}")

                if self.total_training_steps % self._log_every_n_steps == 0:
                    self._metric_logger.log_dict(
                        {
                            "loss": loss.item(),
                            "lr": self._optimizer.param_groups[0]["lr"],
                            "gpu_resources": torch.cuda.memory_allocated(),
                        },
                        step=self.total_training_steps,
                    )

                # Does loss normalization need to happen within autocast context?
                loss = loss / self._gradient_accumulation_steps
                self._grad_scaler.scale(loss).backward()
                if self._should_update_weights(idx):
                    self._grad_scaler.step(self._optimizer)
                    self._grad_scaler.update()
                    self._optimizer.zero_grad(set_to_none=True)

                    # Update the number of steps when the weights are updated
                    self.total_training_steps += 1

            self.epochs_run += 1
            self.save_checkpoint(epoch=curr_epoch)

    def cleanup(self) -> None:
        self._metric_logger.close()


@config.parse
def recipe_main(cfg: DictConfig) -> None:
    """
    Entry point for the recipe.

    Configurable parameters are read in the following order:
        - Parameters specified in ``alpaca_llama2_full_finetune.yaml``
        - Overwritten by arguments from the command-line using ``--override``
    """
    if utils.is_distributed():
        init_process_group(backend="nccl")

    recipe = FullFinetuneRecipe(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())
