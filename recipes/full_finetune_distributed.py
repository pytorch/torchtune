# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys

from functools import partial
from typing import Any, Dict, Optional, Tuple
from warnings import warn

import torch
from omegaconf import DictConfig

from torch import nn
from torch.distributed import init_process_group
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler

from torchtune import config, modules, utils

from torchtune.recipe_interfaces import FTRecipeInterface

from tqdm import tqdm


log = utils.get_logger("DEBUG")


class FullFinetuneRecipe(FTRecipeInterface):
    """
    Full finetuning recipe for dense transformer-based LLMs such as Llama2.

    This recipe supports:
        - FSDP and activation checkpointing. This is enabled by default but can be
            configured using the ``enable_fsdp`` and ``enable_activation_checkpointing`` flags.
        - Full bf16 training via setting the ``dtype`` flag to bf16.
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
        RECIPE                          CONFIG
        full_finetune_distributed       alpaca_llama2_full_finetune_distributed

    Args:
        cfg (DictConfig): OmegaConf object parsed from yaml file

    Raises:
        ValueError: If ``dtype`` is set to fp16.
    """

    def __init__(self, cfg: DictConfig) -> None:

        self._device = utils.get_device(device=cfg.device)
        self._dtype = utils.get_dtype(dtype=cfg.dtype)
        self._training_precision = utils.get_dtype(dtype=cfg.dtype)
        # Disable for fp16, as we haven't validated "full" fp16 with this recipe, nor
        # enabled necessary features such as gradient scaling.
        if self._training_precision == torch.float16:
            raise ValueError(
                "full fp16 training is not supported with this recipe. Please use bf16 or fp32 instead."
            )
        # logging attributes
        self._output_dir = cfg.output_dir
        self._log_every_n_steps = cfg.log_every_n_steps if cfg.log_every_n_steps else 1
        self._log_peak_memory_every_n_steps = 100

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
        Extract the checkpoint state from file and validate. If resume_from_checkpoint
        is True, this also includes the recipe state.
        """
        self._checkpointer = config.instantiate(
            cfg,
            resume_from_checkpoint=self._resume_from_checkpoint,
        )
        checkpoint_dict = self._checkpointer.load_checkpoint()

        if self._resume_from_checkpoint:
            self._update_recipe_state(checkpoint_dict)
        return checkpoint_dict

    def _update_recipe_state(self, ckpt_dict: Dict[str, Any]) -> None:
        """
        Updates the recipe state from checkpoint.
        """
        # If seed, total_epoch or max_steps_per_epoch don't match,
        # warn the user and overwrite
        try:
            if (
                self.seed != ckpt_dict[utils.SEED_KEY]
                or self.total_epochs != ckpt_dict[utils.TOTAL_EPOCHS_KEY]
                or self.max_steps_per_epoch != ckpt_dict[utils.MAX_STEPS_KEY]
            ):
                warn(
                    message="""Configured value for seed, epochs or max_steps_per_epoch
                    does not match the value stored in checkpoint."""
                )
            self.seed = utils.set_seed(seed=ckpt_dict[utils.SEED_KEY])
            self.epochs_run = ckpt_dict[utils.EPOCHS_KEY]
            self.total_epochs = ckpt_dict[utils.TOTAL_EPOCHS_KEY]
            self.max_steps_per_epoch = ckpt_dict[utils.MAX_STEPS_KEY]
        except KeyError as e:
            raise KeyError from e(
                "Checkpoint does not contain the required keys needed for updating recipe state."
                "Are you sure you passed in the right recipe checkpoint?"
            )

    def setup(self, cfg: DictConfig) -> None:
        """
        Sets up the recipe state correctly. This includes setting recipe attributes based
        on the ``resume_from_checkpoint`` flag.
        """
        self._metric_logger = config.instantiate(cfg.metric_logger)

        ckpt_dict = self.load_checkpoint(cfg.checkpointer)

        # ``_setup_model`` handles initialization and loading the state dict. This method
        # should be called before ``_setup_optimizer`` since transforming the optimizer
        # state dict requires the model
        self._model = self._setup_model(
            cfg_model=cfg.model,
            enable_fsdp=cfg.enable_fsdp,
            enable_activation_checkpointing=cfg.enable_activation_checkpointing,
            model_state_dict=ckpt_dict[utils.MODEL_KEY],
        )

        self._tokenizer = config.instantiate(cfg.tokenizer)
        if self._is_rank_zero:
            log.info("Tokenizer is initialized from file.")

        # _setup_optimizer should take in ckpt_dict only if training is resumed from
        # checkpoint. Transforming the opt state dict is handled by this method
        self._optimizer = self._setup_optimizer(
            cfg_optimizer=cfg.optimizer,
            opt_state_dict=(
                ckpt_dict[utils.OPT_KEY] if self._resume_from_checkpoint else None
            ),
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
        with utils.set_default_dtype(self._training_precision), self._device:
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
        # Validate model was loaded in with the expected dtype.
        utils.validate_expected_param_dtype(model, dtype=self._training_precision)
        if self._is_rank_zero:
            log.info(f"Model is initialized with precision {self._training_precision}.")
            log.info(utils.memory_stats_log(
                "Memory Stats after model init:", device=self._device
            ))
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
        Save state dict to file. The recipe save_checkpoint method is responsible for
        correctly creating the checkpoint dict and passing to the checkpointer.
        """
        ckpt_dict = {utils.MODEL_KEY: self._model.state_dict()}
        # if training is in-progress, checkpoint the optimizer state as well
        if epoch + 1 < self.total_epochs:
            optimizer_state_dict = (
                FSDP.optim_state_dict(self._model, self._optimizer)
                if utils.contains_fsdp(self._model)
                else self._optimizer.state_dict()
            )
            ckpt_dict.update(
                {
                    utils.OPT_KEY: optimizer_state_dict,
                    utils.SEED_KEY: self.seed,
                    utils.EPOCHS_KEY: self.epochs_run,
                    utils.TOTAL_EPOCHS_KEY: self.total_epochs,
                    utils.MAX_STEPS_KEY: self.max_steps_per_epoch,
                }
            )
        if self._is_rank_zero:
            self._checkpointer.save_checkpoint(
                ckpt_dict,
                epoch=epoch,
                intermediate_checkpoint=(epoch + 1 < self.total_epochs),
            )

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
                logits = self._model(input_ids)
                # Shift so that tokens < n predict n
                logits = logits[..., :-1, :].contiguous()
                labels = labels[..., 1:].contiguous()
                logits = logits.transpose(1, 2)
                # Compute loss
                loss = self._loss_fn(logits, labels)
                # Note: We're always logging the loss before normalizing it
                # Check if this is the norm or not
                if self.total_training_steps % self._log_every_n_steps == 0:
                    pbar.set_description(f"{curr_epoch+1}|{idx+1}|Loss: {loss.item()}")
                    self._metric_logger.log_dict(
                        {
                            "loss": loss.item(),
                            "lr": self._optimizer.param_groups[0]["lr"],
                            "gpu_resources": torch.cuda.memory_allocated(),
                        },
                        step=self.total_training_steps,
                    )

                loss = loss / self._gradient_accumulation_steps
                loss.backward()
                if self._should_update_weights(idx):
                    self._optimizer.step()
                    self._optimizer.zero_grad(set_to_none=True)
                    # Update the number of steps when the weights are updated
                    self.total_training_steps += 1

                # Log peak memory for iteration
                if self.total_training_steps % self._log_peak_memory_every_n_steps == 0:
                    log.info(utils.memory_stats_log("Memory Stats:", device=self._device))

            self.epochs_run += 1
            self.save_checkpoint(epoch=curr_epoch)

    def cleanup(self) -> None:
        self._metric_logger.close()
        torch.distributed.destroy_process_group()


@config.parse
def recipe_main(cfg: DictConfig) -> None:
    """
    Entry point for the recipe.

    Configurable parameters are read in the following order:
        - Parameters specified in ``alpaca_llama2_full_finetune_distributed.yaml``
        - Overwritten by arguments from the command-line
    """
    if not utils.is_distributed():
        raise RuntimeError(
            "Distributed finetune recipe should be run via a distributed launcher."
            "If using tune CLI, please specify --nnodes 1 and --nproc_per_node [num_gpus]"
        )

    init_process_group(backend="gloo" if cfg.device == "cpu" else "nccl")

    recipe = FullFinetuneRecipe(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())
