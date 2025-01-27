# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import time
from dataclasses import dataclass
from typing import Any, Dict, Union

import torch
from omegaconf import DictConfig

from torch.distributed.checkpoint.state_dict import (
    _init_optim_state,
    set_model_state_dict,
    set_state_dict,
)
from torchtune import config, training, utils
from torchtune.training.checkpointing._checkpointer import DistributedCheckpointer
from torchtune.training.memory import OptimizerInBackwardWrapper

log = utils.get_logger("DEBUG")


@dataclass
class TrainingProgress:
    """
    This is training progress metadata.
    """

    seed: int
    epochs_run: int
    total_epochs: int
    max_steps_per_epoch: int

    def state_dict(self) -> Dict[str, object]:
        return {
            training.SEED_KEY: self.seed,
            training.EPOCHS_KEY: self.epochs_run,
            training.TOTAL_EPOCHS_KEY: self.total_epochs,
            training.MAX_STEPS_KEY: self.max_steps_per_epoch,
        }


class CheckpointClient:
    """
    Stateful checkpointing client for TorchTune recipes. This class is responsible for
    saving and loading checkpoints using the user configured checkpointers or distributed
    checkpointer if asynchronous checkpointing is enabled.

    Args:
        cfg (DictConfig): Configuration object used to instantiate the recipe.
    """

    def __init__(
        self,
        cfg: DictConfig,
    ) -> None:
        self._cfg = cfg

        # _checkpointer is the user configured checkpointer
        self._checkpointer = None

        # DistributedCheckpointer is used for asynchronous checkpointing, if enabled.
        self._dcp_checkpointer = None

        self._resume_from_checkpoint = self._cfg.get("resume_from_checkpoint", False)
        self._enable_async_checkpointing = self._cfg.get(
            "enable_async_checkpointing", False
        )
        self._optimizer_in_bwd = self._cfg.get("optimizer_in_bwd", False)
        self._device = utils.get_device(device=self._cfg.device)

        _, self._rank = utils.get_world_size_and_rank()
        self._is_rank_zero = self._rank == 0

    def _get_checkpointer(self):
        """
        Builds and returns the user configured Checkpointer.
        """
        if not self._checkpointer:
            should_load_recipe_state: bool = (
                False
                if self._enable_async_checkpointing
                else self._resume_from_checkpoint
            )
            self._checkpointer = config.instantiate(
                self._cfg.checkpointer,
                should_load_recipe_state=should_load_recipe_state,
            )
        return self._checkpointer

    def _get_dcp_checkpointer(self):
        """
        Builds and returns the DistributedCheckpointer.
        DistributedCheckpointer is used for asynchronous checkpointing, if enabled.
        Uses the user configured checkpointer directory and outout directories.
        """
        if not self._dcp_checkpointer:
            checkpointer = self._get_checkpointer()

            self._dcp_checkpointer = DistributedCheckpointer(
                checkpoint_dir=checkpointer._checkpoint_dir,
                output_dir=checkpointer._output_dir,
            )

        return self._dcp_checkpointer

    def _save_checkpoint_async(
        self,
        model: torch.nn.Module,
        optimizer: Union[torch.optim.Optimizer, OptimizerInBackwardWrapper],
        training_progress: TrainingProgress,
        epoch: int,
    ) -> None:
        """
        Checkpoint the training state asynchronously as a distributed checkpoint. Saving
        asnchronously unblocks the training sooner to continue for the next epoch.
        The constructed checkpoint state dict contains the following information:
        - Model weights with key training.MODEL_KEY
        - Relevant recipe state, including optimizer, if training is not complete

        To correctly resume training from a distributed checkpoint, user needs to have both
        resume_from_checkpoint and enable_async_checkpointing flags set to True in the config.
        User does not need to provide any paths to checkpoint or recipe files. Latest intermediate
        and valid checkpoint will be loaded from the output directory and training progress will be
        restored automatically.
        """

        if self._is_rank_zero:
            log.info("Saving checkpoint asynchronously. Retrieving full state dict...")
            cp_start = time.perf_counter()

        # Create the checkpoint dict to be sent to the checkpointer and ultimately persisted to storage
        ckpt_dict = {}
        ckpt_dict.update(training_progress.state_dict())

        ckpt_dict[training.MODEL_KEY] = model.state_dict()
        ckpt_dict[training.OPT_KEY] = optimizer.state_dict()

        dcp_saver = self._get_dcp_checkpointer()
        dcp_saver.save_checkpoint(
            ckpt_dict,
            epoch=epoch,
            save_async=True,
        )

        if self._is_rank_zero:
            log.info(
                f"Saving asynchronous checkpoint took {time.perf_counter() - cp_start:.2f} secs"
            )

    def _save_checkpoint_sync(
        self,
        model: torch.nn.Module,
        optimizer: Union[torch.optim.Optimizer, OptimizerInBackwardWrapper],
        training_progress: TrainingProgress,
        epoch: int,
    ) -> None:
        """
        Checkpoint the training state synchronously.
        The constructed checkpoint state dict contains the following information:
        - Model weights with key training.MODEL_KEY
        - Relevant recipe state, including optimizer, if training is not complete

        To correctly resume training from this checkpoint, user needs to have both
        resume_from_checkpoint flag set to True and recipe file paths set in the config.
        """

        intermediate_checkpoint = epoch + 1 < training_progress.total_epochs
        checkpointer = self._get_checkpointer()
        no_dist = not isinstance(checkpointer, DistributedCheckpointer)

        # final dict passed onto the checkpointer
        checkpoint_dict = {}

        if self._is_rank_zero:
            log.info(
                "Saving checkpoint. This may take some time. Retrieving full model state dict..."
            )
            cp_start = time.perf_counter()

        model_state_dict = {}
        optim_state_dict = {}

        if no_dist:
            # To prevent GPU memory from spiking during checkpoint save,
            # we consolidate the full model and optim state dicts on CPU for rank 0
            model_state_dict = training.gather_cpu_state_dict(
                model,
                self._is_rank_zero,
                device=self._device,
            )

            if self._is_rank_zero:
                log.info(
                    f"Getting full model state dict took {time.perf_counter() - cp_start:.2f} secs"
                )
        else:
            model_state_dict = model.state_dict()

        if intermediate_checkpoint:
            if self._is_rank_zero:
                log.info("Getting optimizer state dict...")
                optim_start = time.perf_counter()

            if no_dist:
                if not self._optimizer_in_bwd:
                    optim_state_dict = training.get_full_optimizer_state_dict(
                        model,
                        optimizer,
                        self._is_rank_zero,
                        device=self._device,
                    )
                else:
                    for param, opt in optimizer.optim_map.items():
                        optim_state_dict[
                            param
                        ] = training.get_full_optimizer_state_dict(
                            model, opt, self._is_rank_zero, device=self._device
                        )
            else:
                optim_state_dict = optimizer.state_dict()

            if self._is_rank_zero:
                log.info(
                    f"Getting optimizer state dict took {time.perf_counter() - optim_start:.2f} secs"
                )
        else:
            optim_state_dict = None

        def _save_checkpoint_helper():
            checkpoint_dict.update({training.MODEL_KEY: model_state_dict})
            # if training is in-progress, checkpoint the optimizer state and recipe state
            # as well.
            if intermediate_checkpoint:
                checkpoint_dict.update({training.OPT_KEY: optim_state_dict})
                checkpoint_dict.update(training_progress.state_dict())

            self._get_checkpointer().save_checkpoint(
                checkpoint_dict,
                epoch=epoch,
                intermediate_checkpoint=intermediate_checkpoint,
            )

            if self._is_rank_zero:
                log.info(
                    f"Saving checkpoint took {time.perf_counter() - cp_start:.2f} secs"
                )

        # Now that we have the model and optim state dict, create the actual checkpoint dict
        # to be sent to the checkpointer and ultimately written to file
        if no_dist:
            if self._is_rank_zero:
                _save_checkpoint_helper()

            torch.distributed.barrier()
        else:
            _save_checkpoint_helper()

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Union[torch.optim.Optimizer, OptimizerInBackwardWrapper],
        training_progress: TrainingProgress,
        epoch: int,
    ) -> None:
        """
        Checkpoint the training state.
        The constructed checkpoint state dict contains the following information:
        - Model weights with key training.MODEL_KEY
        - Relevant recipe state, including optimizer state, if training is not complete

        If asynchronous checkpointing is enabled, the checkpoint will be saved asynchronously
        as a distributed checkpoint.
        Otherwise, the checkpoint will be saved synchronously with the
        checkpointer user has configured.
        """
        intermediate_checkpoint = epoch + 1 < training_progress.total_epochs

        if intermediate_checkpoint and self._enable_async_checkpointing:
            self._save_checkpoint_async(model, optimizer, training_progress, epoch)
        else:
            self._save_checkpoint_sync(model, optimizer, training_progress, epoch)

    def load_base_checkpoint(self) -> Dict[str, Any]:
        """
        This method is used to load the base model from the checkpoint
        configured by the user.
        """
        return self._get_checkpointer().load_checkpoint()

    def load_distributed_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Union[torch.optim.Optimizer, OptimizerInBackwardWrapper],
    ) -> Dict[str, Any]:
        """
        This method is used to resume training from a distributed checkpoint state.
        Due to being distributed, this method is called on every rank.
        """
        if self._is_rank_zero:
            dcp_load_start = time.perf_counter()

        if not self._optimizer_in_bwd:
            _init_optim_state(optimizer)

        # Build the state dict to be loaded from the distributed checkpoint
        checkpoint_dict: Dict[str:Any] = {}
        model_state_dict = model.state_dict()
        optim_state_dict = optimizer.state_dict()
        checkpoint_dict.update(
            {
                training.MODEL_KEY: model_state_dict,
                training.OPT_KEY: optim_state_dict,
                training.SEED_KEY: 0,
                training.EPOCHS_KEY: 0,
                training.TOTAL_EPOCHS_KEY: 0,
                training.MAX_STEPS_KEY: 0,
            }
        )

        # Load the checkpoint state dict from the distributed checkpoint
        checkpoint_dict = self._get_dcp_checkpointer().load_checkpoint(checkpoint_dict)

        # Load the checkpoint state dict into model and optimizer
        if not self._optimizer_in_bwd:
            if training.OPT_KEY in checkpoint_dict:
                set_state_dict(
                    model,
                    optimizer,
                    model_state_dict=checkpoint_dict[training.MODEL_KEY],
                    optim_state_dict=checkpoint_dict[training.OPT_KEY],
                )
            else:
                set_model_state_dict(
                    model=model,
                    model_state_dict=checkpoint_dict[training.MODEL_KEY],
                )
        else:
            set_model_state_dict(
                model=model,
                model_state_dict=checkpoint_dict[training.MODEL_KEY],
            )

            if training.OPT_KEY in checkpoint_dict:
                optimizer.load_state_dict(checkpoint_dict[training.OPT_KEY])

        if self._is_rank_zero:
            log.info(
                f"DistributedCheckpointer loaded the checkpoint in {time.perf_counter() - dcp_load_start:.2f} seconds."
            )

        return checkpoint_dict
