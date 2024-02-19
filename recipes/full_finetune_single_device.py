# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import sys
import time

from functools import partial
from typing import Any, Dict, Optional, Tuple
from warnings import warn

import torch

from recipes.interfaces import FTRecipeInterface
from recipes.params.full_finetune import FullFinetuneParams

from torch import nn
from torch.cuda.amp import GradScaler
from torch.distributed import init_process_group
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler

from torchtune import datasets, models, modules, utils
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


def memory_stats_log(msg: str) -> str:
    return f"""
    Memory Stats {msg}:
    Memory Allocated: {torch.cuda.memory_allocated() / 1000**3:.2f} GB
    Memory Reserved: {torch.cuda.memory_reserved() / 1000**3:.2f} GB
    Peak Memory: {torch.cuda.max_memory_allocated() / 1000**3:.2f} GB
    """


class FullFinetuneSingleDeviceRecipe(FTRecipeInterface):
    """

    """

    def __init__(self, params: FullFinetuneParams) -> None:

        self._device = utils.get_device(device=params.device)
        self._dtype = utils.get_dtype(dtype=params.dtype)

        # logging attributes
        self._output_dir = params.output_dir
        self._metric_logger = utils.get_metric_logger(
            metric_logger_type=params.metric_logger_type,
            project=params.project,
            log_dir=params.output_dir,
        )
        self._log_every_n_steps = (
            params.log_every_n_steps if params.log_every_n_steps else 1
        )

        # Training params
        self._gradient_accumulation_steps = params.gradient_accumulation_steps

        self.seed = utils.set_seed(seed=params.seed)
        self.epochs_run = 0
        self.total_epochs = params.epochs
        self.max_steps_per_epoch = params.max_steps_per_epoch
        self.total_training_steps = 0

    def load_checkpoint(self, ckpt_path: str):
        """
        Extract the checkpoint state from file and validate.
        """
        ckpt_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        return ckpt_dict

    def setup(self, params: FullFinetuneParams) -> None:
        """
        Sets up the recipe state correctly.
        """

        ckpt_dict = self.load_checkpoint(ckpt_path=params.model_checkpoint)

        log.info(memory_stats_log("after checkpoint load"))

        # ``_setup_model`` handles initialization and loading the state dict
        self._model = self._setup_model(
            model=params.model,
            enable_activation_checkpointing=params.enable_activation_checkpointing,
            model_state_dict=ckpt_dict[MODEL_KEY],
        )

        log.info(memory_stats_log("after model setup"))

        self._tokenizer = self._setup_tokenizer(
            tokenizer=params.tokenizer, tokenizer_checkpoint=params.tokenizer_checkpoint
        )

        if params.optimizer_in_bwd:
            self._optimizer_dict = self._setup_optimizer_in_bwd(
                optimizer=params.optimizer,
                lr=params.lr,
            )
        else:
            self._optimizer = self._setup_optimizer(
                optimizer=params.optimizer,
                lr=params.lr,
            )
        self._optimizer_in_bwd = params.optimizer_in_bwd

        self._loss_fn = self._setup_loss(loss=params.loss)

        # sampler and dataloader depend on the tokenizer and loss_fn and should be
        # setup after both of these are initialized
        self._sampler, self._dataloader = self._setup_data(
            dataset=params.dataset,
            train_on_input=params.train_on_input,
            shuffle=params.shuffle,
            batch_size=params.batch_size,
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
        model: str,
        enable_activation_checkpointing: bool,
        model_state_dict: Dict[str, Any],
    ) -> nn.Module:
        """

        """
        load_on_cpu = True

        t0 = time.time()
        model = models.get_model(
            model,
            device=torch.device("cpu") if load_on_cpu else self._device,
        )

        model.bfloat16()
        model.to(self._device)
        print(f"Time to load model: {time.time() - t0:.02f} seconds.")

        if enable_activation_checkpointing:
            utils.set_activation_checkpointing(
                model, auto_wrap_policy={modules.TransformerDecoderLayer}
            )

        model.load_state_dict(model_state_dict)

        log.info("Model is initialized.")
        return model

    def _setup_tokenizer(
        self, tokenizer: str, tokenizer_checkpoint: str
    ) -> modules.Tokenizer:
        """
        Unlike ```setup_model```, this takes in the checkpoint and loads the sentencepiece
        tokenizer model. This is related to how the tokenizer is implemented and should
        change in a future iteration.
        """
        tokenizer = models.get_tokenizer(tokenizer, path=tokenizer_checkpoint)

        log.info("Tokenizer is initialized from file.")
        return tokenizer

    def _setup_optimizer(self, optimizer: str, lr: float) -> Optimizer:
        """
        Set up the optimizer.
        """

        trainable_params = [p for n, p in self._model.named_parameters() if p.requires_grad]

        if optimizer == "AdamW8Bit":
            try:
                import bitsandbytes as bnb
            except AttributeError as e:
                raise ValueError(
                    "Bits and Bytes is not installed. Please install with `pip install bitsandbytes"
                )
            return bnb.optim.AdamW8bit(trainable_params, lr=lr)
        else:
            try:
                return getattr(torch.optim, optimizer)(
                    trainable_params, lr=lr, foreach=False
                )
            except AttributeError as e:
                raise ValueError(
                    f"{optimizer} is not a valid optimizer from torch.optim"
                ) from e

        log.info("Optimizer is initialized.")
        return optimizer

    def _setup_optimizer_in_bwd(self, optimizer: str, lr: float):
        if optimizer == "AdamW8Bit":
            import bitsandbytes as bnb
            optimizer_dict = {p: bnb.optim.AdamW8bit([p], lr=lr) for p in self._model.parameters()}
        elif optimizer == "AdamW":
            optimizer_dict = {p: torch.optim.AdamW([p], lr=lr) for p in self._model.parameters()}
        elif optimizer == "SGD":
            optimizer_dict = {p: torch.optim.SGD([p], lr=lr) for p in self._model.parameters()}
        else:
            raise ValueError(f"{optimizer} is not supported in bwd")

        def optimizer_hook(parameter) -> None:
            optimizer_dict[parameter].step()
            optimizer_dict[parameter].zero_grad()

        for p in self._model.parameters():
            p.register_post_accumulate_grad_hook(optimizer_hook)

    def _setup_loss(self, loss: str) -> nn.Module:
        loss_fn = modules.get_loss(loss)

        log.info("Loss is initialized.")
        return loss_fn

    def _setup_data(
        self, dataset: str, shuffle: bool, batch_size: int, train_on_input: bool
    ) -> Tuple[DistributedSampler, DataLoader]:
        """
        All data related setup happens here. Currently this recipe only supports the
        DistributedSamplers with Map-style Datasets which fit into memory. Other samplers,
        iterable datasets and streaming datasets are not supported.
        """
        world_size, rank = utils.get_world_size_and_rank()
        ds = datasets.get_dataset(
            dataset,
            split="train",
            tokenizer=self._tokenizer,
            train_on_input=train_on_input,
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

        log.info("Dataset and Sampler are initialized.")

        return sampler, dataloader

    def save_checkpoint(self, epoch: int) -> None:
        """
        Checkpoint the relevant state of a recipe.

        This makes use of the `save_checkpoint` utility which is responsible for
        writing the checkpoint dictionary to file. The contents of the dict are dictated
        by whether training is complete or not.

        If training is ongoing, optimizer state, seed and epochs_run are saved along with the
        model weights.
        """
        os.makedirs(self._output_dir, exist_ok=True)
        output_loc = f"{self._output_dir}/model_{epoch}.ckpt"
        ckpt_dict = {MODEL_KEY: self._model}
        utils.save_checkpoint(ckpt_dict, output_loc)

        log.info(
            f"Model checkpoint of size {os.path.getsize(output_loc) >> 20} MB saved to {output_loc}"
        )

    def _should_update_weights(self, curr_step: int) -> bool:
        """
        Determines whether the weights should be updated on the current step or not.
        True is returned either if we've accumulated gradients for enough steps or if this
        is the last step in the epoch.
        """
        should_update_weights = (
            (curr_step + 1) % self._gradient_accumulation_steps == 0 or
            (curr_step + 1) // self._gradient_accumulation_steps == self._steps_per_epoch
        )
        return should_update_weights

    def train(self) -> None:
        """
        The core training loop. Supports training on subsets of the dataset using the
        ``max_steps_per_epoch``.
        """
        _, rank = utils.get_world_size_and_rank()

        # zero out the gradients before starting training
        if not self._optimizer_in_bwd:
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

                # with self._autocast:
                logits = self._model(input_ids)
                # Shift so that tokens < n predict n
                logits = logits[..., :-1, :].contiguous()
                labels = labels[..., 1:].contiguous()
                logits = logits.transpose(1, 2)
                # Compute loss
                loss = self._loss_fn(logits, labels)

                log.info(memory_stats_log("after fwd"))

                # Note: We're always logging the loss before normalizing it
                # Check if this is the norm or not
                pbar.set_description(f"{curr_epoch+1}|{idx+1}|Loss: {loss.item()}")

                if self.total_training_steps % self._log_every_n_steps == 0:
                    self._metric_logger.log_dict(
                        {
                            "loss": loss.item(),
                            # "lr": self._optimizer.param_groups[0]["lr"],
                            "allocated_memory": torch.cuda.memory_allocated(),
                            "peak_memory": torch.cuda.max_memory_allocated(),
                        },
                        step=self.total_training_steps,
                    )

                # Does loss normalization need to happen within autocast context?
                loss = loss / self._gradient_accumulation_steps
                loss.backward()

                log.info(memory_stats_log("after bwd"))

                if self._should_update_weights(idx):
                    if not self._optimizer_in_bwd:
                        self._optimizer.step()
                        self._optimizer.zero_grad()
                        log.info(memory_stats_log("opt step"))

                    # Update the number of steps when the weights are updated
                    self.total_training_steps += 1

                if self.total_training_steps % 8 == 0:
                    log.info(memory_stats_log("Memory Stats: "))

            self.epochs_run += 1
            self.save_checkpoint(epoch=curr_epoch)

    def cleanup(self) -> None:
        self._metric_logger.close()


def recipe_main() -> None:
    """
    Entry point for the recipe.

    Configurable parameters are read in the following order:
        - Parameters specified in ``FullFinetuneParams``
        - Overwritten by Parameters specified in ``alpaca_llama2_full_finetune.yaml``
        - Overwritten by arguments from the command-line using ``TuneArgumentParser``
    """
    parser = utils.TuneArgumentParser(
        description=FullFinetuneParams.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    args, _ = parser.parse_known_args()
    args = vars(args)
    recipe_params = FullFinetuneParams(**args)

    # Env variables set by torch run; only need to initialize process group
    init_process_group(backend="nccl")

    recipe = FullFinetuneSingleDeviceRecipe(params=recipe_params)
    recipe.setup(params=recipe_params)
    recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())
