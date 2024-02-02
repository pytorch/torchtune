# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys

from functools import partial
from typing import Any, Dict, Optional, Tuple
from warnings import warn

import torch

from torch import nn
from torch.cuda.amp import GradScaler
from torch.distributed import init_process_group
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler

from torchtune import datasets, losses, models, modules, optim, utils
from torchtune.utils.constants import (
    EPOCHS_KEY,
    MAX_STEPS_KEY,
    MODEL_KEY,
    OPT_KEY,
    SEED_KEY,
    TOTAL_EPOCHS_KEY,
)

from tqdm import tqdm

from recipes.interfaces import FTRecipeInterface
from recipes.params import FullFinetuneParams


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
    """

    def __init__(self, params: FullFinetuneParams) -> None:

        self._device = utils.get_device(device=params.device)
        self._dtype = utils.get_dtype(dtype=params.dtype)

        # logging attributes
        self._output_dir = params.output_dir

        # _is_rank_zero is used primarily for logging. In the future, the logger
        # should directly take care of this
        _, rank = utils.get_world_size_and_rank()
        self._is_rank_zero = rank == 0

        # These are public properties which are updated by the checkpoint loader
        # when ``resume_from_checkpoint`` is `True`
        self.seed = utils.set_seed(seed=params.seed)
        self.epochs_run = 0
        self.total_epochs = params.epochs
        self.max_steps_per_epoch = params.max_steps_per_epoch

        self._resume_from_checkpoint = params.resume_from_checkpoint

    def load_checkpoint(self, ckpt_path: str):
        """
        Extract the checkpoint state from file and validate.
        """
        ckpt_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        utils.validate_checkpoint(ckpt_dict, self._resume_from_checkpoint)
        return ckpt_dict

    def setup(self, params: FullFinetuneParams) -> None:
        """
        Sets up the recipe state correctly. This includes setting recipe attributes based
        on the ``resume_from_checkpoint`` flag.
        """

        ckpt_dict = self.load_checkpoint(ckpt_path=params.model_checkpoint)

        # If we're resuming from checkpoint, the recipe's state should be updated before
        # initializing the training components. This ensures that the seed is correctly
        # propagated to the relevant components
        if self._resume_from_checkpoint:
            self._update_recipe_state(ckpt_dict)

        # ``_setup_model`` handles initialization and loading the state dict. This method
        # should be called before ``_setup_optimizer`` since transforming the optimizer
        # state dict requires the model
        self._model = self._setup_model(
            model=params.model,
            enable_fsdp=params.enable_fsdp,
            enable_activation_checkpointing=params.enable_activation_checkpointing,
            model_state_dict=ckpt_dict[MODEL_KEY],
        )

        self._tokenizer = self._setup_tokenizer(
            tokenizer=params.tokenizer, tokenizer_checkpoint=params.tokenizer_checkpoint
        )

        # _setup_optimizer should take in ckpt_dict only if training is resumed from
        # checkpoint. Transforming the opt state dict is handled by this method
        self._optimizer = self._setup_optimizer(
            optimizer=params.optimizer,
            lr=params.lr,
            opt_state_dict=ckpt_dict[OPT_KEY] if self._resume_from_checkpoint else None,
        )

        self._loss_fn = self._setup_loss(loss=params.loss)

        # sampler and dataloader depend on the tokenizer and loss_fn and should be
        # setup after both of these are initialized
        self._sampler, self._dataloader = self._setup_data(
            dataset=params.dataset,
            shuffle=params.shuffle,
            batch_size=params.batch_size,
        )

        # training setup
        self._autocast = utils.get_autocast(self._dtype, self._device)
        self._grad_scaler = None
        if self._dtype == torch.float16:
            self._grad_scaler = utils.get_gradient_scaler(fsdp=params.enable_fsdp)
        else:
            self._grad_scaler = GradScaler(enabled=False)

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
        model: str,
        enable_fsdp: bool,
        enable_activation_checkpointing: bool,
        model_state_dict: Dict[str, Any],
    ) -> nn.Module:
        """
        Set up the model including enabling FSDP and activation checkpointing. For this recipe,
        ``enable_fsdp`` should always be ``True``. This is currently a configurable flag for
        running tests on CPUs.
        """
        model = models.get_model(model, device=self._device)
        model = (
            utils.get_fsdp(
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
            log.info(
                "Model is initialized. FSDP and Activation Checkpointing are enabled."
            )
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

        if self._is_rank_zero:
            log.info("Tokenizer is initialized from file.")
        return tokenizer

    def _setup_optimizer(
        self, optimizer: str, lr: float, opt_state_dict: Optional[Dict[str, Any]] = None
    ) -> Optimizer:
        """
        Set up the optimizer. This method also handles transforing the state dict
        for FSDP.
        """
        optimizer = optim.get_optimizer(optimizer, self._model, lr)
        if opt_state_dict:
            opt_state_dict = utils.transform_opt_state_dict(
                opt_state_dict, self._model, optimizer
            )
            optimizer.load_state_dict(opt_state_dict)

        if self._is_rank_zero:
            log.info("Optimizer is initialized.")
        return optimizer

    def _setup_loss(self, loss: str) -> nn.Module:
        loss_fn = losses.get_loss(loss)

        if self._is_rank_zero:
            log.info("Loss is initialized.")

        return loss_fn

    def _setup_data(
        self, dataset: str, shuffle: bool, batch_size: int
    ) -> Tuple[DistributedSampler, DataLoader]:
        """
        All data related setup happens here. Currently this recipe only supports the
        DistributedSamplers with Map-style Datasets which fit into memory. Other samplers,
        iterable datasets and streaming datasets are not supported.
        """
        world_size, rank = utils.get_world_size_and_rank()
        ds = datasets.get_dataset(dataset, split="train", tokenizer=self._tokenizer)
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
        utils.save_checkpoint(ckpt_dict, output_loc)

        if self._is_rank_zero:
            log.info(
                f"Model checkpoint of size {os.path.getsize(output_loc) >> 20} MB saved to {output_loc}"
            )

    def train(self) -> None:
        """
        The core training loop. Supports training on subsets of the dataset using the
        ``max_steps_per_epoch``.
        """
        _, rank = utils.get_world_size_and_rank()

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
                    and idx == self.max_steps_per_epoch
                ):
                    break
                self._optimizer.zero_grad()

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

                pbar.set_description(f"{curr_epoch+1}|{idx+1}|Loss: {loss.item()}")

                self._grad_scaler.scale(loss).backward()
                self._grad_scaler.step(self._optimizer)
                self._grad_scaler.update()

            self.epochs_run += 1
            self.save_checkpoint(epoch=curr_epoch)


def recipe_main() -> None:
    """
    Entry point for the recipe.

    Configurable parameters are read in the following order:
        - Parameters specified in ``FullFinetuneParams``
        - Overwritten by Parameters specified in ``alpaca_llama2_full_finetune.yaml``
        - Overwritten by arguments from the command-line using ``TuneArgumentParser``
    """
    parser = utils.TuneArgumentParser(
        description=recipe.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    args, _ = parser.parse_known_args()
    parser.log_args(args)
    args = vars(args)

    recipe_params = FullFinetuneParams(**args)

    # Env variables set by torch run; only need to initialize process group
    init_process_group(backend="nccl")

    recipe = FullFinetuneRecipe(params=recipe_params)
    recipe.setup(params=recipe_params)
    recipe.train()


if __name__ == "__main__":
    sys.exit(recipe_main())
