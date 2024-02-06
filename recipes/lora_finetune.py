# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import sys

from functools import partial
from typing import Any, Dict, List, Tuple

import torch

from torch import nn
from torch.cuda.amp import GradScaler
from torch.distributed import init_process_group
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler

from torchtune import datasets, losses, lr_schedulers, models, modules, optim, utils
from torchtune.modules.peft.peft_utils import get_adapter_params, set_trainable_params
from torchtune.utils.constants import (
    EPOCHS_KEY,
    MAX_STEPS_KEY,
    MODEL_KEY,
    SEED_KEY,
    TOTAL_EPOCHS_KEY,
)
from tqdm import tqdm

from recipes.interfaces import FTRecipeInterface
from recipes.params import LoRAFinetuneParams

log = utils.get_logger("DEBUG")


class LoRAFinetuneRecipe(FTRecipeInterface):
    """
    LoRA finetuning recipe for dense transformer-based LLMs such as Llama2.

    This recipe supports:
        - FSDP and activation checkpointing. This is enabled by default but is
            configurable.
        - Mixed precision training - fp32, fp16 and bf16 are supported.
        - Checkpointing of full model weights and optionally of optimizer state (for
            checkpoints created during training).
        - Logging to terminal, WandB, or TensorBoard.

    Assumptions:
        - Training happens on CUDA (CPU training is not supported)
        - Checkpoints are ONLY saved at epoch boundaries. In case of failure, work done
            in ongoing epoch is lost.
        - Datasets are Map-style and data fits in memory (not streamed).

    TODO:
        - Support saving LoRA params only and resuming from checkpoint.
        - Checkpoint epoch and seed to ensure training restarts are correct.
    """

    def __init__(self, params: LoRAFinetuneParams) -> None:

        self._device = utils.get_device(device=params.device)
        self._dtype = utils.get_dtype(dtype=params.dtype)

        # _is_rank_zero is used primarily for logging. In the future, the logger
        # should directly take care of this
        _, rank = utils.get_world_size_and_rank()
        self._is_rank_zero = rank == 0

        # logging attributes
        self._output_dir = params.output_dir
        self._log_every_n_steps = (
            params.log_every_n_steps if params.log_every_n_steps else 1
        )
        if self._is_rank_zero:
            self._metric_logger = utils.get_metric_logger(
                metric_logger_type=params.metric_logger_type,
                project=params.project,
                log_dir=params.output_dir,
            )

        # These are public properties which are updated by the checkpoint loader
        # when ``resume_from_checkpoint`` is `True` or validated in tests
        self.seed = utils.set_seed(seed=params.seed)
        self.epochs_run = 0
        self.total_epochs = params.epochs
        self.max_steps_per_epoch = params.max_steps_per_epoch
        self.total_training_steps = 0

        self._resume_from_checkpoint = params.resume_from_checkpoint

    def setup(self, params: LoRAFinetuneParams) -> None:
        """
        TODO: add short docstring here
        """
        ckpt_dict = self.load_checkpoint(ckpt_path=params.model_checkpoint)

        # If we're resuming from checkpoint, the recipe's state should be updated before
        # initializing the training components. This ensures that the seed is correctly
        # propagated to the relevant components
        if self._resume_from_checkpoint:
            self._update_recipe_state(ckpt_dict)

        self._model = self._setup_model(
            model=params.model,
            lora_attn_modules=params.lora_attn_modules,
            lora_rank=params.lora_rank,
            lora_alpha=params.lora_alpha,
            enable_fsdp=params.enable_fsdp,
            enable_activation_checkpointing=params.enable_activation_checkpointing,
            base_model_state_dict=ckpt_dict[MODEL_KEY],
        )

        self._tokenizer = self._setup_tokenizer(
            tokenizer=params.tokenizer, tokenizer_checkpoint=params.tokenizer_checkpoint
        )

        self._optimizer = self._setup_optimizer(
            optimizer=params.optimizer, lr=params.lr, weight_decay=params.weight_decay
        )
        self._loss_fn = self._setup_loss(loss=params.loss)

        # sampler and dataloader depend on the tokenizer and loss_fn and should be
        # setup after all of these are setup
        self._sampler, self._dataloader = self._setup_data(
            dataset=params.dataset,
            shuffle=params.shuffle,
            batch_size=params.batch_size,
            train_on_input=params.train_on_input,
            use_clean=params.use_clean,
        )

        # training setup
        self._autocast = utils.get_autocast(self._dtype, self._device)
        self._grad_scaler = None
        if self._dtype == torch.float16:
            self._grad_scaler = utils.get_gradient_scaler(fsdp=params.enable_fsdp)
        else:
            self._grad_scaler = GradScaler(enabled=False)

        # Finally update the recipe state which can only be correctly set after all of the
        # other components have been initialized and updated.

        # Number of training steps in each epoch depends on the number of batches produced
        # by the dataloader and the max_steps_per_epoch param set by the user and is used
        # for logging and tracking training state. This should be computed after the dataloader
        # has been setup
        steps_per_epoch = len(self._dataloader)
        if self.max_steps_per_epoch and self.max_steps_per_epoch < len(
            self._dataloader
        ):
            steps_per_epoch = self.max_steps_per_epoch
            self.total_training_steps = self.epochs_run * steps_per_epoch

        # Learning rate scheduler can only be set up after number of steps
        # has been computed
        self._lr_scheduler = self._setup_lr_scheduler(
            lr_scheduler=params.lr_scheduler,
            num_warmup_steps=params.num_warmup_steps,
            num_training_steps=self.total_epochs * steps_per_epoch,
            last_epoch=self.total_training_steps - 1,
        )

    def load_checkpoint(self, ckpt_path: str):
        """
        Extract the checkpoint state from file and validate.
        """
        ckpt_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        utils.validate_checkpoint(ckpt_dict, self._resume_from_checkpoint)
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
        model: str,
        lora_attn_modules: List[str],
        lora_rank: int,
        lora_alpha: float,
        enable_fsdp: bool,
        enable_activation_checkpointing: bool,
        base_model_state_dict: Dict[str, Any],
    ) -> nn.Module:
        """
        This assumes FSDP and activation checkpointing are enabled by default.
        """
        model = models.get_model(
            model,
            device=self._device,
            lora_attn_modules=lora_attn_modules,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )

        adapter_params = get_adapter_params(model)
        set_trainable_params(model, adapter_params)

        # TODO: decide where to put this
        def lora_custom_auto_wrap_policy(module: nn.Module, recurse: bool, **kwargs):
            if recurse:
                return True
            # Wrap each transformer decoder layer
            if isinstance(module, modules.TransformerDecoderLayer):
                return True
            # Only lora_a and lora_b params require grads
            # They should be wrapped separately from frozen params
            if hasattr(module, "weight") and module.weight.requires_grad:
                return True
            return False

        model = (
            model
            if not enable_fsdp
            else utils.wrap_fsdp(
                model=model,
                device=self._device,
                dtype=self._dtype,
                strategy="FULL_SHARD",
                custom_wrap_policy=lora_custom_auto_wrap_policy,
            )
        )

        if enable_activation_checkpointing:
            utils.set_activation_checkpointing(
                model, auto_wrap_policy={modules.TransformerDecoderLayer}
            )

        missing, unexpected = model.load_state_dict(base_model_state_dict, strict=False)
        for x in missing:
            assert any([k in x for k in lora_attn_modules])
        assert not unexpected

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

    # TODO: support loading optimizer state for resuming training
    def _setup_optimizer(
        self, optimizer: str, lr: float, weight_decay: float
    ) -> Optimizer:
        optimizer = optim.get_optimizer(optimizer, self._model, lr, weight_decay)

        if self._is_rank_zero:
            log.info("Optimizer and loss are initialized.")
        return optimizer

    def _setup_lr_scheduler(
        self,
        lr_scheduler: str,
        num_warmup_steps: int,
        num_training_steps: int,
        last_epoch: int,
    ) -> Optimizer:
        lr_scheduler = lr_schedulers.get_lr_scheduler(
            lr_scheduler,
            self._optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch,
        )
        if self._is_rank_zero:
            log.info("Learning rate scheduler is initialized.")
        return lr_scheduler

    def _setup_loss(self, loss: str) -> nn.Module:
        loss_fn = losses.get_loss(loss)

        if self._is_rank_zero:
            log.info("Loss is initialized.")

        return loss_fn

    def _setup_data(
        self,
        dataset: str,
        shuffle: bool,
        batch_size: int,
        train_on_input: bool,
        use_clean: bool,
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

        if self._is_rank_zero:
            log.info("Dataset and Sampler are initialized.")

        return sampler, dataloader

    def save_checkpoint(self, epoch: int) -> None:
        """
        Checkpoint the state of the recipe. Currently this only includes checkpointing
        model weights and optimizer state.
        """
        output_loc = f"{self.output_dir}/ft_ckpts/model_{epoch}.ckpt"
        ckpt_dict = {"model": self._model}

        # if training is in-progress, checkpoint the optimizer state as well
        if epoch + 1 < self.total_epochs:
            ckpt_dict.update({"optimizer": self._optimizer})
        utils.save_checkpoint(ckpt_dict, output_loc)

        if self._is_rank_zero:
            log.info(
                msg=f"Model checkpoint of size {os.path.getsize(output_loc) >> 20} MB saved to {output_loc}"
            )

    def train(self) -> None:
        """
        The core training loop.
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
                self.total_training_steps += 1
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

                if (
                    self.total_training_steps % self._log_every_n_steps == 0
                    and self._is_rank_zero
                ):
                    self._metric_logger.log_dict(
                        {
                            "loss": loss.item(),
                            "lr": self._optimizer.param_groups[0]["lr"],
                            "gpu_resources": torch.cuda.memory_allocated(),
                        },
                        step=self.total_training_steps,  # Each step is unique, not limited to each epoch
                    )

                self._grad_scaler.scale(loss).backward()
                self._grad_scaler.step(self._optimizer)
                self._grad_scaler.update()
                self._lr_scheduler.step()

            self.epochs_run += 1
            self.save_checkpoint(epoch=curr_epoch)

    def cleanup(self) -> None:
        if self._is_rank_zero:
            self._metric_logger.close()


def recipe_main() -> None:
    """
    Entry point for the recipe.

    Configurable parameters are read in the following order:
        - Parameters specified in ``LoRAFinetuneParams``
        - Overwritten by Parameters specified in ``alpaca_llama2_lora_finetune.yaml``
        - Overwritten by arguments from the command-line using ``TuneArgumentParser``
    """
    parser = utils.TuneArgumentParser(
        description=LoRAFinetuneParams.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    args, _ = parser.parse_known_args()
    args = vars(args)
    recipe_params = LoRAFinetuneParams(**args)

    # Env variables set by torch run; only need to initialize process group
    init_process_group(backend="nccl")

    recipe = LoRAFinetuneRecipe(params=recipe_params)
    recipe.setup(params=recipe_params)
    recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())
