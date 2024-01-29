# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys

from functools import partial
from typing import Tuple

import torch

from torch import nn
from torch.cuda.amp import GradScaler
from torch.distributed import init_process_group
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler

from torchtune import datasets, losses, models, modules, optim, utils
from tqdm import tqdm

from recipes.args import create_full_finetune_args
from recipes.interfaces import FTRecipeInterface
from recipes.params import FullFinetuneParams


log = utils.get_logger("DEBUG")


class FullFinetuneRecipe(FTRecipeInterface):
    """
    Full finetuning recipe for dense transformer-based LLMs such as Llama2.

    This recipe supports:
        - FSDP and activation checkpointing. This is enabled by default and is NOT
            configurable.
        - Mixed precision training - fp32, fp16 and bf16 are supported.
        - Checkpointing of model weights and optionally of optimizer state (for checkpoints
            created during training).
        - Resume from checkpoint.
        - Logging to terminal. WandB and TensorBoard are currently not supported.

    Assumptions:
        - Training is launched with the Tune CLI (recommended) which uses TorchRun under the
            hood. Setting up the env variables is handled by TorchRun.
        - Training happens on CUDA (CPU training is not supported)
        - Checkpoints are ONLY saved at epoch boundaries. In case of failure, work done
            in ongoing epoch is lost.
        - Datasets are Map-style and data fits in memory (not streamed).

    TODO:
        - Checkpoint epoch and seed to ensure training restarts are correct.
    """

    def __init__(self, params: FullFinetuneParams) -> None:

        self.device = utils.get_device(device=params.device)
        self.dtype = utils.get_dtype(dtype=params.dtype)
        self.seed = utils.set_seed(seed=params.seed)
        self.output_dir = params.output_dir

        _, rank = utils.get_world_size_and_rank()
        self.is_rank_zero = rank == 0

        self.model = self._setup_model(model=params.model)
        self.tokenizer = self._setup_tokenizer(
            tokenizer=params.tokenizer, tokenizer_checkpoint=params.tokenizer_checkpoint
        )
        self.optimizer = self._setup_optimizer(optimizer=params.optimizer, lr=params.lr)
        self.loss_fn = self._setup_loss(loss=params.loss)

        # sampler and dataloader depend on the tokenizer and loss_fn and should be
        # setup after all of these are setup
        self.sampler, self.dataloader = self._setup_data(
            dataset=params.dataset,
            shuffle=params.shuffle,
            batch_size=params.batch_size,
        )

        self.total_epochs = params.epochs
        self.max_steps_per_epoch = params.max_steps_per_epoch

        # epochs_run tracks the number of epochs completed; this is updated by
        # `load_checkpoint` if `resume_from_checkpoint` is `True`.
        self.epochs_run = 0
        self.resume_from_checkpoint = params.resume_from_checkpoint

        self.autocast = utils.get_autocast(self.dtype, self.device)
        self.grad_scaler = None
        if self.dtype == torch.float16:
            self.grad_scaler = utils.get_gradient_scaler(fsdp=True)
        else:
            self.grad_scaler = GradScaler(enabled=False)

    def _setup_model(self, model: str) -> nn.Module:
        """
        This assumes FSDP and activation checkpointing are enabled by default.
        """
        model = models.get_model(model, device=self.device)
        model = utils.get_fsdp(
            model=model,
            device=self.device,
            dtype=self.dtype,
            strategy="FULL_SHARD",
            auto_wrap_policy={modules.TransformerDecoderLayer},
        )
        utils.set_activation_checkpointing(
            model, auto_wrap_policy={modules.TransformerDecoderLayer}
        )

        if self.is_rank_zero:
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

        if self.is_rank_zero:
            log.info("Tokenizer is initialized from file.")
        return tokenizer

    def _setup_optimizer(self, optimizer: str, lr: float) -> Optimizer:
        optimizer = optim.get_optimizer(optimizer, self.model, lr)

        if self.is_rank_zero:
            log.info("Optimizer and loss are initialized.")
        return optimizer

    def _setup_loss(self, loss: str) -> nn.Module:
        loss_fn = losses.get_loss(loss)

        if self.is_rank_zero:
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
        ds = datasets.get_dataset(dataset, split="train", tokenizer=self.tokenizer)
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
                padding_idx=self.tokenizer.pad_id,
                ignore_idx=self.loss_fn.ignore_index,  # TODO support loss without ignore_index
            ),
        )

        if self.is_rank_zero:
            log.info("Dataset and Sampler are initialized.")

        return sampler, dataloader

    def load_checkpoint(self, model_checkpoint: str) -> None:
        """
        Update the state of the recipe based on the checkpoint. This includes loading
        state dictionaries for the `model` and optionally for the `optimizer`. This also
        includes updating `epochs_run` and ensure `seed` is set correctly.
        """

        # model and optimizer are set only when resuming training
        ckpt_dict = utils.load_checkpoint_updated(
            ckpt_path=model_checkpoint,
            resume_from_checkpoint=self.resume_from_checkpoint,
            model=self.model if self.resume_from_checkpoint else None,
            optimizer=self.optimizer if self.resume_from_checkpoint else None,
        )

        self.model.load_state_dict(ckpt_dict["model"])

        # use this section to set all of the state which needs to be updated when
        # resuming from training
        if self.resume_from_checkpoint:
            self.optimizer.load_state_dict(ckpt_dict["optimizer"])

            # temporary place holder till we start tracking epoch and seed in
            # `save_checkpoint`
            self.epochs_run = (
                ckpt_dict["epoch"] if "epoch" in ckpt_dict else self.epochs_run
            )
            self.seed = (
                utils.set_seed(seed=ckpt_dict["seed"])
                if "seed" in ckpt_dict
                else self.seed
            )

        if self.is_rank_zero:
            log.info(
                msg=f"Loaded state of the recipe from checkpoint at {model_checkpoint}"
            )

    def save_checkpoint(self, epoch: int) -> None:
        """
        Checkpoint the state of the recipe. Currently this only includes checkpointing
        model weights and optimizer state.
        """
        output_loc = f"{self.output_dir}/model_{epoch}.ckpt"
        ckpt_dict = {"model": self.model}

        # if training is in-progress, checkpoint the optimizer state as well
        if epoch + 1 < self.total_epochs:
            ckpt_dict.update({"optimzer": self.optimizer})
        utils.save_checkpoint(ckpt_dict, output_loc)

        if self.is_rank_zero:
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
            self.sampler.set_epoch(curr_epoch)

            for idx, batch in enumerate(
                pbar := tqdm(self.dataloader, disable=not (rank == 0))
            ):
                if (
                    self.max_steps_per_epoch is not None
                    and idx == self.max_steps_per_epoch
                ):
                    break
                self.optimizer.zero_grad()

                input_ids, labels = batch
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)

                with self.autocast:
                    logits = self.model(input_ids)
                    # Shift so that tokens < n predict n
                    logits = logits[..., :-1, :].contiguous()
                    labels = labels[..., 1:].contiguous()
                    logits = logits.transpose(1, 2)
                    # Compute loss
                    loss = self.loss_fn(logits, labels)

                pbar.set_description(f"{curr_epoch+1}|{idx+1}|Loss: {loss.item()}")

                self.grad_scaler.scale(loss).backward()
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()

            self.save_checkpoint(epoch=curr_epoch)


def recipe_main() -> None:
    parser = utils.TuneArgumentParser(description="Fine-tune an LLM.")
    kwargs = vars(create_full_finetune_args(parser).parse_args())

    recipe_params = FullFinetuneParams(**kwargs)

    # Env variables set by torch run; only need to initialize process group
    init_process_group(backend="nccl")

    recipe = FullFinetune(params=recipe_params)
    recipe.load_checkpoint(model_checkpoint=recipe_params.model_checkpoint)
    recipe.train()


if __name__ == "__main__":
    sys.exit(recipe_main())
