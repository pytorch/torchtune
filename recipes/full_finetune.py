# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

from functools import partial

import torch
from torch import nn

from torch.cuda.amp import GradScaler

from torch.distributed import init_process_group
from torch.utils.data import DataLoader, DistributedSampler
from torchtune import datasets, losses, models, modules, optim, utils

from tqdm import tqdm

from recipes.args import create_full_finetune_args
from recipes.interfaces import FTRecipeInterface


class FullFinetune(FTRecipeInterface):
    """
    Full finetuning recipe for dense trasnformer-based LLMs such as Llama2.

    This consists of three stages:
        - Initialize the recipe through config or args.create_full_finetune_args
        - Train the model and save checkpoints appropriately
        - Cleanup at the end of training

    This recipe supports:
        - FSDP and activation checkpointing. This is enabled by default and is NOT
            configurable.
        - Mixed precision training - fp32, fp16 and bf16 are supported.
        - Checkpointing of model weights and optionally of optimizer state (for checkpoints
            created during training).
        - Resume from checkpoint.

    Assumptions:
        - Training is launched with the Tune CLI (recommended) which uses TorchRun under the
            hood. Setting up the env variables is handled by TorchRun.
        - Training happens on CUDA (CPU training is not supported)
        - Checkpoints are ONLY saved at epoch boundaries. In case of failure, work done
            in ongoing epoch is lost.
        - Datasets are Map-style and data fits in memory (not streamed).

    TODO:
        - The recipe is littered with "rank == 0" checks to prevent log spew. Logging needs to be
            converted into a wrapper class which can handle this more gracefully.
    """

    def __init__(
        self,
        device: str,
        dtype: str,
        seed: int,
        model: str,
        model_checkpoint: str,
        tokenizer: str,
        tokenizer_checkpoint: str,
        dataset: str,
        shuffle: bool,
        batch_size: int,
        optimizer: str,
        loss: str,
        lr: float,
        output_dir: str,
        epochs: int,
        max_steps_per_epoch: int,
        resume_from_checkpoint: bool,
    ) -> None:

        self.setup_environment(device=device, dtype=dtype, seed=seed, output_dir=output_dir)
        self.setup_model(model=model)
        self.setup_tokenizer(
            tokenizer=tokenizer, tokenizer_checkpoint=tokenizer_checkpoint
        )
        self.setup_optimizer_and_loss(optimizer=optimizer, lr=lr, loss=loss)
        self.setup_data(dataset=dataset, shuffle=shuffle, batch_size=batch_size)
        self.setup_training_params(epochs=epochs, max_steps_per_epoch=max_steps_per_epoch)
        self.load_checkpoint(
            model_checkpoint=model_checkpoint,
            resume_from_checkpoint=resume_from_checkpoint,
        )


    def setup_environment(self, device: str, dtype: str, seed: int, output_dir: str) -> None:
        """
        Initialize the environment - setting up distributed, loggers, devices and seed.
        """

        # Env variables set by torch run; only need to initialize process group
        init_process_group(backend="nccl")

        self.logger = utils.get_logger("DEBUG")
        self.output_dir = output_dir
        self.device = utils.get_device(device)
        self.dtype = utils.get_dtype(dtype)
        self.seed = utils.set_seed(seed)


    def setup_model(self, model: str) -> None:
        """
        This assumes FSDP and activation checkpointing are enabled by default.
        """
        model = models.get_model(model, device=self.device)
        self.model = utils.get_fsdp(
            model=model,
            device=self.device,
            dtype=self.dtype,
            strategy="FULL_SHARD",
            auto_wrap_policy={modules.TransformerDecoderLayer},
        )
        utils.set_activation_checkpointing(
            self.model, auto_wrap_policy={modules.TransformerDecoderLayer}
        )

        _, rank = utils.get_world_size_and_rank()
        if rank == 0:
            self.logger.info(
                "Model is initialized. FSDP and Activation Checkpointing are enabled."
            )


    def setup_tokenizer(self, tokenizer: str, tokenizer_checkpoint: str) -> None:
        """
        Unlike ```setup_model```, this takes in the checkpoint and loads the sentencepiece
        tokenizer model. This is related to how the tokenizer is implemented and should
        change in a future iteration.
        """
        self.tokenizer = models.get_tokenizer(tokenizer, path=tokenizer_checkpoint)

        _, rank = utils.get_world_size_and_rank()
        if rank == 0:
            self.logger.info("Tokenizer is initialized from file.")


    def setup_optimizer_and_loss(self, optimizer:str, lr: float, loss: str) -> None:
        self.optimizer = optim.get_optimizer(optimizer, self.model, lr)
        self.loss_fn = losses.get_loss(loss)

        _, rank = utils.get_world_size_and_rank()
        if rank == 0:
            self.logger.info("Optimizer and loss are initialized.")


    def setup_data(self, dataset: str, shuffle: bool, batch_size: int) -> None:
        """
        All data related setup happens here. Currently this recipe only supports the
        DistributedSamplers with Map-style Datasets which fit into memory. Other samplers,
        iterable datasets and streaming datasets are not supported.
        """
        world_size, rank = utils.get_world_size_and_rank()
        ds = datasets.get_dataset(dataset, split="train", tokenizer=self.tokenizer)
        self.sampler = DistributedSampler(
            ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=0,
        )
        self.dataloader = DataLoader(
            dataset=ds,
            batch_size=batch_size,
            sampler=self.sampler,
            collate_fn=partial(
                utils.padded_collate,
                padding_idx=self.tokenizer.pad_id,
                ignore_idx=self.loss_fn.ignore_index,  # TODO support loss without ignore_index
            ),
        )

        if rank == 0:
            self.logger.info("Dataset and Sampler are initialized.")


    def setup_training_params(self, epochs: int, max_steps_per_epoch: int) -> None:
        """
        This sets the training parameters for the recipe.
        """
        self.epochs = epochs
        self.max_steps_per_epoch = max_steps_per_epoch

        # parameter which determines where training begins; if needed, this is updated
        # when the state of the recipe is loaded from checkpoint in ```load_checkpoint`
        self.epochs_run = 0

        self.autocast = utils.get_autocast(self.dtype, self.device)
        self. grad_scaler = None
        if self.dtype == torch.float16:
            self. grad_scaler = utils.get_gradient_scaler(fsdp=True)
        else:
            self. grad_scaler = GradScaler(enabled=False)

        _, rank = utils.get_world_size_and_rank()
        if rank == 0:
            self.logger.info("Training parameters are initialized.")


    def load_checkpoint(self, model_checkpoint: str, resume_from_checkpoint: bool) -> None:
        """
        Loading the state for the recipe from a previous checkpoint happens here.
        """

        # TODO: Update this comment
        # load_checkpoint takes in the model and optimizer, but only uses these
        # in case we're resuming from checkpoint

        ckpt_dict = utils.load_checkpoint(model_checkpoint, self.model)
        self.model.load_state_dict(ckpt_dict["model"])

        # # use this section to set all of the state which needs to be updated when
        # # resuming from training
        # if resume_from_checkpoint:
        #     self.optimizer.load_state_dict(ckpt_dict["optimizer"])

        #     # temporary place holder till we start tracking epoch in
        #     # utils.save_checkpoint
        #     self.epochs_run = ckpt_dict["epoch"] if "epoch" in ckpt_dict else 0

        _, rank = utils.get_world_size_and_rank()
        if rank == 0:
            self.logger.info(msg=f"Loaded pretrained model from {model_checkpoint}")

    def save_checkpoint(self, epoch, rank) -> None:
        """
        Checkpoint the state of the recipe. Currently this only includes checkpointing
        model weights.
        """
        output_loc = f"{self.output_dir}/model_{epoch}.ckpt"
        ckpt_dict = {
            "model": self.model,
            "optimizer": self.optimizer,
        }
        utils.save_checkpoint(ckpt_dict, output_loc)

        _, rank = utils.get_world_size_and_rank()
        if rank == 0:
            self.logger.info(
                msg=f"{rank}: Model checkpoint of size {os.path.getsize(output_loc) >> 20} MB saved to {output_loc}"
            )

    def train(self) -> None:
        """
        The core training loop.

            - Appropriately sets autocase based on precision
            - Gradient Scaler is appropirately selected
            - Metrics are logged to the appropriate metric logger (WandB, Tensorboard)
        """
        _, rank = utils.get_world_size_and_rank()

        # self.epochs_run should be non-zero when we're resuming from a checkpoint
        for curr_epoch in range(self.epochs_run, self.epochs):

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

            self.save_checkpoint(epoch=curr_epoch, rank=rank)


if __name__ == "__main__":

    parser = utils.TuneArgumentParser(description="Fine-tune an LLM.")
    kwargs = vars(create_full_finetune_args(parser).parse_args())

    recipe = FullFinetune(**kwargs)
    recipe.train()
    recipe.cleanup()
