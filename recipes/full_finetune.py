# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

from functools import partial

import torch

from torch.cuda.amp import GradScaler

from torch.distributed import init_process_group
from torch.utils.data import DataLoader, DistributedSampler
from torchtune import datasets, losses, models, modules, optim, utils

from tqdm import tqdm

from recipes.args import create_full_finetune_args
from recipes.recipe_interface import Recipe


class FullFinetune(Recipe):
    """
    Full finetuning recipe.

    This recipe currently supports or assumes the following:

    Support:
        - FSDP and activation checkpointing. This is enabled by default and is NOT
            configurable.
        - Mixed precision training - fp32, fp16 and bf16 are supported.
        - Resuming from checkpoint is NOT currently supported.
        - Checkpoints are ONLY saved at epoch boundaries. In case of failure, work done
            in ongoing epoch is lost.

    Assumptions:
        - Training is launched with the Tune CLI which uses TorchRun under the hood. Setting
            up the env variables are handled by TorchRun.
        - Training happens on CUDA.

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
        metric_logger_type: str,
        project: str,
    ) -> None:

        self.setup_environment(
            device=device,
            dtype=dtype,
            seed=seed,
            metric_logger_type=metric_logger_type,
            project=project,
            output_dir=output_dir,
        )
        self.setup_model(model=model)
        self.setup_tokenizer(
            tokenizer=tokenizer, tokenizer_checkpoint=tokenizer_checkpoint
        )
        self.setup_optimizer_and_loss(optimizer=optimizer, learning_rate=lr, loss=loss)
        self.setup_data(dataset=dataset, shuffle=shuffle, batch_size=batch_size)
        self.load_checkpoint(model_checkpoint=model_checkpoint)

    def setup_environment(
        self,
        device,
        dtype,
        seed,
        metric_logger_type,
        project,
        output_dir,
    ) -> None:
        """
        Initialize the environment - setting up distributed, loggers, devices and seed.
        """

        # Env variables set by torch run; only need to initialize process group
        init_process_group(backend="nccl")

        self.logger = utils.get_logger("DEBUG")
        self.metric_logger = utils.get_metric_logger(
            metric_logger_type=metric_logger_type, project=project, log_dir=output_dir
        )
        self.output_dir = output_dir

        self.device = utils.get_device(device)
        self.dtype = utils.get_dtype(dtype)
        self.seed = utils.set_seed(seed)

    def setup_model(self, model) -> None:
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

    def setup_tokenizer(self, tokenizer, tokenizer_checkpoint) -> None:
        """
        Unlike ```setup_model```, this takes in the checkpoint and loads the sentencepiece
        tokenizer model. This is related to how the tokenizer is implemented and should
        change in a future iteration.
        """
        self.tokenizer = models.get_tokenizer(tokenizer, path=tokenizer_checkpoint)

        _, rank = utils.get_world_size_and_rank()
        if rank == 0:
            self.logger.info("Tokenizer is initialized from file.")

    def setup_optimizer_and_loss(self, optimizer, learning_rate, loss) -> None:
        self.optimizer = optim.get_optimizer(optimizer, self.model, learning_rate)
        self.loss_fn = losses.get_loss(loss)

        _, rank = utils.get_world_size_and_rank()
        if rank == 0:
            self.logger.info("Optimizer and loss are initialized.")

    def setup_data(self, dataset, shuffle, batch_size) -> None:
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

    def load_checkpoint(self, model_checkpoint) -> None:
        """
        Loading the state for the recipe from a previous checkpoint happens here.
        Currently this does not support resuming training. As a result, we only
        load the model checkpoint.
        """
        ckpt_dict = utils.load_checkpoint(model_checkpoint, self.model)
        self.model.load_state_dict(ckpt_dict["model"])

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

    def train(self, epochs, max_steps_per_epoch) -> None:
        """
        The core training loop.

            - Appropriately sets autocase based on precision
            - Gradient Scaler is appropirately selected
            - Metrics are logged to the appropriate metric logger (WandB, Tensorboard)
        """
        autocast = utils.get_autocast(self.dtype, self.device)
        if self.dtype == torch.float16:
            grad_scaler = utils.get_gradient_scaler(fsdp=True)
        else:
            grad_scaler = GradScaler(enabled=False)

        _, rank = utils.get_world_size_and_rank()

        for epoch in range(epochs):
            self.sampler.set_epoch(epoch)
            for idx, batch in enumerate(
                pbar := tqdm(self.dataloader, disable=not (rank == 0))
            ):
                if max_steps_per_epoch is not None and idx == max_steps_per_epoch:
                    break
                self.optimizer.zero_grad()

                input_ids, labels = batch
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)

                with autocast:
                    logits = self.model(input_ids)
                    # Shift so that tokens < n predict n
                    logits = logits[..., :-1, :].contiguous()
                    labels = labels[..., 1:].contiguous()
                    logits = logits.transpose(1, 2)
                    # Compute loss
                    loss = self.loss_fn(logits, labels)

                pbar.set_description(f"{epoch+1}|{idx+1}|Loss: {loss.item()}")

                # Log metrics at each step
                # If no metric logger is specified, this is a no-op
                if rank == 0:
                    self.metric_logger.log_dict(
                        {
                            "loss": loss.item(),
                            "lr": self.optimizer.param_groups[0]["lr"],
                            "gpu_resources": torch.cuda.memory_allocated(),
                        },
                        step=epoch * len(self.dataloader)
                        + idx,  # Each step is unique, not limited to each epoch
                    )

                grad_scaler.scale(loss).backward()
                grad_scaler.step(self.optimizer)
                grad_scaler.update()

            self.save_checkpoint(epoch=epoch, rank=rank)

    def cleanup(self) -> None:
        self.metric_logger.close()


if __name__ == "__main__":

    parser = utils.TuneArgumentParser(description="Fine-tune an LLM.")
    kwargs = vars(create_full_finetune_args(parser).parse_args())

    # remove arguments not needed for init
    epochs = kwargs.pop("epochs")
    max_steps_per_epoch = kwargs.pop("max_steps_per_epoch")

    recipe = FullFinetune(**kwargs)
    recipe.train(epochs=epochs, max_steps_per_epoch=max_steps_per_epoch)
    recipe.cleanup()
