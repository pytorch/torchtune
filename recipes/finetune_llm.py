# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import dataclasses
import os
from functools import partial

import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, DistributedSampler

from torchtune import datasets, losses, models, modules, optim, utils
from torchtune.utils.checkpoint import load_checkpoint, save_checkpoint
from torchtune.utils.generation import generate_from_prompt
from tqdm import tqdm

from recipes.params import FullFinetuneParams


def recipe(
    *,
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
    epochs: int,
    optimizer: str,
    loss: str,
    lr: float,
    enable_activation_checkpointing: bool,
    output_dir: str,
    run_generation: int,
    max_steps_per_epoch: int,
    metric_logger_type: str,
    project: str,
    resume_from_checkpoint: bool,
    cpu_offload: bool,
) -> None:
    """Training loop for fine-tuning an LLM on a provided dataset. Supports evals,
    checkpointing, and distributed training.

    Args:
        device (str): Device to use for training. Options are "cpu" and "cuda"
        dtype (str): Data type to use for training.
        seed (int): Random seed to use for training.
        model (str): String specifying model architecture to fine-tune. See ``torchtune.models.get_model`` for options.
        model_checkpoint (str): Local path to load model checkpoint from.
        tokenizer (str): String specifying tokenizer to use. See ``torchtune.models.get_tokenizer`` for options.
        tokenizer_checkpoint (str): Local path to load tokenizer checkpoint from.
        dataset (str): String specifying dataset to use. See ``torchtune.datasets.get_dataset`` for options.
            Currently, only predefined datasets in library are supported.
        shuffle (bool): Whether to shuffle dataset.
        batch_size (int): Batch size to use for training.
        epochs (int): Number of epochs to train for.
        optimizer (str): String specifying optimizer to use. See ``torchtune.optim.get_optimizer`` for options.
        loss (str): String specifying loss function to use. See ``torchtune.losses.get_loss`` for options.
        lr (float): Learning rate to use for optimizer.
        enable_activation_checkpointing (bool): Whether to use activation checkpointing.
        output_dir (str): Local path to save checkpoints and logs to.
        run_generation (int): Run eval on a prompt every ``run_generation`` steps. Set to 0 to disable.
        max_steps_per_epoch (int): Maximum number of steps to take per epoch.
        metric_logger_type (str): String specifying metric logger to use. See ``torchtune.utils.get_metric_logger``
            for options.
        project (str): Project name to use for logging. Used by ``WandBLogger``.
        resume_from_checkpoint (bool): Whether to resume fine-tuning from a previous checkpoint.
        cpu_offload (bool): Whether to offload model to CPU.

    Raises:
        ValueError: If ``cpu_offload`` is ``True`` but ``device`` is not ``cuda`` and <= 1 GPUs.
    """
    # ---- Initialize components ---- #
    distributed = utils.init_distributed()
    world_size, rank = utils.get_world_size_and_rank()

    logger = utils.get_logger("DEBUG")
    metric_logger = utils.get_metric_logger(
        metric_logger_type=metric_logger_type, project=project, log_dir=output_dir
    )

    device = utils.get_device(device)
    dtype = utils.get_dtype(dtype)
    seed = utils.set_seed(seed)

    # ---- Setup model and load checkpoint ---- #
    tokenizer = models.get_tokenizer(tokenizer, path=tokenizer_checkpoint)
    logger.info(msg=f"Loaded tokenizer from {tokenizer_checkpoint}")

    # TODO: initialize models for distributed on meta or cpu device to avoid OOMs
    model = models.get_model(model, device=device)

    if cpu_offload and not distributed:
        raise ValueError(
            "CPU offload is only supported with FSDP in a distributed setting."
            "Please launch in a distributed setting. If you do not wish to use > 1 GPU,"
            "use ``tune --nnodes 1 --nproc_per_node 1 ...``. FSDP will not shard"
            "any parameters."
        )

    if distributed:  # Use FSDP model for distributed training
        model = utils.get_fsdp(
            model=model,
            device=device,
            dtype=dtype,
            strategy="FULL_SHARD",
            auto_wrap_policy={modules.TransformerDecoderLayer},
            cpu_offload=cpu_offload,
        )
    if enable_activation_checkpointing:
        utils.set_activation_checkpointing(
            model, auto_wrap_policy={modules.TransformerDecoderLayer}
        )

    # ---- Setup optimization functions ---- #
    opt = optim.get_optimizer(optimizer, model, lr)
    # Load model and possibly optimizer states
    if resume_from_checkpoint:
        ckpt_dict = load_checkpoint(model_checkpoint, model, opt)
        model.load_state_dict(ckpt_dict["model"])
        # Note: optimizer entry in dictionary is pre-transformed if using FSDP
        opt.load_state_dict(ckpt_dict["optimizer"])
        if rank == 0:
            logger.info(
                msg=f"Loaded checkpoint from previous finetune from {model_checkpoint}"
            )
    else:
        ckpt_dict = load_checkpoint(model_checkpoint, model)
        model.load_state_dict(ckpt_dict["model"])
        if rank == 0:
            logger.info(msg=f"Loaded pretrained model from {model_checkpoint}")

    # TODO add lr schedule option
    loss_fn = losses.get_loss(loss)

    autocast = utils.get_autocast(dtype, device)
    if dtype == torch.float16:
        grad_scaler = utils.get_gradient_scaler(distributed)
    else:
        grad_scaler = GradScaler(enabled=False)

    # ---- Load dataset, set up sampler, and dataloader ---- #
    ds = datasets.get_dataset(dataset, split="train", tokenizer=tokenizer)
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
            padding_idx=tokenizer.pad_id,
            ignore_idx=loss_fn.ignore_index,  # TODO support loss without ignore_index
        ),
    )
    logger.info(msg=f"Loaded dataset {dataset}")

    # ---- Train loop ---- #
    for epoch in range(epochs):
        sampler.set_epoch(epoch)  # distributed sampler requires set_epoch
        for idx, batch in enumerate(pbar := tqdm(dataloader, disable=not (rank == 0))):
            if max_steps_per_epoch is not None and idx == max_steps_per_epoch:
                break
            opt.zero_grad()

            input_ids, labels = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            with autocast:
                logits = model(input_ids)
                # Shift so that tokens < n predict n
                logits = logits[..., :-1, :].contiguous()
                labels = labels[..., 1:].contiguous()
                logits = logits.transpose(1, 2)
                # Compute loss
                loss = loss_fn(logits, labels)

            pbar.set_description(f"{epoch+1}|{idx+1}|Loss: {loss.item()}")

            # Log metrics at each step
            # If no metric logger is specified, this is a no-op
            if rank == 0:
                metric_logger.log_dict(
                    {
                        "loss": loss.item(),
                        "lr": opt.param_groups[0]["lr"],
                        "gpu_resources": torch.cuda.memory_allocated(),
                    },
                    step=epoch * len(dataloader)
                    + idx,  # Each step is unique, not limited to each epoch
                )

            grad_scaler.scale(loss).backward()
            grad_scaler.step(opt)
            grad_scaler.update()

            # --- TODO TEMPORARY EVAL Code ---- #
            if run_generation and idx % run_generation == 0:
                # Log a sample generation for the instruction.
                # Just using a hardcoded prompt for now
                prompt = (
                    "Below is an instruction that describes a task, paired with an input that provides further context. "
                    "Write a response that appropriately completes the request.\n\n### Instruction:\nCreate a classification task "
                    "by clustering the given list of items.\n\n### Input:\nApples, oranges, bananas, strawberries, pineapples\n\n"
                    "### Response:"
                )
                generation_str, decoded_tokens = generate_from_prompt(
                    prompt=prompt, tokenizer=tokenizer, decoder=model
                )
                if rank == 0:
                    logger.info(f"Generation tokens: {decoded_tokens}")
                    logger.info(f"Generation: {generation_str}")
            # --- TODO TEMPORARY EVAL Code Ends ---- #

        # ---- Save checkpoint at end of each epoch (to be changed later) ---- #
        os.makedirs(output_dir, exist_ok=True)
        output_loc = f"{output_dir}/model_{epoch}.ckpt"
        ckpt_dict = {
            "model": model,
            "optimizer": opt,
        }
        if epoch == epochs - 1:
            # Don't save optimizer state when producing final checkpoint to reduce checkpoint file size.
            ckpt_dict.pop("optimizer")
        if rank == 0:
            logger.info(msg=f"Saving model checkpoint to {output_loc}")
        save_checkpoint(ckpt_dict, output_loc)
        if rank == 0:
            logger.info(
                msg=f"Model checkpoint of size {os.path.getsize(output_loc) >> 20} MB saved to {output_loc}"
            )

    metric_logger.close()


if __name__ == "__main__":
    parser = utils.TuneArgumentParser(
        description=recipe.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Get user-specified args from config and CLI and create params for recipe
    args, _ = parser.parse_known_args()
    args = vars(args)
    params = FullFinetuneParams(**args)

    logger = utils.get_logger("DEBUG")
    logger.info(msg=f"Running finetune_llm.py with parameters {params}")

    # Temporary hack as we migrate to new recipe
    params_dict = dataclasses.asdict(params)
    del params_dict["enable_fsdp"]
    recipe(**params_dict)
