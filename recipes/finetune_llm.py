# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import argparse
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
    params: FullFinetuneParams,
) -> None:
    """Training loop for fine-tuning an LLM on a provided dataset. Supports evals,
    checkpointing, and distributed training.

    Args:
        params (FullFinetuneParams): dataclass containing all args for recipe. See ``FullFinetuneParams`` for
             more details.

    Raises:
        ValueError: If ``cpu_offload`` is ``True`` but ``device`` is not ``cuda`` and <= 1 GPUs.
    """
    # ---- Initialize components ---- #
    distributed = utils.init_distributed()
    world_size, rank = utils.get_world_size_and_rank()

    logger = utils.get_logger("DEBUG")
    metric_logger = utils.get_metric_logger(
        metric_logger_type=params.metric_logger_type,
        project=params.project,
        log_dir=params.output_dir,
    )

    device = utils.get_device(params.device)
    dtype = utils.get_dtype(params.dtype)
    seed = utils.set_seed(params.seed)

    # ---- Setup model and load checkpoint ---- #
    tokenizer = models.get_tokenizer(params.tokenizer, path=params.tokenizer_checkpoint)
    logger.info(msg=f"Loaded tokenizer from {params.tokenizer_checkpoint}")

    # TODO: initialize models for distributed on meta or cpu device to avoid OOMs
    model = models.get_model(params.model, device=device)

    if params.cpu_offload and not distributed:
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
            cpu_offload=params.cpu_offload,
        )
    if params.enable_activation_checkpointing:
        utils.set_activation_checkpointing(
            model, auto_wrap_policy={modules.TransformerDecoderLayer}
        )

    # ---- Setup optimization functions ---- #
    opt = optim.get_optimizer(params.optimizer, model, params.lr)
    # Load model and possibly optimizer states
    if params.resume_from_checkpoint:
        ckpt_dict = load_checkpoint(params.model_checkpoint, model, opt)
        model.load_state_dict(ckpt_dict["model"])
        # Note: optimizer entry in dictionary is pre-transformed if using FSDP
        opt.load_state_dict(ckpt_dict["optimizer"])
        if rank == 0:
            logger.info(
                msg=f"Loaded checkpoint from previous finetune from {params.model_checkpoint}"
            )
    else:
        ckpt_dict = load_checkpoint(params.model_checkpoint, model)
        model.load_state_dict(ckpt_dict["model"])
        if rank == 0:
            logger.info(msg=f"Loaded pretrained model from {params.model_checkpoint}")

    # TODO add lr schedule option
    loss_fn = losses.get_loss(params.loss)

    autocast = utils.get_autocast(dtype, device)
    if dtype == torch.float16:
        grad_scaler = utils.get_gradient_scaler(distributed)
    else:
        grad_scaler = GradScaler(enabled=False)

    # ---- Load dataset, set up sampler, and dataloader ---- #
    ds = datasets.get_dataset(params.dataset, split="train", tokenizer=tokenizer)
    sampler = DistributedSampler(
        ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=params.shuffle,
        seed=0,
    )
    dataloader = DataLoader(
        dataset=ds,
        batch_size=params.batch_size,
        sampler=sampler,
        collate_fn=partial(
            utils.padded_collate,
            padding_idx=tokenizer.pad_id,
            ignore_idx=loss_fn.ignore_index,  # TODO support loss without ignore_index
        ),
    )
    logger.info(msg=f"Loaded dataset {params.dataset}")

    # ---- Train loop ---- #
    for epoch in range(params.epochs):
        sampler.set_epoch(epoch)  # distributed sampler requires set_epoch
        for idx, batch in enumerate(pbar := tqdm(dataloader, disable=not (rank == 0))):
            if (
                params.max_steps_per_epoch is not None
                and idx == params.max_steps_per_epoch
            ):
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
            if params.run_generation and idx % params.run_generation == 0:
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
        os.makedirs(params.output_dir, exist_ok=True)
        output_loc = f"{params.output_dir}/model_{epoch}.ckpt"
        ckpt_dict = {
            "model": model,
            "optimizer": opt,
        }
        if epoch == params.epochs - 1:
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
        description=FullFinetuneParams.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Get user-specified args from config and CLI and create params for recipe
    args, _ = parser.parse_known_args()
    args = vars(args)
    params = FullFinetuneParams(**args)

    logger = utils.get_logger("DEBUG")
    logger.info(msg=f"Running finetune_llm.py with parameters {params}")

    recipe(params)
