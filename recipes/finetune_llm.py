# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
from functools import partial
from typing import Callable

import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, DistributedSampler

from torchtune.datasets import get_dataset, list_datasets
from torchtune.models import get_model, get_tokenizer, list_models, list_tokenizers
from torchtune.modules import TransformerDecoderLayer
from torchtune.utils import TuneArgumentParser
from torchtune.utils.batch_pad_sequence import batch_pad_to_longest_seq
from torchtune.utils.env import get_world_size_and_rank, init_from_env, seed
from torchtune.utils.generation import generate_from_prompt
from torchtune.utils.precision import (
    get_autocast_manager,
    get_grad_scaler,
    get_supported_dtypes,
)
from tqdm import tqdm


def get_optimizer(model: torch.nn.Module, optimizer: str, lr: float) -> Optimizer:
    return getattr(torch.optim, optimizer)(model.parameters(), lr=lr)


def get_loss(loss_fn: str) -> Callable:
    return getattr(torch.nn, loss_fn)()


def get_logger():
    import logging

    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.DEBUG)
    return logger.info


def recipe(kwargs):
    # ---- Initialize components ---- #
    logger = get_logger()

    # ---- Initialize seed ---- #
    world_size, rank = get_world_size_and_rank()
    if world_size > 1 and "seed" not in kwargs:
        raise ValueError("Must set seed during distributed training. ")

    base_seed = kwargs["seed"] or torch.empty((), dtype=torch.int32).random_().item()

    # Ensure that seed is different per rank (and its dataloader workers)
    seed(base_seed + rank)

    # ---- Initialize distributed process group ---- #
    device = init_from_env(device_type=kwargs["device"])
    # TODO: only supporting devices specified as "cpu", "cuda", or "cuda:n" currently
    device_type = (
        kwargs["device"]
        if kwargs["device"] in ("cpu", "cuda")
        else kwargs["device"].split(":")[0]
    )
    tokenizer = get_tokenizer(kwargs["tokenizer"], path=kwargs["tokenizer_checkpoint"])
    logger(msg=f"Loaded tokenizer from {kwargs['tokenizer_checkpoint']}")

    autocast_precision = kwargs.get("autocast_precision", None)
    autocast_mgr = get_autocast_manager(
        device_type=device_type, precision=autocast_precision
    )
    grad_scaler = get_grad_scaler(autocast_precision, fsdp=kwargs["fsdp"])

    # When using fsdp, init on meta device to avoid OOM
    model = get_model(
        kwargs["model"],
        "meta" if kwargs["fsdp"] else device,
    )

    if kwargs["fsdp"] or kwargs["activation_checkpointing"]:
        auto_wrap_policy = ModuleWrapPolicy(
            {TransformerDecoderLayer}
        )  # TODO: remove model specific components
    if kwargs["fsdp"]:
        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            device_id=device,
            param_init_fn=lambda m: m.to_empty(device=device, recurse=False),
        )
    if kwargs["activation_checkpointing"]:
        apply_activation_checkpointing(
            model,
            check_fn=lambda mod: isinstance(
                mod, TransformerDecoderLayer
            ),  # TODO: remove model specific components
            auto_wrap_policy=auto_wrap_policy,
        )

    loaded_ckpt = torch.load(
        kwargs["model_checkpoint"], map_location="cpu", weights_only=True
    )
    model.load_state_dict(loaded_ckpt)
    logger(msg=f"Loaded model from {kwargs['model_checkpoint']}")

    opt = get_optimizer(model, kwargs["optimizer"], kwargs["lr"])
    # TODO add lr schedule option
    loss_fn = get_loss(kwargs["loss"])

    # ---- Load dataset, set up sampler, and dataloader ---- #
    dataset = get_dataset(kwargs["dataset"], split="train", tokenizer=tokenizer)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=kwargs["shuffle"],
        seed=base_seed,
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=kwargs["batch_size"],
        sampler=sampler,
        collate_fn=partial(
            batch_pad_to_longest_seq,
            input_padding_idx=tokenizer.pad_id,
            label_padding_idx=loss_fn.ignore_index,  # TODO support loss without ignore_index
        ),
    )
    logger(msg=f"Loaded dataset {kwargs['dataset']}")

    # ---- Train loop ---- #
    for epoch in range(kwargs["epochs"]):
        # Need to set the epoch for changing sample ordering in each epoch
        dataloader.sampler.set_epoch(epoch)
        for idx, batch in enumerate(pbar := tqdm(dataloader)):
            max_steps_per_epoch = kwargs.get("max_steps_per_epoch", None)
            if max_steps_per_epoch is not None and idx == max_steps_per_epoch:
                break
            opt.zero_grad()

            input_ids, labels = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            # Note: context manager for autocast is only applied in forward pass.
            # see https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html#adding-torch-autocast
            # for more details.
            with autocast_mgr:
                logits = model(input_ids)
                # Shift so that tokens < n predict n
                logits = logits[..., :-1, :].contiguous()
                labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                # shift_logits = shift_logits.view(-1, tokenizer.vocab_size)
                # shift_labels = shift_labels.view(-1)
                logits = logits.transpose(1, 2)
                # Compute loss
                loss = loss_fn(logits, labels)

            pbar.set_description(
                f"{epoch+1}|{idx+1}|Loss: {loss.item()}"
            )  # TODO: add terminal logger

            if grad_scaler:
                grad_scaler.scale(loss).backward()
                grad_scaler.step(opt)
                grad_scaler.update()
            else:
                loss.backward()
                opt.step()

            run_generation = kwargs.get("run_generation", None)
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
                if (
                    not torch.distributed.is_initialized()
                    or torch.distributed.get_rank() == 0
                ):
                    logger(f"Generation tokens: {decoded_tokens}")
                    logger(f"Generation: {generation_str}")

        # Save checkpoint at end of each epoch (to be changed later)
        os.makedirs(kwargs["output_dir"], exist_ok=True)
        output_loc = f"{kwargs['output_dir']}/model_{epoch}.ckpt"
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "loss": loss.mean().item(),
            },
            output_loc,
        )
        logger(
            msg=f"Model checkpoint of size {os.path.getsize(output_loc) >> 20}MB saved to {output_loc}"
        )


if __name__ == "__main__":
    parser = TuneArgumentParser(description="Fine-tune an LLM model.")

    # Dataset and DataLoader arguments
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list_datasets(),
        help="Dataset name.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="""
            Seed for dataset shuffling order and setting trainer and dataloader
            workers random number generator state. Using same seed value will
            provide the same ordering and transforms of samples across runs.
            """,
    )
    parser.add_argument("--shuffle", help="Shuffle dataset.", default=True)

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        choices=list_models(),
        help="Model to finetune.",
    )
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        choices=list_tokenizers(),
        help="Model tokenizer.",
    )
    parser.add_argument(
        "--tokenizer-checkpoint",
        type=str,
        help="Path to tokenizer checkpoint.",
    )

    # Fine-tuning arguments
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Batch size for fine-tuning."
    )
    parser.add_argument(
        "--lr", type=float, default=2e-5, help="Learning rate for fine-tuning."
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of epochs for fine-tuning"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help="Optimizer to use for fine-tuning, please consult torch.optim docs for a list of available optimizers",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="CrossEntropyLoss",
        choices=["CrossEntropyLoss"],
        help="Loss to use for fine-tuning.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/tmp/finetune-llm",
        help="Directory in which to save checkpoints.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="`cuda` or `cpu`",
    )
    parser.add_argument(
        "--fsdp",
        type=bool,
        default=False,
        help="Train the model with distributed fully sharded data parallel (FSDP) strategy.",
    )
    parser.add_argument(
        "--activation-checkpointing",
        type=bool,
        default=False,
        help="Train the model with activation checkpointing.",
    )
    parser.add_argument(
        "--run-generation",
        type=int,
        default=None,
        help="Run a dummy alpaca generation every n iterations.",
    )
    parser.add_argument(
        "--max-steps-per-epoch",
        type=int,
        default=None,
        help="Max number of steps per epoch for faster dev/testing. Default is to finetune through the full dataset.",
    )
    parser.add_argument(
        "--autocast-precision",
        type=str,
        choices=get_supported_dtypes(),
        default=None,
        help=f"""Low precision used for CUDA automatic mixed precision.
            If specified, must be one of {get_supported_dtypes()}.
        """,
    )

    kwargs = vars(parser.parse_args())
    sys.exit(recipe(kwargs))
