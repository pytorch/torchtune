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
from torchtune.utils.precision import (
    _get_autocast_manager,
    _get_grad_scaler,
    _LOW_PRECISION_STR_TYPES,
)

from torchtune.datasets import get_dataset, list_datasets
from torchtune.models import get_model, get_tokenizer, list_models, list_tokenizers
from torchtune.models.llama2.transformer import TransformerDecoderLayer
from torchtune.trainer import ReproducibleDataLoader
from torchtune.utils import TuneArgumentParser
from torchtune.utils.batch_pad_sequence import batch_pad_to_longest_seq
from torchtune.utils.env import init_from_env
from tqdm import tqdm


def get_argparser():
    """Return an argument parser for the script. Add all arguments here."""
    parser = argparse.ArgumentParser(
        description="Fine-tune a native PyTorch LLaMA model."
    )
    # Dataset and DataLoader arguments
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name.")
    parser.add_argument(
        "--dataloader-seed",
        type=int,
        required=False,
        default=None,
        help="""
            Seed for dataset shuffling order and multiprocessing
            worker base_seed. Same seed value will provide the
            same ordering and transforms of samples across runs.
            """,
    )
    parser.add_argument("-shuffle", help="Shuffle dataset.", default=True)
    # Model arguments
    parser.add_argument(
        "--tokenizer-checkpoint",
        type=str,
        required=True,
        help="Path to SentencePiece tokenizer.",
    )
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        required=True,
        help="Path to native PyTorch LLaMA model checkpoint.",
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
        choices=["AdamW", "SGD"],
        help="Optimizer to use for fine-tuning.",
    )
    parser.add_argument(
        "--loss-fn",
        type=str,
        default="cross_entropy",
        choices=["cross_entropy"],
        help="Loss function to use for fine-tuning",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/tmp/llama-finetune",
        help="Directory in which to save checkpoints during fine-tuning.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
        help="`cuda` or `cpu`",
    )
    parser.add_argument(
        "--autocast-precision",
        type=str,
        choices=_LOW_PRECISION_STR_TYPES,
        default=None,
        help="""Low precision used for CUDA automatic mixed precision.
            If specified, must be one of fp16 or bf16. --device argument
            must be CUDA for this to apply.
        """,
    )

    return parser


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

    # ---- Initialize distributed process group ---- #
    device = init_from_env(device_type=kwargs["device"])

    tokenizer = get_tokenizer(kwargs["tokenizer"], path=kwargs["tokenizer_checkpoint"])
    logger(msg=f"Loaded tokenizer from {kwargs['tokenizer_checkpoint']}")
    autocast_precision = kwargs["autocast_precision"]
    if device != "cuda" and autocast_precision in _LOW_PRECISION_STR_TYPES:
        raise ValueError(
            f"Specifying autocast precision {autocast_precision} without using CUDA is not yet implemented."
        )

    autocast_mgr = _get_autocast_manager(
        device_type=device, precision=autocast_precision
    )
    grad_scaler = _get_grad_scaler(autocast_precision)

    # When using fsdp, init on meta device to avoid OOM
    model = get_model(
        kwargs["model"],
        "meta" if kwargs["fsdp"] else kwargs["device"],
        vocab_size=tokenizer.vocab_size,
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

    # ---- Load dataset ---- #
    dataset = get_dataset(kwargs["dataset"], split="train", tokenizer=tokenizer)
    dataloader = ReproducibleDataLoader(
        dataset=dataset,
        batch_size=kwargs["batch_size"],
        shuffle=kwargs["shuffle"],
        collate_fn=partial(
            batch_pad_to_longest_seq,
            input_padding_idx=tokenizer.pad_id,
            label_padding_idx=loss_fn.ignore_index,  # TODO support loss without ignore_index
        ),
        seed=kwargs["dataloader_seed"],
    )
    logger(msg=f"Loaded dataset {kwargs['dataset']}")

    # ---- Train loop ---- #
    for epoch in range(kwargs["epochs"]):
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
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                shift_logits = shift_logits.view(-1, tokenizer.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Compute loss
                loss = loss_fn(shift_logits, shift_labels)

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
        "--dataloader-seed",
        type=int,
        default=None,
        help="""
            Seed for dataset shuffling order and multiprocessing
            worker base_seed. Same seed value will provide the
            same ordering and transforms of samples across runs.
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
        "--max-steps-per-epoch",
        type=int,
        default=None,
        help="Max number of steps per epoch for faster dev/testing. Default is to finetune through the full dataset.",
    )

    kwargs = vars(parser.parse_args())
    sys.exit(recipe(kwargs))
