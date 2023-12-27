# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from functools import partial
from pathlib import Path
from typing import Callable

import torch
from torch.optim.optimizer import Optimizer
from torchtune.datasets import get_dataset, list_datasets
from torchtune.models import get_model, get_tokenizer, list_models, list_tokenizers

from torchtune.trainer import ReproducibleDataLoader
from torchtune.utils import ArgumentParser
from torchtune.utils.batch_pad_sequence import batch_pad_to_longest_seq

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

    tokenizer = get_tokenizer(kwargs["tokenizer"], path=kwargs["tokenizer_checkpoint"])
    logger(msg=f"Loaded tokenizer from {kwargs['tokenizer_checkpoint']}")

    device = kwargs["device"]
    model = get_model(kwargs["model"], device, vocab_size=tokenizer.vocab_size)
    model.load_state_dict(torch.load(kwargs["model_checkpoint"], weights_only=True))
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
        for batch in (pbar := tqdm(dataloader)):
            opt.zero_grad()

            input_ids, labels = batch
            input_ids = input_ids.to(device)

            logits = model(input_ids)

            logits = logits.cpu()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, tokenizer.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Compute loss
            loss = loss_fn(shift_logits, shift_labels)
            pbar.set_description(f"{epoch+1}| Loss: {loss.item()}")

            loss.backward()
            opt.step()

        # Save checkpoint at end of each epoch (to be changed later)
        output_loc = f"{kwargs['output_dir']}/model_{epoch}.ckpt"
        Path(output_loc).mkdir(parents=True, exist_ok=True)
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
            msg=f"Model checkpoint of size {os.path.get_size(output_loc)} bytes saved to {output_loc}"
        )


if __name__ == "__main__":
    """Return an argument parser for the script. Add all arguments here."""
    parser = ArgumentParser(description="Fine-tune an LLM model.")

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
        "--epochs", type=int, default=100, help="Number of epochs for fine-tuning"
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

    kwargs = vars(parser.parse_args())
    recipe(kwargs)
