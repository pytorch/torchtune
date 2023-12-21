# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from functools import partial
from typing import Callable

import torch
from torch.optim.optimizer import Optimizer
from torchtune.datasets import get_dataset, list_datasets
from torchtune.models import get_model, get_tokenizer, list_models, list_tokenizers

from torchtune.trainer import ReproducibleDataLoader
from torchtune.utils.batch_pad_sequence import batch_pad_to_longest_seq

from tqdm import tqdm


def get_argparser():
    """Return an argument parser for the script. Add all arguments here."""
    parser = argparse.ArgumentParser(description="Fine-tune an LLM model.")
    # Dataset and DataLoader arguments
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=list_datasets(),
        help="Dataset name.",
    )
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
        "--model",
        type=str,
        choices=list_models(),
        required=True,
        help="Model to finetune",
    )
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        choices=list_tokenizers(),
        required=True,
        help="Model tokenizer.",
    )
    parser.add_argument(
        "--tokenizer-checkpoint",
        type=str,
        required=True,
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
        help="Optimizer to use for fine-tuning, please consult torch.optim Docs for a list of available optimizers",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="CrossEntropyLoss",
        choices=["CrossEntropyLoss"],
        help="Loss to use for fine-tuning",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/tmp/llama-finetune",
        help="Directory in which to save checkpoints during fine-tuning",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="`cuda` or `cpu`",
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


def main():
    # ---- Parse arguments ---- #
    parser = get_argparser()
    args = parser.parse_args()

    # ---- Initialize components ---- #
    logger = get_logger()

    tokenizer = get_tokenizer(args.tokenizer, path=args.tokenizer_checkpoint)
    logger(msg=f"Loaded tokenizer from {args.tokenizer_checkpoint}")

    device = args.device
<<<<<<< HEAD
    model = get_model(args.model, device, vocab_size=tokenizer.vocab_size)
    model.load_state_dict(torch.load(args.model_checkpoint, weights_only=True))
    logger(msg=f"Loaded model from {args.model_checkpoint}")

    opt = get_optimizer(model, args.optimizer, args.lr)
    # TODO add lr schedule option
    loss_fn = get_loss(args.loss)

    # ---- Load dataset ---- #
    dataset = get_dataset(args.dataset, split="train", tokenizer=tokenizer)
    dataloader = ReproducibleDataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        collate_fn=partial(
            batch_pad_to_longest_seq,
            input_padding_idx=tokenizer.pad_id,
            label_padding_idx=loss_fn.ignore_index,  # TODO support loss without ignore_index
        ),
        seed=args.dataloader_seed,
    )
    logger(msg=f"Loaded dataset {args.dataset}")

    # ---- Train loop ---- #
    for epoch in range(args.epochs):
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
        output_loc = f"{args.output_dir}/model_{epoch}.ckpt"
        torch.save(model.state_dict(), output_loc)
        logger(
            msg=f"Model checkpoint of size {os.path.get_size(output_loc)} bytes saved to {output_loc}"
        )


if __name__ == "__main__":
    main()
