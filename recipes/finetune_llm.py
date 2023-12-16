# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import contextlib
import os
from functools import partial
from typing import Callable

import torch
from torch.cuda.amp import GradScaler
from torch.optim.optimizer import Optimizer
from torchtune.datasets import get_dataset
from torchtune.models.llama2.tokenizer import Tokenizer
from torchtune.models.llama2.transformer import TransformerDecoder

from torchtune.trainer import ReproducibleDataLoader
from torchtune.utils.batch_pad_sequence import (
    _DEFAULT_INPUT_PADDING_IDX,
    _DEFAULT_LABEL_PADDING_IDX,
    batch_pad_to_longest_seq,
)
from torchtune.utils.precision import (
    _get_grad_scaler,
    _LOW_PRECISION_STR_TYPES,
    _type_str_to_dtype,
)

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
    return getattr(torch.nn.functional, loss_fn)


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

    tokenizer = Tokenizer.from_file(args.tokenizer_checkpoint)
    # Original tokenizer has no pad_id, which causes indexing errors when batch training
    tokenizer.pad_id = _DEFAULT_INPUT_PADDING_IDX
    logger(msg=f"Loaded tokenizer from {args.tokenizer_checkpoint}")

    if args.device != "cuda" and args.autocast_precision in _LOW_PRECISION_STR_TYPES:
        raise ValueError(
            f"Cannot specify autocast precision {args.autocast_precision} without using CUDA."
        )

    device = args.device
    # NOTE: we don't use autocast w/dtype=fp32 when autocast precision is not specified, as this could cause
    # unexpected upcasts when running w/FSDP mixed precision.
    autocast_mgr = (
        torch.autocast(
            device_type=device, dtype=_type_str_to_dtype[args.autocast_precision]
        )
        if args.autocast_precision is not None
        else contextlib.nullcontext()
    )
    grad_scaler: GradScaler = _get_grad_scaler(args.autocast_precision)
    with torch.device(device):
        model = TransformerDecoder(
            vocab_size=tokenizer.vocab_size,
            num_layers=32,
            num_heads=32,
            embed_dim=4096,
            max_seq_len=2048,
            norm_eps=1e-5,
        )
    model.load_state_dict(torch.load(args.model_checkpoint))
    logger(msg=f"Loaded model from {args.model_checkpoint}")

    opt = get_optimizer(model, args.optimizer, args.lr)
    loss_fn = get_loss(args.loss_fn)

    # ---- Load dataset ---- #
    dataset = get_dataset(args.dataset, split="train", tokenizer=tokenizer)
    dataloader = ReproducibleDataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        collate_fn=partial(
            batch_pad_to_longest_seq,
            input_padding_idx=_DEFAULT_INPUT_PADDING_IDX,
            label_padding_idx=_DEFAULT_LABEL_PADDING_IDX,
        ),
        seed=args.dataloader_seed,
    )

    # ---- Train loop ---- #
    for epoch in tqdm(range(args.epochs)):
        for i, batch in enumerate(dataloader):
            opt.zero_grad()

            input_ids, labels = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            logits = model(input_ids)

            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, tokenizer.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Compute loss
            loss = loss_fn(shift_logits, shift_labels)
            logger(msg=f"Loss @ step {i} in epoch {epoch}: {loss}")
            if grad_scaler:
                grad_scaler.scale(loss).backward()
                grad_scaler.step(opt)
                grad_scaler.update()
            else:
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
