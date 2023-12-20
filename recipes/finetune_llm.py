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
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
    CheckpointImpl,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.optim.optimizer import Optimizer
from torchtune.datasets import get_dataset
from torchtune.models.llama2.tokenizer import Tokenizer
from torchtune.models.llama2.transformer import (
    TransformerDecoder,
    TransformerDecoderLayer,
)

from torchtune.trainer import ReproducibleDataLoader
from torchtune.utils.batch_pad_sequence import (
    _DEFAULT_INPUT_PADDING_IDX,
    _DEFAULT_LABEL_PADDING_IDX,
    batch_pad_to_longest_seq,
)
from tqdm import tqdm


def get_argparser():
    """Return an argument parser for the script. Add all arguments here."""
    parser = argparse.ArgumentParser(
        description="Fine-tune a native PyTorch Llama model."
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
        help="Path to native PyTorch Llama model checkpoint.",
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
        choices=["AdamW"],
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

    # ---- Initialize distributed process group ---- #
    torch.distributed.init_process_group("nccl")

    tokenizer = Tokenizer.from_file(args.tokenizer_checkpoint)
    # Original tokenizer has no pad_id, which causes indexing errors when batch training
    tokenizer.pad_id = _DEFAULT_INPUT_PADDING_IDX
    logger(msg=f"Loaded tokenizer from {args.tokenizer_checkpoint}")

    device = torch.device(args.device)
    with torch.device("meta"):
        model = TransformerDecoder(
            vocab_size=tokenizer.vocab_size,
            num_layers=32,
            num_heads=32,
            embed_dim=4096,
            max_seq_len=2048,
            norm_eps=1e-5,
        )

    device_id = torch.distributed.get_rank() % torch.cuda.device_count()
    print(f"Rank {torch.distributed.get_rank()} setting CUDA device {device_id}")
    # This appears to be needed to avoid torch.load putting all tensors on
    # GPU 0. Might also be able to use map_location for that.
    torch.cuda.set_device(device_id)
    print(f"RV: torch.cuda.current_device() gives {torch.cuda.current_device()}, device count is {torch.cuda.device_count()}", flush=True)
    #torch.set_default_device(torch.cuda.current_device())
    model = FSDP(
        model,
        auto_wrap_policy=ModuleWrapPolicy({TransformerDecoderLayer}),
        device_id=torch.cuda.current_device(),
        param_init_fn=lambda m: m.to_empty(device=torch.cuda.current_device(), recurse=False),
    )
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=partial(
            checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
        ),
        check_fn=lambda mod: isinstance(mod, TransformerDecoderLayer),
    )

    loaded_ckpt = torch.load(args.model_checkpoint, map_location='cpu')
    model.load_state_dict(loaded_ckpt)
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
            input_ids = input_ids.to(torch.cuda.current_device())

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
            logger(msg=f"Loss @ step {i} in epoch {epoch}: {loss}")

            loss.backward()
            opt.step()

        # Save checkpoint at end of each epoch (to be changed later)
        output_loc = f"{args.output_dir}/model_{epoch}.ckpt"
        torch.save(model.state_dict(), output_loc)
        logger(
            msg=f"Model checkpoint of size {os.path.get_size(output_loc)} bytes saved to {output_loc}"
        )
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
