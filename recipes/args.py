# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse

from torchtune import datasets, models, utils


def create_full_finetune_args(parser) -> argparse.ArgumentParser:

    # Dataset arguments
    parser.add_argument(
        "--dataset",
        type=str,
        choices=datasets.list_datasets(),
        help="Dataset name.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="""
            Seed for setting trainer and dataloader workers random number generator state. Using same seed value will
            provide the same transforms of samples across runs.
            """,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--shuffle", action="store_true", help="Shuffle dataset.", default=True
    )
    group.add_argument(
        "--no-shuffle",
        dest="shuffle",
        action="store_false",
        help="Don't shuffle dataset.",
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        choices=models.list_models(),
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
        choices=models.list_tokenizers(),
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
        help="Directory in which to save checkpoints."
        "If using a metric logger like Tensorboard, this dir will also contain those logs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="`cuda` or `cpu`",
    )
    parser.add_argument(
        "--max-steps-per-epoch",
        type=int,
        default=None,
        help="Max number of steps per epoch for faster dev/testing. Default is to finetune through the full dataset.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=utils.list_dtypes(),
        default=None,
        help="Tensor dtype used for finetuning, lower precision types result in mixed precision training.",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--resume-from-checkpoint",
        action="store_true",
        help="Resume from checkpoint",
        default=True,
    )
    group.add_argument(
        "--no-resume",
        dest="resume_from_checkpoint",
        action="store_false",
        help="Don't resume.",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--enable-fsdp",
        action="store_true",
        help="Enable FSDP",
        default=True,
    )
    group.add_argument(
        "--disable-fsdp",
        dest="enable_fsdp",
        action="store_false",
        help="Disable FSDP",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--enable-activation-checkpointing",
        action="store_true",
        help="Enable Activation Checkpointing",
        default=True,
    )
    group.add_argument(
        "--disable-activation-checkpointing",
        dest="enable_activation_checkpointing",
        action="store_false",
        help="Disable Activation Checkpointing",
    )
    return parser


def create_lora_finetune_args(parser) -> argparse.ArgumentParser:

    parser = create_full_finetune_args(parser)

    parser.add_argument(
        "--lora-attn-modules",
        nargs="+",
        help="List of modules to apply LoRA to in self-attention",
        default=["q_proj", "v_proj"],
    )

    parser.add_argument(
        "--metric-logger",
        type=str,
        default="stdout",
        help="metric logger",
    )

    parser.add_argument(
        "--project",
        type=str,
        default="lora-debug",
        help="Project for wandb logging",
    )

    parser.add_argument(
        "--num-warmup-steps",
        type=int,
        help="Number of warmup steps in LR scheduler",
        default=100,
    )

    parser.add_argument("--log-interval", type=int, help="Log every n steps", default=5)

    return parser
