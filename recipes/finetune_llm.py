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
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
from torchtune.utils.generation import GenerationUtils
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.optim.optimizer import Optimizer

from torchtune.datasets import get_dataset, list_datasets
from torchtune.models import get_model, get_tokenizer, list_models, list_tokenizers
from torchtune.models.llama2.transformer import TransformerDecoderLayer
from torchtune.trainer import ReproducibleDataLoader
from torchtune.utils import TuneArgumentParser
from torchtune.utils.batch_pad_sequence import batch_pad_to_longest_seq
from torchtune.utils.env import init_from_env
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

    # ---- Initialize distributed process group ---- #
    device = init_from_env(device_type=kwargs["device"])

    tokenizer = get_tokenizer(kwargs["tokenizer"], path=kwargs["tokenizer_checkpoint"])
    logger(msg=f"Loaded tokenizer from {kwargs['tokenizer_checkpoint']}")

    autocast_precision = kwargs.get("autocast_precision", None)
    autocast_mgr = get_autocast_manager(
        device_type=kwargs["device"], precision=autocast_precision
    )
    grad_scaler = get_grad_scaler(autocast_precision, fsdp=kwargs["fsdp"])

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
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16
            ),
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
    ### BEGIN HACK: remove rope prefix from loaded checkpoint ---- #
    keys_to_del = [k for k in loaded_ckpt.keys() if "rope" in k]
    for k in keys_to_del:
        del loaded_ckpt[k]
    ### END HACK: remove rope prefix from loaded checkpoint ---- #
    missing_keys, unexpected_keys = model.load_state_dict(loaded_ckpt, strict=False)
    logger(
        msg=f"Loaded model from {kwargs['model_checkpoint']}. Keys missing = {missing_keys}, unexpected = {unexpected_keys}"
    )

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

    mean_loss = None
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
                # logits are (batch_size, sequence_length, num_classes), transpose to
                # (batch_size, num_classes, sequence_length)
                logits = logits.transpose(1, 2)
                loss = loss_fn(logits, labels)

            if grad_scaler:
                grad_scaler.scale(loss).backward()
                grad_scaler.step(opt)
                grad_scaler.update()
            else:
                loss.backward()
                opt.step()

            if mean_loss is None:
                mean_loss = loss.item()
            else:
                mean_loss = (mean_loss * idx + loss) / (idx + 1)
            pbar.set_description(
                f"{epoch+1}|{idx+1}|Loss: {mean_loss}"
            )  # TODO: add terminal logger

            if idx % 50 == 0:
                # Log a sample generation for the instruction.
                if not torch.distributed.is_initialized() or dist.get_rank() == 0: print(f"RV: Running generation at index {idx}", flush=True)
                response_tag = "\n\n### Response:\n"
                prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nCreate a classification task by clustering the given list of items.\n\n### Input:\nApples, oranges, bananas, strawberries, pineapples\n\n### Response:"
                prompt_tokens = [tokenizer.encode(prompt, add_eos=False)]
                with torch.no_grad():
                    generations_no_kv_cache, _ = GenerationUtils(
                        decoder_lm=model,
                        eos_id=tokenizer.eos_id,
                        pad_id=tokenizer.pad_id,
                    ).generate(
                        prompt_tokens=prompt_tokens,
                        incremental_decode=False,
                        min_gen_len=1,
                        max_gen_len=256,
                        top_k=3,
                        device=torch.cuda.current_device(),
                    )

                    gens = generations_no_kv_cache.tolist()[0]
                    print(f"RV: got gen tokens {gens}", flush=True)
                    gens = gens[:gens.index(2)] if 2 in gens else gens
                    if not torch.distributed.is_initialized() or dist.get_rank() == 0: print(f"RV: generation {tokenizer.decode(gens)}", flush=True)

        # Save checkpoint at end of each epoch (to be changed later)
        os.makedirs(kwargs["output_dir"], exist_ok=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            output_loc = (
                f"{kwargs['output_dir']}/model_{epoch}_rank{dist.get_rank()}.ckpt"
            )
            torch.save(model.state_dict(), output_loc)
            # # distributed_state_dict
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
