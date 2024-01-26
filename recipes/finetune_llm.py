# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import os
from functools import partial

import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DistributedSampler

from torchtune import datasets, losses, models, modules, optim, utils
from torchtune.utils import CheckpointableDataLoader
from torchtune.utils.checkpoint import load_checkpoint, save_checkpoint
from torchtune.utils.generation import generate_from_prompt
from tqdm import tqdm


def recipe(
    device,
    dtype,
    seed,
    model,
    model_checkpoint,
    tokenizer,
    tokenizer_checkpoint,
    dataset,
    shuffle,
    batch_size,
    epochs,
    optimizer,
    loss,
    lr,
    activation_checkpointing,
    output_dir,
    run_generation,
    max_steps_per_epoch,
    metric_logger_type,
    project,
    resume_from_previous_checkpoint,
    checkpoint_interval,
    cpu_offload,
):
    # ---- Initialize components ---- #
    distributed = utils.init_distributed()
    world_size, rank = utils.get_world_size_and_rank()

    logger = utils.get_logger("DEBUG")
    metric_logger = utils.get_metric_logger(
        metric_logger_type=metric_logger_type, project=project, log_dir=output_dir
    )

    device = utils.get_device(device)
    dtype = utils.get_dtype(dtype)

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
    if activation_checkpointing:
        utils.set_activation_checkpointing(
            model, auto_wrap_policy={modules.TransformerDecoderLayer}
        )

    # ---- Load dataset, set up sampler, and dataloader ---- #
    ds = datasets.get_dataset(dataset, split="train", tokenizer=tokenizer)
    sampler = DistributedSampler(
        ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        seed=0,
    )
    dataloader = CheckpointableDataLoader(
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

    # ---- Setup optimization functions ---- #
    opt = optim.get_optimizer(optimizer, model, lr)
    # Restore from previous state or start from pretrained model checkpoint.
    if resume_from_previous_checkpoint:
        ckpt_dict = load_checkpoint(model_checkpoint, model, opt)
        model.load_state_dict(ckpt_dict["model"])
        # Note: optimizer entry in dictionary is pre-transformed if using FSDP
        opt.load_state_dict(ckpt_dict["optimizer"])
        # Restore seed.
        restored_seed = ckpt_dict["seed"]
        if seed != restored_seed:
            logger.warning(
                f"Seed {seed} does not match seed {restored_seed} in checkpoint. "
                f"Overwriting seed to {restored_seed}."
                "Start the run with the checkpointed seed to avoid this warning."
            )
            seed = restored_seed
        # Restore epoch.
        start_epoch = ckpt_dict["epoch"]
        # Restore dataloader
        dataloader.load_state_dict(ckpt_dict["dataloader"])
        if rank == 0:
            logger.info(
                msg=f"Loaded checkpoint from previous finetune from {model_checkpoint}"
            )
    else:
        ckpt_dict = load_checkpoint(model_checkpoint, model)
        model.load_state_dict(ckpt_dict["model"])
        start_epoch = 0
        if rank == 0:
            logger.info(msg=f"Loaded pretrained model from {model_checkpoint}")

    seed = utils.set_seed(seed)

    # TODO add lr schedule option
    loss_fn = losses.get_loss(loss)

    autocast = utils.get_autocast(dtype, device)
    if dtype == torch.float16:
        grad_scaler = utils.get_gradient_scaler(distributed)
    else:
        grad_scaler = GradScaler(enabled=False)

    # Make a directory to store checkpoints.
    os.makedirs(output_dir, exist_ok=True)
    # ---- Train loop ---- #
    training_step = 0
    for epoch in range(start_epoch, epochs):
        sampler.set_epoch(epoch)  # distributed sampler requires set_epoch
        for idx, batch in enumerate(pbar := tqdm(dataloader, disable=not (rank == 0))):
            training_step += 1
            if max_steps_per_epoch is not None and idx == max_steps_per_epoch:
                break
            opt.zero_grad()

            input_ids, labels = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            # Debug code for dataloader restore determinism.
            with open(
                f"/tmp/samples_{torch.distributed.get_rank()}.text", mode="w+"
            ) as f:
                f.write(
                    "\n".join(
                        [
                            input_ids.cpu().numpy().tolist(),
                            labels.cpu().numpy().tolist(),
                        ]
                    )
                )
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

            if (
                checkpoint_interval is not None
                and training_step % checkpoint_interval == 0
            ):
                # Save checkpoint
                output_loc = f"{output_dir}/model_{training_step}.ckpt"
                save_checkpoint(
                    {
                        "model": model,
                        "optimizer": opt,
                        "dataloader": dataloader,
                        "epoch": epoch,
                        "seed": seed,
                    },
                    output_loc,
                )

        # ---- Save checkpoint at end of each epoch (to be changed later) ---- #
        output_loc = f"{output_dir}/model_{epoch}.ckpt"
        ckpt_dict = {
            "model": model,
            "optimizer": opt,
            "dataloader": dataloader,
            "epoch": epoch,
            "seed": seed,
        }
        if epoch == epochs - 1:
            # Don't save optimizer state when producing final checkpoint to reduce checkpoint file size.
            ckpt_dict.pop("optimizer")
        save_checkpoint(ckpt_dict, output_loc)

    metric_logger.close()


if __name__ == "__main__":
    parser = utils.TuneArgumentParser(description="Fine-tune an LLM.")

    # Dataset and DataLoader arguments
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
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--activation-checkpointing",
        action="store_true",
        default=False,
        help="Train the model with activation checkpointing.",
    )
    group.add_argument(
        "--no-activation-checkpointing",
        dest="activation_checkpointing",
        action="store_false",
        help="Don't train the model with activation checkpointing.",
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
        "--dtype",
        type=str,
        choices=utils.list_dtypes(),
        default=None,
        help="Tensor dtype used for finetuning, lower precision types result in mixed precision training.",
    )
    parser.add_argument(
        "--metric-logger-type",
        type=str,
        choices=utils.list_metric_loggers(),
        help="Metric logger platform to use. E.g. Weights & Biases, Tensorboard, to disk, or just plain stdout.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Project name for WandB metric logger.",
    )
    parser.add_argument(
        "--resume-from-previous-checkpoint",
        help="""Resume from a previous finetune checkpoint. Note that the previous
            checkpoints must have been taken with `torchtune.utils.checkpoint.save_checkpoint` utility. If this flag
            is used, note that --model-checkpoint flag will be used as the path to the previous finetune's checkpoint.
            """,
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--checkpoint-interval",
        help="""Save a checkpoint (model, optimizer, dataloader, etc) every N iterations. The training state will be saved such that
            deterministic training can be resumed from the checkpoint. Note that this flag is inpendent of
            --checkpoint-at-epoch-end, and either or neither flag can be used.
            """,
        default=None,
        type=int,
    )
    parser.add_argument(
        "--cpu-offload",
        action="store_true",
        default=False,
        help="Offload parameters and gradients to CPU when not involved in computation. Optimizer step runs on CPU.",
    )

    kwargs = vars(parser.parse_args())
    recipe(**kwargs)
