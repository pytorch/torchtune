# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
import sys
import time

from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union
from warnings import warn
import re
import torch
from omegaconf import DictConfig, ListConfig

from torch import nn
from torch.distributed import destroy_process_group, init_process_group

from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler, Subset
from torchtune import config, modules, training, utils
from torchtune.config._utils import _get_component_from_path
from torchtune.data import padded_collate_packed
from torchtune.datasets import ConcatDataset
from torchtune.recipe_interfaces import FTRecipeInterface
from torchtune.training import DummyProfiler, PROFILER_KEY
from torchtune.training.activations import apply_selective_activation_checkpointing
from torchtune.training.lr_schedulers import get_lr
from omegaconf import OmegaConf
import pprint
import os
import re

import random
from tqdm import tqdm
import torch.nn.functional as F
from full_finetune_distributed import FullFinetuneRecipeDistributed

log = utils.get_logger("DEBUG")


class FullFinetuneRecipeDistributedPlus(FullFinetuneRecipeDistributed):
    """
    Full finetuning recipe for dense transformer-based LLMs such as Llama2. This recipe supports
    distributed training and can be run on a single node (1 to 8 GPUs).

    Features:
        - FSDP. Supported using PyTorch's FSDP APIs. CPU offload of parameters, gradients, and optimizer states
            is supported via ``fsdp_cpu_offload``. Resharding of parameters after the forward pass is
            done by default (corresponding to FULL_SHARD sharding strategy), but can be disabled by setting the config
            ``fsdp_reshard_after_forward`` to False (this corresponds to SHARD_GRAD_OP sharding strategy).
            DDP is currently not supported. Training on CPU is not supported.

        - Activation Checkpointing. This can be controlled using the ``enable_activation_checkpointing``
            flag. Activation checkpointing helps reduce the memory footprint since we no longer keep
            activations in memory and instead recompute them during the backward pass. This is especially
            helpful for larger batch sizes when you're memory constrained. But these savings in memory
            come at the cost of training performance. In most cases training can slow-down quite a bit as
            a result of this activation recomputation.

        - Activation Offloading. This can be controlled using the ``enable_activation_offloading``
            flag. Activation offloading is a technique similar to activations checkpointing that helps
            reduce the memory footprint to prevent OOMs on CUDA and enable bigger batches. Where activations
            checkpointing drops the activation in the forward to recompute it later in the backward,
            activations offloading will drop the activation in the forward to the CPU and bring it
            back during the backward pass. As always, there is a tradeoff--these savings in memory can
            come at the cost of training performance and CPU resources. To recover some runtime cost,
            we've added an option to enable offloading on a different stream to permit overlapping with
            the computation. This option is currently only available on PyTorch 2.5 or later and will
            be enabled by default if an acceptable torch version is found. Activation offloading can be
            used in conjunction with activation checkpointing.

        - Precision. Full fp32 and bf16 training are supported. Precision is controlled using the ``dtype``
            flag. When ``dtype=bf16``, all activations, gradients and optimizer states are in bfloat16. In
            most cases this should halve the memory footprint of full precision (fp32) training, without
            loss in model quality (will depend on the model, training data and other settings). For
            GPUs which do not support bfloat16, we fall back to fp32. Mixed precision training and fp16
            precision are currently not supported.

        - Gradient Accumulation. You can simulate larger batch sizes by accumulating gradients. This is
            controlled using the ``gradient_accumulation_steps`` flag.

                Total Batch Size = batch_size * number of GPUs * gradient accumulation steps.

            For example: with batch_size=1, nproc_per_node=2 and gradient_accumulation_steps=32 we get a
            total batch size of 64.

            Gradient accumulation is especially useful when you are memory constrained. In this case,
            accumulating gradients might give you better training speed than enabling activation
            checkpointing.

        - Checkpointing. Model weights are checkpointed both at the end of each epoch and at the end of
            training. Optimizer state and recipe state (seed, total_epochs, number of epochs run etc) are
            only saved at the end of a given epoch and used in case of resuming training.

            Resuming training is controlled by the ``resume_from_checkpoint`` flag. Mid-epoch checkpointing is
            currently not supported.

            For more details on the checkpointer, please take a look at
            our checkpointer deepdive (https://pytorch.org/torchtune/main/deep_dives/checkpointer.html).

        - Logging. Terminal, Disk, WandB and TensorBoard are all supported.

        - Gradient Clipping. Gradient clipping is supported using the ``clip_grad_norm`` flag. By default,
            ``clip_grad_norm`` is set to ``None``. If you only want to log the grad norm, you can set
            ``clip_grad_norm='inf'``.

    For a full list of example configs for this recipe, run ``tune ls`` on the command line. Each config
    has example commands for how to kick-off training.

    Args:
        cfg (DictConfig): OmegaConf object parsed from yaml file

    Raises:
        ValueError: If ``dtype`` is set to fp16.
        RuntimeError: If ``dtype`` is set to bf16 and the hardware does not support bf16.
        RuntimeError: If ``left_pad_sequence`` is set as the data collator.
        RuntimeError: If ``enable_activation_offloading`` is True and device is not CUDA.
        RuntimeError: If ``enable_activation_offloading`` is True and ``enable_activation_checkpointing`` is False.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg=cfg)

    def precompute_reference_logprobs(self, labels=None):
        """
        Precompute reference logprobs for the reference model.
        This is used for the REINFORCE method with importance sampling.
        """
        # assert we are using chucnked cross entropy
        if self._loss_fn.__class__.__name__ != "CEWithChunkedOutputLoss":
            raise RuntimeError(
                "Precomputing reference logprobs is only supported with CEWithChunkedOutputLoss."
                "Behaviour added by us to handle reference model"
            )
        utils.log_rank_zero(log, "Precomputing reference logprobs...")
        self._model.eval()
        self.reference_logprobs_cache = {}
        with torch.no_grad():
            for batch_idx, batch in enumerate(
                tqdm(self._dataloader, desc="Precomputing reference logprobs")
            ):
                
                # Move batch to device
                batch = {k: v.to(self._device) for k, v in batch.items()}
                # Get a unique identifier for the batch (hash of input tokens)
                batch_key = hash(tuple(batch["tokens"].flatten().tolist()))

                # Extract labels before forward pass
                labels = batch.pop("labels")
                reward = batch.pop("reward", None)

                # Forward pass through reference model
                ref_logits = self._model(**batch)

                # Shift labels as in training loop
                shifted_labels = torch.hstack(
                    (labels[..., 1:], self.ignore_labels_cache[: labels.shape[0]])
                )

                # Process reference logits - chunking them like in the original code
                logits_chunks = [
                    logit_chunk.reshape(-1, logit_chunk.size(-1))
                    for logit_chunk in ref_logits
                ]

                # Chunk labels too
                labels_chunks = [
                    target_chunk.reshape(-1)
                    for target_chunk in shifted_labels.chunk(
                        self._loss_fn.num_output_chunks, dim=1
                    )
                ]

                # Pre-gather log probabilities for each chunk
                ref_gathered_logprobs = []
                for i, (logits_chunk, labels_chunk) in enumerate(
                    zip(logits_chunks, labels_chunks)
                ):
                    # Convert to log probabilities
                    log_probs = F.log_softmax(logits_chunk.float(), dim=-1)

                    # Perform the gather operation now to save computation later
                    vocab_size = log_probs.size(-1)
                    valid_indices = labels_chunk.clamp(0, vocab_size - 1)
                    gathered_logprobs = torch.gather(
                        log_probs, dim=-1, index=valid_indices.unsqueeze(-1)
                    ).squeeze(-1)

                    ref_gathered_logprobs.append(gathered_logprobs.to("cpu"))

                # Store the pre-gathered values
                self.reference_logprobs_cache[batch_key] = ref_gathered_logprobs

                # Put labels back for subsequent processing
                batch["labels"] = labels

        utils.log_rank_zero(
            log,
            f"Precomputed logprobs for {len(self.reference_logprobs_cache)} batches",
        )
        # set model back to train
        self._model.train()

    def _calculate_importance_ratio(
        self, new_log_ps, old_log_ps, epsilon_low, epsilon_high, indices
    ):
        """
        Calculate the importance ratio for the new and old log probabilities.
        """
        # Calculate importance ratio without tracking gradients
        with torch.no_grad():
            # Use gather to extract the log probabilities for each token index
            # First, make sure indices are valid (within vocabulary size)
            vocab_size = new_log_ps.size(-1)
            # Create a mask for tokens that are the ignore index
            ignore_mask = indices == self._loss_fn.ignore_index

            # For valid calculation, clamp indices to be within vocab range
            valid_indices = indices.clamp(0, vocab_size - 1)

            # Use the valid indices for gathering
            new_selected = torch.gather(
                new_log_ps, dim=-1, index=valid_indices.unsqueeze(-1)
            ).squeeze(-1)

            # For ignored tokens, set their importance ratio to 1.0 later (neutral value)
            # This is handled in the importance ratio calculation below
            old_selected = torch.gather(
                old_log_ps, dim=-1, index=valid_indices.unsqueeze(-1)
            ).squeeze(-1)

            # Calculate the importance ratio
            importance_ratio = torch.exp(new_selected - old_selected)

            # Set importance ratio to 1.0 for ignored tokens (won't affect loss)
            importance_ratio = torch.where(
                ignore_mask, torch.ones_like(importance_ratio), importance_ratio
            )

            # Clip the importance ratio to be between epsilon_low and epsilon_high
            importance_ratio = torch.clamp(
                importance_ratio, min=epsilon_low, max=epsilon_high
            )

            # Free memory
            del new_selected, old_selected, valid_indices, ignore_mask

        return importance_ratio

    def train(self) -> None:
        """
        The core training loop.
        """
        # clean up before training begins
        training.cleanup_before_training()

        world_size, rank = training.get_world_size_and_rank()

        # zero out the gradients before starting training
        if not self._optimizer_in_bwd:
            self._optimizer.zero_grad()
        else:
            for opt in self._optim_ckpt_wrapper.optim_map.values():
                opt.zero_grad()

        # Initialize tokens count and running loss (for grad accumulation)
        t0 = time.perf_counter()
        running_loss = 0
        num_tokens = 0
        running_ent = 0

        # NOTE: added by us - sample just once at the beginning of the epoch loop
        self._sampler.set_epoch(0)

        # Precompute reference logprobs before training starts
        if self.use_reference:
            self.precompute_reference_logprobs()

        self._profiler.start()
        # self.epochs_run should be non-zero when we're resuming from a checkpoint
        for curr_epoch in range(self.epochs_run, self.total_epochs):
            # Update the sampler to ensure data is correctly shuffled across epochs
            # in case shuffle is True
            # NOTE: removing it from here and putting it before the epoch loop
            # because our epochs are not the same as the dataloader epochs
            for _sampler_validation in self._sampler_validation_list:
                _sampler_validation.set_epoch(curr_epoch)  # NOTE: added by us

            # NOTE: added by us
            # ------ Validation Step ------ #
            self._model.eval()

            with torch.no_grad():
                for i, dataloader_validation in enumerate(
                    self._dataloader_validation_list
                ):
                    cum_val_loss = 0
                    num_eval_steps = (
                        min(self._max_validation_steps, len(dataloader_validation))
                        if self._max_validation_steps is not None
                        else len(dataloader_validation)
                    )

                    max_len_samples = 0

                    pbar_val = tqdm(total=num_eval_steps, desc="Validation")
                    # NOTE: added by us - counter to account for samples that are too long
                    idx = 0
                    for _, batch in enumerate(dataloader_validation):

                        if self._skip_max_seq_len_samples(batch):
                            max_len_samples += 1
                            continue

                        utils.batch_to_device(batch, self._device)
                        # try:
                        if self.method in ["reinforce", "grpo"]:
                            rewards = batch.pop("reward")

                        val_loss = self._loss_step(batch)
                        # except RuntimeError as e:
                        #     log.error(f"Error in validation loss computation: {e}")
                        #     val_loss = torch.tensor(0.0, device=self._device)
                        #     continue
                        cum_val_loss += val_loss
                        pbar_val.update(1)
                        pbar_val.set_description(
                            f"{self.epochs_run+1}|{self.global_step}|Validation Loss: {cum_val_loss / (idx + 1)}"
                        )
                        idx += 1

                        if (
                            self._max_validation_steps is not None
                            and idx == self._max_validation_steps
                        ):
                            break

                    mean_val_loss = cum_val_loss / (idx + 1 - max_len_samples)

                    gathered_val_loss = [
                        torch.zeros_like(mean_val_loss) for _ in range(world_size)
                    ]
                    torch.distributed.all_gather(gathered_val_loss, mean_val_loss)
                    mean_val_loss = torch.stack(gathered_val_loss).mean().cpu()
                    if self._is_rank_zero:
                        self._metric_logger.log_dict(
                            {"val_loss": mean_val_loss},
                            step=self.global_step,
                        )
                    utils.log_rank_zero(
                        log, f"Number of samples that were too long: {max_len_samples}"
                    )
                    pbar_val.close()

            # ------ Training Epoch ------ #
            # Initialize tokens count and running loss (for grad accumulation)
            t0 = time.perf_counter()
            running_loss = 0
            num_tokens = 0
            real_num_tokens = 0
            max_len_samples = 0
            # Update entropy tracking variables to include sum and mean metrics
            running_per_token_ent_sum = 0
            running_full_token_ent_sum = 0
            running_per_token_ent_mean = 0
            running_full_token_ent_mean = 0
            self._model.train()  # NOTE: added by us

            pbar = tqdm(
                total=self._steps_per_epoch, disable=not (rank == 0), desc="Training"
            )

            # NOTE: added by us - counter to account for samples that are too long
            idx = 0
            n_samples = len(self._dataloader)
            n_gpus = torch.distributed.get_world_size()
            number_leftover_samples = (
                n_samples * n_gpus
            ) % self._gradient_accumulation_steps
            for _, batch in enumerate(self._dataloader):
                if ((idx // self._gradient_accumulation_steps)) >= (
                    self._steps_per_epoch
                ) and not self.max_bsize:
                    break

                # NOTE: added by us
                if self._skip_max_seq_len_samples(batch):
                    max_len_samples += 1
                    # TODO: eventually remove this
                    continue

                # Start tracking CUDA memory for active steps for just the first epoch
                if (
                    self._is_rank_zero
                    and curr_epoch == 0
                    and self.profiler_profile_memory
                    and idx == self.profiler_wait_steps + self.profiler_warmup_steps
                ):
                    torch.cuda.memory._record_memory_history()

                utils.batch_to_device(batch, self._device)

                # Calculate the number of unmasked tokens in the current batch
                # and increment the total number of tokens seen in the step
                current_num_tokens = (
                    batch["labels"] != self._loss_fn.ignore_index
                ).sum()
                num_tokens += current_num_tokens
                # NOTE: added by us
                # let's monitor the total number of tokens
                real_num_tokens = batch["labels"].numel()

                # Shape [b, s], needed for the loss not the model
                labels = batch.pop("labels")
                if self.method in ["reinforce", "grpo"]:
                    reward = batch.pop("reward")
                else:
                    reward = 1

                with self.activations_handling_ctx:
                    logits = self._model(**batch)

                if self.use_reference:
                    # Get batch key to look up precomputed logprobs
                    batch_key = hash(tuple(batch["tokens"].flatten().tolist()))

                    # Move operations that don't need gradients out of the computation graph
                    with torch.no_grad():
                        # Use precomputed log probs for reference model
                        old_log_ps = self.reference_logprobs_cache[batch_key]
                        # Move old_log_ps back to device if needed
                        if isinstance(old_log_ps, list):
                            old_log_ps = [lp.to(self._device) for lp in old_log_ps]
                        else:
                            old_log_ps = old_log_ps.to(self._device)

                # Shift labels to compute loss
                # equivalent to doing labels[..., 1:] and logits[..., :-1, :]
                # But this way we dont need to slice the logits. We just add an ignore index to labels.
                labels = torch.hstack(
                    (labels[..., 1:], self.ignore_labels_cache[: labels.shape[0]])
                )

                # Get epsilon values based on reward sign if using importance sampling
                if self.use_reference:
                    with torch.no_grad():
                        reward_sign = torch.sign(reward)
                        # assign the epsilon values based on the reward sign in a tensorized way
                        epsilon_low = torch.where(
                            reward_sign > 0,
                            torch.tensor(self.epsilon_low_pos, device=self._device),
                            torch.tensor(self.epsilon_low_neg, device=self._device),
                        )
                        epsilon_high = torch.where(
                            reward_sign > 0,
                            torch.tensor(self.epsilon_high_pos, device=self._device),
                            torch.tensor(self.epsilon_high_neg, device=self._device),
                        )

                    # Compute loss using integrated importance sampling in the loss function
                    current_loss = (
                        self._loss_fn(
                            logits=logits,
                            labels=labels,
                            ref_logprobs=old_log_ps,
                            reward=reward,
                            epsilon_low=epsilon_low,
                            epsilon_high=epsilon_high,
                        )
                        * current_num_tokens
                    )
                else:
                    # Standard loss calculation without importance sampling
                    current_loss = self._loss_fn(logits, labels) * current_num_tokens
                    if not isinstance(reward, (int, float)) or reward != 1:
                        # Apply reward scaling if it's not the default value of 1
                        is_vector_reward = (
                            not isinstance(reward, (int, float)) and reward.numel() > 1
                        )
                        if is_vector_reward:
                            current_loss = current_loss * reward.mean()
                        else:
                            current_loss = current_loss * reward

                (
                    per_token_entropy_sum,
                    full_token_entropy_sum,
                    per_token_entropy_mean,
                    full_token_entropy_mean,
                ) = self._loss_fn.compute_entropy(logits, labels)

                # Clean up intermediate tensors to free memory
                del logits
                if self.use_reference:
                    del old_log_ps
                torch.cuda.empty_cache()  # Force CUDA to release memory

                running_loss += current_loss + self.ent_weight * full_token_entropy_mean
                running_per_token_ent_sum += per_token_entropy_sum.detach()
                running_full_token_ent_sum += full_token_entropy_sum.detach()
                running_per_token_ent_mean += per_token_entropy_mean.detach()
                running_full_token_ent_mean += full_token_entropy_mean.detach()
                del per_token_entropy_sum
                del full_token_entropy_sum
                del per_token_entropy_mean
                del full_token_entropy_mean

                # For optimizer in backward, we need to normalize before calling backward
                # This case and gradient accumulation are mutually exclusive
                if self._optimizer_in_bwd:
                    torch.distributed.all_reduce(num_tokens)
                    torch.distributed.all_reduce(running_loss)
                    current_loss = current_loss / num_tokens
                current_loss.backward()
                # Step with optimizer
                if (idx + 1) % self._gradient_accumulation_steps == 0 or (
                    (idx + 1) == n_samples
                ):

                    if not self._optimizer_in_bwd:
                        # Get total number of tokens across all ranks to normalize gradients
                        torch.distributed.all_reduce(num_tokens)
                        # This will ensure that the logged loss matches what we're optimizing
                        torch.distributed.all_reduce(running_loss)
                        # All-reduce all entropy metrics
                        torch.distributed.all_reduce(running_per_token_ent_sum)
                        torch.distributed.all_reduce(running_full_token_ent_sum)
                        torch.distributed.all_reduce(running_per_token_ent_mean)
                        torch.distributed.all_reduce(running_full_token_ent_mean)
                        training.scale_grads(self._model, 1 / num_tokens)
                        # scale grads by max_batchsize and real_batchsize
                        if self.max_bsize and (idx + 1) == n_samples:
                            # should be bsize/number of gpus
                            training.scale_grads(
                                self._model,
                                torch.tensor(number_leftover_samples / self.max_bsize),
                            )
                            log.info(
                                f"Scaling gradients by {number_leftover_samples/self.max_bsize} Original bsize = {number_leftover_samples}"
                            )
                        if self._clip_grad_norm is not None:
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                self._model.parameters(),
                                max_norm=float(self._clip_grad_norm),
                            )
                        self._optimizer.step()
                        log.info(f"optimizer step")

                        self._optimizer.zero_grad(set_to_none=True)

                    # Update the number of steps when the weights are updated
                    self.global_step += 1

                    # Step the learning rate scheduler
                    if self._lr_scheduler is not None:
                        self._lr_scheduler.step()

                    loss_to_log = running_loss.item() / num_tokens
                    pbar.update(1)
                    pbar.set_description(
                        f"{curr_epoch + 1}|{self.global_step}|Loss: {loss_to_log}"
                    )

                    # Log per-step metrics
                    if self._is_rank_zero:
                        time_per_step = time.perf_counter() - t0
                        log_dict = {
                            "loss": loss_to_log.cpu().item(),
                            "lr": get_lr(
                                (
                                    self._optimizer
                                    if not self._optimizer_in_bwd
                                    else self._optim_ckpt_wrapper
                                ),
                            ),
                            "tokens_per_second_per_gpu": real_num_tokens  # NOTE: added by us
                            / (time_per_step * world_size),
                            "per_token_entropy_sum": running_per_token_ent_sum.item()
                            / n_samples,
                            "full_token_entropy_sum": running_full_token_ent_sum.item()
                            / n_samples,
                            "per_token_entropy_mean": running_per_token_ent_mean.item()
                            / n_samples,
                            "full_token_entropy_mean": running_full_token_ent_mean.item()
                            / n_samples,
                        }
                        if self._log_peak_memory_stats:
                            log_dict.update(
                                training.get_memory_stats(device=self._device)
                            )
                        if self._clip_grad_norm is not None:
                            log_dict.update({"grad_norm": grad_norm})
                        self._metric_logger.log_dict(
                            log_dict,
                            step=self.global_step,
                        )

                    # Reset running stats for the next step
                    running_loss = 0
                    combined_loss = 0
                    num_tokens = 0
                    real_num_tokens = 0
                    running_per_token_ent_sum = 0
                    running_full_token_ent_sum = 0
                    running_per_token_ent_mean = 0
                    running_full_token_ent_mean = 0
                    t0 = time.perf_counter()

                    # Stop tracking CUDA memory now that active steps are complete
                    if (
                        self._is_rank_zero
                        and curr_epoch == 0
                        and self.profiler_profile_memory
                        and idx
                        == self.profiler_wait_steps
                        + self.profiler_warmup_steps
                        + self.profiler_active_steps
                    ):
                        torch.cuda.memory._record_memory_history(enabled=None)

                    # Step profiler
                    # Note that this is called within gradient accumulation block, hence
                    # will include multiple forward / backward passes if gradient accumulation > 1
                    self._profiler.step()

                idx += 1  # NOTE: added by us

            self.epochs_run += 1
            self.save_checkpoint(epoch=curr_epoch)

        self._profiler.stop()

    def cleanup(self) -> None:
        if self._is_rank_zero:
            self._metric_logger.close()
        destroy_process_group()


@config.parse
def recipe_main(cfg: DictConfig) -> None:
    """
    Entry point for the recipe.

    Configurable parameters are read in the following order:
        - Parameters specified in config (see available configs through ``tune ls``)
        - Overwritten by arguments from the command-line
    """
    if not training.is_distributed():
        raise RuntimeError(
            "Distributed finetune recipe should be run via a distributed launcher."
            "If using tune CLI, please specify --nnodes 1 and --nproc_per_node [num_gpus]"
        )
    init_process_group(backend="gloo" if cfg.device == "cpu" else "nccl")
    if cfg.get("fsdp_cpu_offload", False):
        # Utilize all available CPU cores for intra-op parallelism. This provides ~2x
        # speed up when benchmarking fused AdamW on CPU
        training.set_torch_num_threads()

    config.log_config(recipe_name="FullFinetuneRecipeDistributedPlus", cfg=cfg)

    recipe = FullFinetuneRecipeDistributedPlus(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())
