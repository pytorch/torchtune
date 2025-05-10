# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import time

from typing import Any, Dict  # List, Optional, Union
from warnings import warn

import torch
from omegaconf import DictConfig  # ListConfig

from torchtune import training, utils  # config, modules,
from torchtune.modules.loss import SFTLoss
from torchtune.recipe_interfaces import FTRecipeInterface

from tqdm import tqdm


class QATRecipeSingleDevice(FTRecipeInterface):
    """
    Quantization-aware training (QAT) recipe for dense transformer-based LLMs such as Llama2.
    This recipe only supports single device training. Only compatible with PyTorch 2.4+.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self._device = utils.get_device(device=cfg.device)
        self._dtype = training.get_dtype(cfg.dtype, device=self._device)

        if self._dtype == torch.float16:
            raise ValueError(
                "fp16 precision is not supported in this recipe. Please use fp32 or bf16."
            )

        # logging attributes
        self._output_dir = cfg.output_dir
        self._log_every_n_steps = cfg.get("log_every_n_steps", 1)
        self._log_peak_memory_stats = cfg.get("log_peak_memory_stats", False)
        self._logger = utils.get_logger(cfg.log_level)

        if self._log_peak_memory_stats and self._device.type == "cpu":
            self._logger.info(
                "log_peak_memory_stats was set to True, however, training uses cpu. Setting log_peak_memory_stats=False."
            )
            self._log_peak_memory_stats = False

        # Training cfg
        self._resume_from_checkpoint = cfg.resume_from_checkpoint
        self._gradient_accumulation_steps = cfg.gradient_accumulation_steps
        self._optimizer_in_bwd = cfg.get("optimizer_in_bwd", False)
        self._clip_grad_norm = cfg.get("clip_grad_norm", None)
        self._fake_quant_after_n_steps = cfg.get("fake_quant_after_n_steps", None)
        self._quantizer_mode = None

        # Optimizer in backward is not compatible with gradient accumulation or gradient clipping
        if self._optimizer_in_bwd:
            if self._clip_grad_norm is not None:
                raise RuntimeError(
                    "Gradient clipping is not supported with optimizer in bwd."
                    "Please set clip_grad_norm=None, or optimizer_in_bwd=False."
                )
            if self._gradient_accumulation_steps > 1:
                raise RuntimeError(
                    "Gradient accumulation is not supported with optimizer in bwd."
                    "Please set gradient_accumulation_steps=1, or optimizer_in_bwd=False."
                )

        # activation checkpointing/offloading
        self._enable_activation_checkpointing = cfg.get(
            "enable_activation_checkpointing", False
        )
        self._enable_activation_offloading = cfg.get(
            "enable_activation_offloading", False
        )
        if self._enable_activation_offloading:
            if self._device.type != "cuda":
                raise RuntimeError(
                    "enable_activation_offloading should only be True when training on CUDA"
                )
            if not self._enable_activation_checkpointing:
                raise RuntimeError(
                    "enable_activation_offloading should only be True when enable_activation_checkpointing is True"
                )
        elif (
            self._enable_activation_checkpointing
            and cfg.checkpointer.model_type != "LLAMA3_VISION"
        ):
            utils.log_rank_zero(
                self._logger,
                "Hint: enable_activation_checkpointing is True, but enable_activation_offloading isn't. "
                "Enabling activation offloading should reduce memory further.",
            )

        # These are public properties which are updated by the checkpoint loader
        # when ``resume_from_checkpoint`` is `True` or validated in tests
        self.seed = training.set_seed(
            seed=cfg.seed, debug_mode=cfg.get("cudnn_deterministic_mode", None)
        )
        self.epochs_run = 0
        self.total_epochs = cfg.epochs
        self.max_steps_per_epoch = cfg.max_steps_per_epoch
        self.global_step = 0

    def load_checkpoint(self, **kwargs) -> None:
        """
        Responsible for loading ALL of the state for the recipe from the
        checkpoint file, including state for the model, optimizer, dataloader and training
        parameters such as the epoch and seed.
        """
        return

    def _update_recipe_state(self, ckpt_dict: Dict[str, Any]) -> None:
        """
        Updates the recipe state from checkpoint.
        """
        try:
            self.epochs_run = ckpt_dict[training.EPOCHS_KEY]

            # on mismatch, warn the user and prevent the override
            if self.seed != ckpt_dict[training.SEED_KEY]:
                warn(
                    message=(
                        "Config value for seed does not match the checkpoint value, "
                        f"using the checkpoint value: {ckpt_dict[training.SEED_KEY]}"
                    )
                )
                self.seed = ckpt_dict[training.SEED_KEY]
            if self.max_steps_per_epoch != ckpt_dict[training.MAX_STEPS_KEY]:
                warn(
                    message=(
                        "Config value for max_steps_per_epoch does not match the checkpoint value, "
                        f"using the checkpoint value: {ckpt_dict[training.MAX_STEPS_KEY]}"
                    )
                )
                self.max_steps_per_epoch = ckpt_dict[training.MAX_STEPS_KEY]

            # on mismatch, warn the user but allow the override
            if self.total_epochs != ckpt_dict[training.TOTAL_EPOCHS_KEY]:
                warn(
                    message=(
                        "Config value for total_epochs does not match the checkpoint value, "
                        f"using the config value: {self.total_epochs}"
                    )
                )

        except KeyError as e:
            raise KeyError(
                "Checkpoint does not contain the required keys needed for updating recipe state. "
                "Are you sure you passed in the right recipe checkpoint?"
            ) from e

    def setup(self, **kwargs) -> None:
        """
        Responsible for setting up all of the components necessary for training. This includes
        model, optimizer, loss function and dataloader.
        """
        return

    def train(self, **kwargs) -> None:
        """
        The core training loop
        """

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

        self._profiler.start()

        # self.epochs_run should be non-zero when we're resuming from a checkpoint
        for curr_epoch in range(self.epochs_run, self.total_epochs):
            # Update the sampler to ensure data is correctly shuffled across epochs
            # in case shuffle is True
            pbar = tqdm(total=self._steps_per_epoch)
            self._dataloader.sampler.set_epoch(curr_epoch)
            for idx, batch in enumerate(self._dataloader):
                if (
                    self.max_steps_per_epoch is not None
                    and (idx // self._gradient_accumulation_steps)
                    == self.max_steps_per_epoch
                ):
                    break

                # Start tracking CUDA memory for active steps for just the first epoch
                if (
                    curr_epoch == 0
                    and self.profiler_profile_memory
                    and idx == self.profiler_wait_steps + self.profiler_warmup_steps
                    and self._device.type == "cuda"
                ):
                    torch.cuda.memory._record_memory_history()

                # Optionally wait N steps before enabling fake quant
                if self._fake_quant_after_n_steps is not None:
                    if self.global_step == 0:
                        self._logger.info(
                            "Step 0: Disabling fake quant, will re-enable in step %s"
                            % self._fake_quant_after_n_steps
                        )
                        disable_fq = training.quantization._get_disable_fake_quant(
                            self._quantizer_mode
                        )
                        self._model.apply(disable_fq)
                    elif self.global_step == self._fake_quant_after_n_steps:
                        self._logger.info(
                            "Step %s: Enabling fake quant"
                            % self._fake_quant_after_n_steps
                        )
                        enable_fq = training.quantization._get_enable_fake_quant(
                            self._quantizer_mode
                        )
                        self._model.apply(enable_fq)

                utils.batch_to_device(batch, self._device)

                # Calculate the number of unmasked tokens in the current batch
                # and increment the total number of tokens seen in the step
                current_num_tokens = (
                    batch["labels"] != self._loss_fn.ignore_index
                ).sum()
                num_tokens += current_num_tokens

                # Shape [b, s], needed for the loss not the model
                labels = batch.pop("labels")

                with self.activations_handling_ctx:
                    outputs = self._model(**batch)
                # post process for third party loss functions
                if not isinstance(self._loss_fn, SFTLoss):
                    labels = labels.reshape(-1)
                    outputs = outputs.reshape(-1, outputs.size(-1))

                """ ensure normalizing is correct here """

                # compute loss
                current_loss = self._loss_fn(outputs, labels)

                # Loss is normalized by default so we multiply by the number of tokens
                # This way we can normalize by the total number of tokens if we're accumulating gradients
                current_loss = current_loss * current_num_tokens

                # free outputs otherwise it peaks backward memory
                del outputs

                running_loss += current_loss

                # For optimizer in backward, we need to normalize before calling backward
                # This case and gradient accumulation are mutually exclusive
                if self._optimizer_in_bwd:
                    current_loss = current_loss / current_num_tokens

                current_loss.backward()

                if (idx + 1) % self._gradient_accumulation_steps == 0:
                    if not self._optimizer_in_bwd:
                        training.scale_grads(self._model, 1 / num_tokens)
                        if self._clip_grad_norm is not None:
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                self._model.parameters(),
                                max_norm=float(self._clip_grad_norm),
                            ).full_tensor()
                        self._optimizer.step()
                        self._optimizer.zero_grad(set_to_none=True)

        return

    def save_checkpoint(self, **kwargs) -> None:
        """
        Responsible for saving ALL of the state for the recipe,
        including state for the model, optimizer, dataloader and training
        parameters such as the epoch and seed.
        """
        return

    def cleanup(self, **kwargs) -> None:
        """
        Any cleaning up needed for the recipe.
        """
        return
