# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import sys
import time

from functools import partial

from typing import Any, Optional, Union
from warnings import warn

import torch
from omegaconf import DictConfig, ListConfig

from torch import nn

from torch.optim import Optimizer
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler
from torchtune import config, modules, training, utils
from torchtune.config._utils import _get_component_from_path
from torchtune.data import padded_collate_packed
from torchtune.datasets import ConcatDataset
from torchtune.modules.loss import SFTLoss
from torchtune.recipe_interfaces import FTRecipeInterface
from torchtune.training import DummyProfiler, PROFILER_KEY

from torchtune.training.lr_schedulers import get_lr

from tqdm import tqdm


class QATRecipeSingleDevice(FTRecipeInterface):
    """
    Quantization-aware training (QAT) recipe for dense transformer-based LLMs such as Llama2.
    This recipe supports single device training (CPU or GPU).
    It is recommended to use PyTorch 2.4+ for optimal QAT support.

    Features:
        - Quantization-aware training (QAT): Perform fake quantization on weights and/or activations
          during finetuning, with the goal of ultimately producing a quantized model with minimal
          accuracy degradation. This recipe produces an unquantized model in the original dtype
          (e.g., bf16 or fp32), which has learned to be robust to quantization effects.
          This output model can then be quantized separately using standard post-training quantization (PTQ)
          techniques or by applying the quantizer's convert step.

        - Delayed fake quantization: Optionally specify the step after which fake quantization is enabled.
          Empirically, allowing the model to finetune without fake quantization initially can allow
          weight and activation values to stabilize before fake quantization is applied,
          potentially leading to improved quantized accuracy. This can be specified
          through the ``fake_quant_after_n_steps`` config option.

        - Activation Checkpointing: Controlled using the ``enable_activation_checkpointing``
          flag. This technique helps reduce memory footprint by recomputing activations
          during the backward pass instead of storing them. This is beneficial for larger
          models or batch sizes but may increase training time.

        - Activation Offloading: Controlled using the ``enable_activation_offloading``
          flag (requires ``enable_activation_checkpointing=True`` and CUDA or XPU device).
          Activations are moved to CPU memory during the forward pass and brought back to GPU
          during the backward pass, further reducing GPU memory usage. This can impact
          training speed but allows for larger models/batches.

        - Precision: Full fp32 and bf16 training are supported, controlled by the ``dtype``
          flag. Using bf16 typically halves memory usage compared to fp32 with minimal
          impact on model quality on supported hardware. fp16 precision is not supported
          for QAT with this recipe.

        - Gradient Accumulation: Simulate larger batch sizes by accumulating gradients,
          controlled by ``gradient_accumulation_steps``.
          Effective Batch Size = batch_size * gradient_accumulation_steps.
          This is useful when memory-constrained.

        - Optimizer in Backward: Optionally perform optimizer steps during the backward pass
          for potential memory savings, controlled by ``optimizer_in_bwd``. This is not
          compatible with gradient accumulation or gradient clipping.

        - Checkpointing: Model weights are checkpointed at the end of each epoch and
          at the end of training. Optimizer state and recipe state (seed, epochs run, etc.)
          are saved with epoch checkpoints for resuming training, controlled by
          ``resume_from_checkpoint``. For more details, see the checkpointer deepdive:
          https://pytorch.org/torchtune/main/deep_dives/checkpointer.html

        - Logging: Supports Terminal, Disk, WandB, and TensorBoard logging via the
          ``metric_logger`` configuration.

        - Gradient Clipping: Optional gradient clipping is supported using the ``clip_grad_norm``
          flag (set to ``None`` by default).

    For a full list of example configs for this recipe, run ``tune ls`` on the command line.
    Each config has example commands for how to kick-off training.

    Args:
        cfg (DictConfig): OmegaConf object parsed from yaml file.

    Raises:
        ValueError: If ``dtype`` is set to fp16.
        ValueError: If ``quantizer`` is not specified in the config.
        ValueError: If the specified ``quantizer`` is not in a QAT mode.
        RuntimeError: If ``optimizer_in_bwd`` is True and ``clip_grad_norm`` is enabled.
        RuntimeError: If ``optimizer_in_bwd`` is True and ``gradient_accumulation_steps`` > 1.
        RuntimeError: If ``enable_activation_offloading`` is True and the device is not CUDA or XPU.
        RuntimeError: If ``enable_activation_offloading`` is True and ``enable_activation_checkpointing`` is False.
        RuntimeError: If ``left_pad_sequence`` is set as the data collator for training.
        KeyError: If resuming from a checkpoint and the checkpoint dictionary is missing required recipe state keys.
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
            if self._device.type != "cuda" and self._device.type != "xpu":
                raise RuntimeError(
                    "enable_activation_offloading should only be True when training on CUDA or XPU"
                )
            if not self._enable_activation_checkpointing:
                raise RuntimeError(
                    "enable_activation_offloading should only be True when enable_activation_checkpointing is True"
                )
        elif (
            self._enable_activation_checkpointing
            and cfg.checkpointer.model_type != "LLAMA3_VISION"
        ):
            self._logger.info(
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

    def load_checkpoint(self, cfg_checkpointer: DictConfig) -> dict[str, Any]:
        """
        Extract the checkpoint state from file and validate. If resume_from_checkpoint
        is True, this also includes the recipe state.
        """
        self._checkpointer = config.instantiate(
            cfg_checkpointer,
            should_load_recipe_state=self._resume_from_checkpoint,
        )
        checkpoint_dict = self._checkpointer.load_checkpoint()

        if self._resume_from_checkpoint:
            self._update_recipe_state(checkpoint_dict)
        return checkpoint_dict

    def _update_recipe_state(self, ckpt_dict: dict[str, Any]) -> None:
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

    def setup(self, cfg: DictConfig) -> None:
        """
        Responsible for setting up all of the components necessary for training. This includes
        model, optimizer, loss function and dataloader.
        """
        self._metric_logger = config.instantiate(cfg.metric_logger)

        # log config with parameter override
        self._metric_logger.log_config(cfg)

        checkpoint_dict = self.load_checkpoint(cfg_checkpointer=cfg.checkpointer)

        self._compile = cfg.get("compile", False)
        self._model = self._setup_model(
            cfg_model=cfg.model,
            enable_activation_checkpointing=self._enable_activation_checkpointing,
            enable_activation_offloading=self._enable_activation_offloading,
            compile_model=self._compile,
            model_state_dict=checkpoint_dict[training.MODEL_KEY],
            quantizer_cfg=cfg.get("quantizer", None),
        )
        self._tokenizer = config.instantiate(cfg.tokenizer)

        self._optimizer = self._setup_optimizer(
            cfg_optimizer=cfg.optimizer,
            optimizer_in_bwd=self._optimizer_in_bwd,
            opt_state_dict=(
                checkpoint_dict[training.OPT_KEY]
                if self._resume_from_checkpoint
                else None
            ),
        )

        # initialize loss
        self._loss_fn = config.instantiate(cfg.loss)
        if isinstance(self._loss_fn, SFTLoss):
            self._loss_fn.set_model_output(self._model)

        if self._compile:
            training.compile_loss(self._loss_fn, verbose=True)

        self._logger.info("Loss is initialized.")

        # sampler and dataloader depend on the tokenizer and loss_fn and should be
        # setup after both of these are initialized
        collate_name = cfg.get("collate_fn", "torchtune.data.padded_collate_sft")
        self._dataloader = self._setup_data(
            cfg_dataset=cfg.dataset,
            shuffle=cfg.shuffle,
            batch_size=cfg.batch_size,
            collate_fn=collate_name,
        )

        # Finally update the recipe state which can only be correctly set after all of the
        # other components have been initialized and updated.
        #
        # Number of training steps in each epoch depends on the number of batches produced
        # by the dataloader, the max_steps_per_epoch param set by the user and the
        # gradient_accumulation_steps param. This value is used for logging and tracking
        # training state. The computation should happen after the dataloader has been setup
        self._steps_per_epoch = (
            len(self._dataloader) // self._gradient_accumulation_steps
        )
        if (
            self.max_steps_per_epoch is not None
            and self.max_steps_per_epoch < self._steps_per_epoch
        ):
            self._steps_per_epoch = self.max_steps_per_epoch
        self.global_step = self.epochs_run * self._steps_per_epoch

        # Set up profiler, returns DummyProfiler (nullcontext object with no-op `step` method)
        # if cfg is missing profiler key or if `cfg.profiler.enabled = False`
        self._profiler = self._setup_profiler(cfg.get(PROFILER_KEY, None))

    def _setup_profiler(
        self, cfg_profiler: Optional[DictConfig] = None
    ) -> Union[torch.profiler.profile, DummyProfiler]:
        """
        Parses the `profiler` section of top-level `cfg` and sets up profiler
        """

        # Missing profiler section in config, assume disabled
        if cfg_profiler is None:
            cfg_profiler = DictConfig({"enabled": False})

        # Check that component is included and set correctly
        if cfg_profiler.get("_component_", None) is None:
            cfg_profiler["_component_"] = "torchtune.training.setup_torch_profiler"
        else:
            assert (
                cfg_profiler.get("_component_")
                == "torchtune.training.setup_torch_profiler"
            ), "Only torch profiler supported currently: component must be `torchtune.training.setup_torch_profiler`"

        profiler, profiler_cfg = config.instantiate(cfg_profiler)

        self._logger.info(f" Profiler config after instantiation: {profiler_cfg}")

        self.profiler_profile_memory = profiler_cfg.get("profile_memory", False)
        if profiler_cfg["enabled"]:
            self.profiler_wait_steps = profiler_cfg["wait_steps"]
            self.profiler_warmup_steps = profiler_cfg["warmup_steps"]
            self.profiler_active_steps = profiler_cfg["active_steps"]

        return profiler

    def _setup_model(
        self,
        cfg_model: DictConfig,
        enable_activation_checkpointing: bool,
        enable_activation_offloading: bool,
        compile_model: bool,
        model_state_dict: dict[str, Any],
        quantizer_cfg: Optional[DictConfig] = None,
    ) -> nn.Module:
        """
        Set up the model
        """
        self._logger.info(
            "Instantiating model and loading checkpoint ...",
        )

        with training.set_default_dtype(self._dtype), self._device:
            model = config.instantiate(cfg_model)

        if compile_model:
            training.compile_model(model)

        if enable_activation_checkpointing:
            training.set_activation_checkpointing(
                model, auto_wrap_policy={modules.TransformerSelfAttentionLayer}
            )

        # Apply quantization-aware training during finetuning
        if quantizer_cfg is None:
            raise ValueError("Quantizer must be specified for QAT recipe.")
        quantizer = config.instantiate(quantizer_cfg)
        quantizer.precision = self._dtype
        quantizer_mode = training.quantization.get_quantizer_mode(quantizer)
        if "qat" not in quantizer_mode:
            raise ValueError(
                "Quantizer mode '%s' is not supported for finetuning" % quantizer_mode
            )
        self._quantizer_mode = quantizer_mode
        model = quantizer.prepare(model)

        # load model state dict
        model.load_state_dict(model_state_dict)

        # Enable activation offloading
        self.activations_handling_ctx = training.get_act_offloading_ctx_manager(
            model, enable_activation_offloading
        )
        self._logger.info(
            f"QAT Model (quantizer applied) is initialized with compute precision {self._dtype}."
        )

        if self._device.type != "cpu":
            memory_stats = training.get_memory_stats(device=self._device)
            training.log_memory_stats(memory_stats)

        return model

    def _setup_optimizer(
        self,
        cfg_optimizer: DictConfig,
        optimizer_in_bwd: bool = False,
        opt_state_dict: Optional[dict[str, Any]] = None,
    ) -> Union[Optimizer]:
        """
        Set up the optimizer. This method also handles loading the optimizer state_dict, if specified.
        """
        if optimizer_in_bwd:
            # Maintain a dict of optims for every parameter.
            optim_dict = {
                p: config.instantiate(cfg_optimizer, [p])
                for p in self._model.parameters()
            }
            # Register optimizer step hooks on the model to run optimizer in backward.
            training.register_optim_in_bwd_hooks(
                model=self._model, optim_dict=optim_dict
            )
            # Create a wrapper for checkpoint save/load of optimizer states when running in backward.
            optimizer = training.create_optim_in_bwd_wrapper(
                model=self._model, optim_dict=optim_dict
            )
        else:
            optimizer = config.instantiate(cfg_optimizer, self._model.parameters())

        if opt_state_dict:
            optimizer.load_state_dict(opt_state_dict)

        self._logger.info("Optimizer is initialized.")
        return optimizer

    def _setup_data(
        self,
        cfg_dataset: DictConfig,
        shuffle: bool,
        batch_size: int,
        collate_fn: str,
    ) -> StatefulDataLoader:
        if isinstance(cfg_dataset, ListConfig):
            datasets = [
                config.instantiate(single_cfg_dataset, self._tokenizer)
                for single_cfg_dataset in cfg_dataset
            ]
            ds = ConcatDataset(datasets=datasets)
            packed = getattr(ds, "packed", False)
        else:
            ds = config.instantiate(cfg_dataset, self._tokenizer)
            packed = cfg_dataset.get("packed", False)

        # Instantiate collate_fn
        if "left_pad_sequence" in collate_fn:
            raise RuntimeError("left_pad_sequence collator is only for inference.")
        collate_fn = _get_component_from_path(collate_fn)

        sampler = StatefulDistributedSampler(
            ds,
            num_replicas=1,
            rank=0,
            shuffle=shuffle,
            seed=0,
        )

        dataloader = StatefulDataLoader(
            dataset=ds,
            batch_size=batch_size,
            sampler=sampler,
            # dropping last avoids shape issues with compile + flex attention
            drop_last=True,
            collate_fn=(
                partial(
                    collate_fn,
                    padding_idx=self._tokenizer.pad_id,
                    ignore_idx=self._loss_fn.ignore_index,
                )
                if not packed
                else padded_collate_packed
            ),
        )

        self._logger.info("Dataset and Sampler are initialized.")

        return dataloader

    def train(self) -> None:
        """
        The core training loop
        """

        # zero out the gradients before starting training
        if not self._optimizer_in_bwd:
            self._optimizer.zero_grad()
        else:
            for opt in self._optimizer.optim_map.values():
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
                            )
                        self._optimizer.step()
                        self._optimizer.zero_grad(set_to_none=True)

                    # Update the number of steps when the weights are updated
                    self.global_step += 1

                    loss_to_log = running_loss.detach().item() / num_tokens
                    pbar.update(1)
                    pbar.set_description(
                        f"{curr_epoch + 1}|{self.global_step}|Loss: {loss_to_log}"
                    )

                    if self.global_step % self._log_every_n_steps == 0:
                        time_per_step = time.perf_counter() - t0
                        log_dict = {
                            "loss": loss_to_log,
                            "lr": get_lr(self._optimizer),
                            "tokens_per_second_per_gpu": num_tokens / time_per_step,
                        }
                        if self._device.type != "cpu" and self._log_peak_memory_stats:
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
                    num_tokens = 0
                    t0 = time.perf_counter()

                    # Stop tracking CUDA memory now that active steps are complete
                    if (
                        curr_epoch == 0
                        and self.profiler_profile_memory
                        and idx
                        == self.profiler_wait_steps
                        + self.profiler_warmup_steps
                        + self.profiler_active_steps
                        and self._device.type == "cuda"
                    ):
                        torch.cuda.memory._record_memory_history(enabled=None)

                    # Step profiler
                    # Note that this is called within gradient accumulation block, hence
                    # will include multiple forward / backward passes if gradient accumulation > 1
                    self._profiler.step()

            self.epochs_run += 1
            self.save_checkpoint(epoch=curr_epoch)

        self._profiler.stop()

    def save_checkpoint(
        self,
        epoch: int,
    ) -> None:
        """
        Responsible for saving ALL of the state for the recipe,
        including state for the model, optimizer, dataloader and training
        parameters such as the epoch and seed.
        """
        self._logger.info(
            "Saving checkpoint. This may take some time. Retrieving full model state dict...",
        )
        start = time.perf_counter()

        # get model's checkpoint dict for current epoch
        checkpoint_dict: dict[str, Any] = {training.MODEL_KEY: self._model.state_dict()}
        intermediate_checkpoint = epoch + 1 < self.total_epochs

        # if training is in-progress, checkpoint the optimizer state and recipe state
        # as well.
        if intermediate_checkpoint:
            checkpoint_dict.update(
                {
                    training.OPT_KEY: self._optimizer.state_dict(),
                    training.SEED_KEY: self.seed,
                    training.EPOCHS_KEY: self.epochs_run,
                    training.TOTAL_EPOCHS_KEY: self.total_epochs,
                    training.MAX_STEPS_KEY: self.max_steps_per_epoch,
                    training.DATALOADER_KEY: self._dataloader.state_dict(),
                }
            )

        self._checkpointer.save_checkpoint(
            checkpoint_dict,
            epoch=epoch,
            intermediate_checkpoint=intermediate_checkpoint,
        )
        self._logger.info(
            f"Saving checkpoint took {time.perf_counter() - start:.2f} secs"
        )

    def cleanup(self) -> None:
        self._metric_logger.close()


@config.parse
def recipe_main(cfg: DictConfig) -> None:
    """
    Entry point for the recipe.

    Configurable parameters are read in the following order:
        - Parameters specified in config (see available configs through ``tune ls``)
        - Overwritten by arguments from the command-line
    """
    config.log_config(recipe_name="QATRecipeSingleDevice", cfg=cfg)
    recipe = QATRecipeSingleDevice(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())
