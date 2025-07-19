# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import sys
import time

from functools import partial
from typing import Any, Optional, Union
from warnings import warn

import torch
import torchtune.modules.common_utils as common_utils
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
from torchtune.modules.peft import (
    get_adapter_params,
    get_lora_module_names,
    set_trainable_params,
    validate_missing_and_unexpected_for_lora,
)
from torchtune.recipe_interfaces import FTRecipeInterface
from torchtune.training import DummyProfiler, PROFILER_KEY
from torchtune.training.checkpointing._checkpoint_client import (
    CheckpointClient,
    TrainingProgress,
)

from tqdm import tqdm


def hash_batch(batch):
    # Assume batch is a dict with 'labels' and data keys like 'input_ids' or 'tokens'
    h = hashlib.sha256()
    # Get the main data tensor (input_ids or tokens)
    data = batch.get("input_ids", batch.get("tokens"))
    labels = batch["labels"]
    h.update(data.cpu().numpy().tobytes())
    h.update(labels.cpu().numpy().tobytes())
    return h.hexdigest()


class LoRAFinetuneRecipeSingleDevice(FTRecipeInterface):
    """
    LoRA finetuning recipe for dense transformer-based LLMs such as Llama2. This recipe is optimized
    for single GPU training. Training on CPU is not supported.

    Features:
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

                Total Batch Size = batch_size * gradient accumulation steps.

            For example: with batch_size=1 and gradient_accumulation_steps=32 we get a total batch size of 32.

            Gradient accumulation is especially useful when you are memory constrained. In this case,
            accumulating gradients might give you better training speed than enabling activation
            checkpointing.

        - Lower precision optimizers. This recipe supports lower-precision optimizers from the bitsandbytes
            library (https://huggingface.co/docs/bitsandbytes/main/en/index). We've tested the recipe with
            8-bit AdamW and Paged AdamW.

        - Checkpointing. Model weights are checkpointed both at the end of each epoch and at the end of
            training. Currently we checkpoint both the adapter weights (trainable params only) and the
            complete merged weights (adapter weights added back to the base model). For more details
            please take a look at our LoRA tutorial
            (https://pytorch.org/torchtune/main/tutorials/lora_finetune.html).

            Optimizer State and recipe state (seed, total_epochs, number of epochs run etc) are
            only saved at the end of a given epoch and used in case of resuming training. Resuming
            training is controlled by the ``resume_from_checkpoint`` flag. Mid-epoch checkpointing is
            currently not supported.

            For more details on the checkpointer, please take a look at
            our checkpointer deepdive (https://pytorch.org/torchtune/main/tutorials/checkpointer.html).

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
        RuntimeError: If ``enable_activation_offloading`` is True and device is not CUDA or XPU.
        RuntimeError: If ``enable_activation_offloading`` is True and ``enable_activation_checkpointing`` is False.
        RuntimeError: If ``left_pad_sequence`` is set as the data collator

    """

    def __init__(self, cfg: DictConfig) -> None:
        self._device = utils.get_device(device=cfg.device)
        # Reduced precision logic
        self._dtype = training.get_dtype(cfg.dtype, device=self._device)
        # fp16 precision is explicitly disabled as it is not supported in this
        # recipe (for example, no gradient scaling).
        if self._dtype == torch.float16:
            raise ValueError(
                "fp16 precision is not supported in this recipe. Please use fp32 or bf16."
            )

        # logging attributes
        self._output_dir = cfg.output_dir
        self._log_every_n_steps = cfg.get("log_every_n_steps", 1)
        self._log_peak_memory_stats = cfg.get("log_peak_memory_stats", False)
        self._logger = utils.get_logger(cfg.log_level)

        self.save_every_n_steps = cfg.get("save_every_n_steps")

        if self._log_peak_memory_stats and self._device.type == "cpu":
            self._logger.info(
                "log_peak_memory_stats was set to True, however, training uses cpu. Setting log_peak_memory_stats=False."
            )
            self._log_peak_memory_stats = False

        # These are public properties which are updated by the checkpoint loader
        # when ``resume_from_checkpoint`` is `True` or validated in tests
        self.seed = training.set_seed(
            seed=cfg.seed, debug_mode=cfg.get("cudnn_deterministic_mode", None)
        )
        self.epochs_run = 0
        self.total_epochs = cfg.epochs
        self.max_steps_per_epoch = cfg.max_steps_per_epoch
        self.global_step = 0
        self._resume_from_checkpoint = cfg.resume_from_checkpoint
        self._save_adapter_weights_only = cfg.get("save_adapter_weights_only", False)
        if cfg.save_last_epoch_only and cfg.epochs_to_save:
            utils.log_rank_zero(
                self._logger,
                "Both save_last_epoch_only and epochs_to_save are in use. "
                "The value for save_last_epoch_only takes precedence but will be removed in a future release.",
            )
        self._save_last_epoch_only = cfg.get("save_last_epoch_only", False)
        self._epochs_to_save = (
            [self.total_epochs - 1]
            if self._save_last_epoch_only
            else cfg.get("epochs_to_save", "all")
        )
        if self._epochs_to_save == "all":
            self._epochs_to_save = list(range(self.total_epochs))
        self._gradient_accumulation_steps = cfg.gradient_accumulation_steps
        self._clip_grad_norm = cfg.get("clip_grad_norm", None)

        self._enable_async_checkpointing = cfg.get("enable_async_checkpointing", False)
        self._checkpoint_client = CheckpointClient(cfg)

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
            utils.log_rank_zero(
                self._logger,
                "Hint: enable_activation_checkpointing is True, but enable_activation_offloading isn't. "
                "Enabling activation offloading should reduce memory further.",
            )

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
        Setup the recipe state. This includes recipe state (if resume_from_checkpoint is True),
        model, tokenizer, loss, optimizer, learning rate scheduler, sampler, and dataloader.
        """
        self._metric_logger = config.instantiate(cfg.metric_logger)

        # log config with parameter override
        self._metric_logger.log_config(cfg)

        self._compile = cfg.compile
        if cfg.device == "npu" and cfg.compile:
            raise ValueError(
                "NPU does not support model compilation. Please set `compile: False` in the config."
            )
        checkpoint_dict = self._checkpoint_client.load_base_checkpoint()

        # hack to toggle to the low cpu ram version of the reparametrize_as_dtype
        # hook based on the config.
        common_utils._use_low_cpu_ram = cfg.get("low_cpu_ram", False)

        # set up model
        self._model = self._setup_model(
            cfg_model=cfg.model,
            enable_activation_checkpointing=self._enable_activation_checkpointing,
            enable_activation_offloading=self._enable_activation_offloading,
            compile_model=cfg.compile,
            base_model_state_dict=checkpoint_dict[training.MODEL_KEY],
            lora_weights_state_dict=(
                checkpoint_dict[training.ADAPTER_KEY]
                if training.ADAPTER_KEY in checkpoint_dict
                else None
            ),
        )

        self._tokenizer = config.instantiate(cfg.tokenizer)
        self._logger.info("Tokenizer is initialized from file.")

        self._optimizer = self._setup_optimizer(
            cfg_optimizer=cfg.optimizer,
            opt_state_dict=(
                checkpoint_dict[training.OPT_KEY]
                if training.OPT_KEY in checkpoint_dict
                else None
            ),
        )

        if self._resume_from_checkpoint:
            # If async checkpointing is enabled, intermediate checkpoints are saved asynchronously
            # using the DistributedCheckpointer.
            # Therefore the recipe needs to load the distributed checkpoint to restore the training
            # progress.
            if self._enable_async_checkpointing:
                checkpoint_dict = self._checkpoint_client.load_distributed_checkpoint(
                    self._model,
                    self._optimizer,
                    self._adapter_config,
                    single_device=True,
                )

            if training.ADAPTER_KEY not in checkpoint_dict:
                raise ValueError(
                    "Adapter weights not found. Please ensure a valid adapter checkpoint is provided."
                )

            # Update the recipe state from the checkpoint state dict.
            self._update_recipe_state(checkpoint_dict)

        # initialize loss
        self._loss_fn = config.instantiate(cfg.loss)
        if isinstance(self._loss_fn, SFTLoss):
            self._loss_fn.set_model_output(self._model)

        if self._compile:
            self._loss_fn = training.compile_loss(self._loss_fn)

        self._logger.info("Loss is initialized.")

        # Dataloader depends on the tokenizer and loss_fn and should be
        # setup after all of these are setup
        collate_name = cfg.get("collate_fn", "torchtune.data.padded_collate_sft")
        self._dataloader = self._setup_data(
            cfg_dataset=cfg.dataset,
            shuffle=cfg.shuffle,
            batch_size=cfg.batch_size,
            collate_fn=collate_name,
        )

        # Finally update the recipe state which can only be correctly set after all of the
        # other components have been initialized and updated.

        # Number of training steps in each epoch depends on the number of batches produced
        # by the dataloader and the max_steps_per_epoch param set by the user and is used
        # for logging and tracking training state. This should be computed after the dataloader
        # has been setup
        self._steps_per_epoch = (
            len(self._dataloader) // self._gradient_accumulation_steps
        )
        if (
            self.max_steps_per_epoch is not None
            and self.max_steps_per_epoch < self._steps_per_epoch
        ):
            self._steps_per_epoch = self.max_steps_per_epoch
            self.global_step = self.epochs_run * self._steps_per_epoch

        self.checkpoint_dir_prefix = ""
        if self.save_every_n_steps is None:
            self.save_every_n_steps = self._steps_per_epoch
            self.checkpoint_dir_prefix = "epoch"
        else:
            self.checkpoint_dir_prefix = "step"

        # Learning rate scheduler can only be set up after number of steps
        # has been computed
        self._lr_scheduler = self._setup_lr_scheduler(
            cfg_lr_scheduler=cfg.lr_scheduler,
            num_training_steps=self.total_epochs * self._steps_per_epoch,
            last_epoch=self.global_step - 1,
        )

        # Set up profiler, returns DummyProfiler (nullcontext object with no-op `step` method)
        # if cfg is missing profiler key or if `cfg.profiler.enabled = False
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
        base_model_state_dict: dict[str, Any],
        lora_weights_state_dict: Optional[dict[str, Any]] = None,
    ) -> nn.Module:
        with training.set_default_dtype(self._dtype), self._device:
            model = config.instantiate(cfg_model)

        self._lora_rank = cfg_model.lora_rank
        self._lora_alpha = cfg_model.lora_alpha
        self._lora_attn_modules = list(cfg_model.lora_attn_modules)
        self._apply_lora_to_mlp = cfg_model.apply_lora_to_mlp
        self._apply_lora_to_output = getattr(cfg_model, "apply_lora_to_output", False)
        self._adapter_config = {
            "r": self._lora_rank,
            "lora_alpha": self._lora_alpha,
            "target_modules": get_lora_module_names(
                self._lora_attn_modules,
                self._apply_lora_to_mlp,
                self._apply_lora_to_output,
            ),
            "peft_type": "LORA",
        }

        self.adapter_params = get_adapter_params(model)
        self._is_dora = any(["magnitude" in k for k in self.adapter_params.keys()])
        set_trainable_params(model, self.adapter_params)

        if compile_model:
            training.compile_model(model)

        if enable_activation_checkpointing:
            training.set_activation_checkpointing(
                model, auto_wrap_policy={modules.TransformerSelfAttentionLayer}
            )

        base_missing, base_unexpected = model.load_state_dict(
            base_model_state_dict, strict=False
        )
        # This is for any adapters that need to be initialized after base weights
        # have been loaded (e.g. DoRA).
        if self._is_dora:
            for m in model.modules():
                if hasattr(m, "initialize_dora_magnitude"):
                    m.initialize_dora_magnitude()
        if lora_weights_state_dict:
            lora_missing, lora_unexpected = model.load_state_dict(
                lora_weights_state_dict, strict=False
            )
        else:
            lora_missing, lora_unexpected = None, None

        validate_missing_and_unexpected_for_lora(
            lora_attn_modules=self._lora_attn_modules,
            apply_lora_to_mlp=self._apply_lora_to_mlp,
            apply_lora_to_output=self._apply_lora_to_output,
            state_dict_keys=model.state_dict().keys(),
            base_missing=base_missing,
            base_unexpected=base_unexpected,
            lora_missing=lora_missing,
            lora_unexpected=lora_unexpected,
        )
        # Validate model adapter params were loaded in with the expected dtype
        # TODO (rohan-varma): Further validation to ensure the appropriate base params
        # are NF4 vs bf16 based on the quantization config.
        training.validate_expected_param_dtype(
            self.adapter_params.items(), dtype=self._dtype
        )

        # activation offloading
        self.activations_handling_ctx = training.get_act_offloading_ctx_manager(
            model, enable_activation_offloading
        )

        self._logger.info(f"Model is initialized with precision {self._dtype}.")

        if self._device.type != "cpu":
            memory_stats = training.get_memory_stats(device=self._device)
            training.log_memory_stats(memory_stats)
        return model

    def _setup_optimizer(
        self, cfg_optimizer: DictConfig, opt_state_dict: Optional[dict[str, Any]] = None
    ) -> Optimizer:
        optimizer = config.instantiate(cfg_optimizer, self._model.parameters())
        if opt_state_dict:
            optimizer.load_state_dict(opt_state_dict)

        self._logger.info("Optimizer and loss are initialized.")
        return optimizer

    def _setup_lr_scheduler(
        self,
        cfg_lr_scheduler: DictConfig,
        num_training_steps: int,
        last_epoch: int,
    ) -> Optimizer:
        lr_scheduler = config.instantiate(
            cfg_lr_scheduler,
            self._optimizer,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch,
        )

        self._logger.info("Learning rate scheduler is initialized.")
        return lr_scheduler

    def _setup_data(
        self,
        cfg_dataset: DictConfig,
        shuffle: bool,
        batch_size: int,
        collate_fn: str,
    ) -> StatefulDataLoader:
        """
        All data related setup happens here. This recipe currently supports only
        map-style datasets. If a state_dict is provided (meaning we are resuming a training run),
        it is loaded into the dataloader.
        """
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
            collate_fn=(
                partial(
                    collate_fn,
                    padding_idx=self._tokenizer.pad_id,
                    ignore_idx=self._loss_fn.ignore_index,
                )
                if not packed
                else padded_collate_packed
            ),
            # dropping last avoids shape issues with compile + flex attention
            drop_last=True,
        )

        return dataloader

    def save_checkpoint(self, epoch: int, full_tensors: bool) -> None:
        self._checkpoint_client.save_checkpoint(
            model=self._model,
            optimizer=self._optimizer,
            training_progress=TrainingProgress(
                seed=self.seed,
                epochs_run=self.epochs_run,
                total_epochs=self.total_epochs,
                max_steps_per_epoch=self.max_steps_per_epoch,
                dataloader_state_dict=self._dataloader.state_dict(),
            ),
            epoch=epoch,
            adapter_config=self._adapter_config.copy(),
            adapter_only=self._save_adapter_weights_only,
            single_device=True,
            full_tensors=full_tensors,
        )

    def _loss_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        # Shape [b, s], needed for the loss not the model
        labels = batch.pop("labels")
        # run model
        with self.activations_handling_ctx:
            outputs = self._model(**batch)

        # post process for third party loss functions
        if not isinstance(self._loss_fn, SFTLoss):
            labels = labels.reshape(-1)
            outputs = outputs.reshape(-1, outputs.size(-1))

        # Compute loss
        loss = self._loss_fn(outputs, labels)

        # free outputs otherwise it peaks backward memory
        del outputs

        return loss

    def train(self) -> None:
        """
        The core training loop.
        """

        # Initialize tokens count and running loss (for grad accumulation)
        t0 = time.perf_counter()
        running_loss = 0
        num_tokens = 0

        with self._profiler as prof:
            # self.epochs_run should be non-zero when we're resuming from a checkpoint
            for curr_epoch in range(self.epochs_run, self.total_epochs):
                pbar = tqdm(total=self._steps_per_epoch)
                self._dataloader.sampler.set_epoch(curr_epoch)
                for idx, batch in enumerate(self._dataloader):
                    # Start tracking CUDA memory for active steps for just the first epoch
                    if (
                        curr_epoch == 0
                        and self.profiler_profile_memory
                        and idx == self.profiler_wait_steps + self.profiler_warmup_steps
                        and self._device.type == "cuda"
                    ):
                        torch.cuda.memory._record_memory_history()

                    utils.batch_to_device(batch, self._device)

                    # Calculate the number of unmasked tokens in the current batch
                    # and increment the total number of tokens seen in the step
                    current_num_tokens = (
                        batch["labels"] != self._loss_fn.ignore_index
                    ).sum()
                    num_tokens += current_num_tokens

                    # Loss is normalized by default so we multiply by the number of tokens
                    # This way we can normalize by the total number of tokens if we're accumulating gradients
                    current_loss = self._loss_step(batch) * current_num_tokens
                    running_loss += current_loss
                    current_loss.backward()

                    # Step with optimizer
                    if (idx + 1) % self._gradient_accumulation_steps == 0:
                        training.scale_grads(self._model, 1 / num_tokens)
                        if self._clip_grad_norm is not None:
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                self._model.parameters(),
                                max_norm=float(self._clip_grad_norm),
                            )
                        self._optimizer.step()
                        self._optimizer.zero_grad(set_to_none=True)
                        self._lr_scheduler.step()
                        # Update the number of steps when the weights are updated
                        self.global_step += 1

                        loss_to_log = running_loss.detach().item() / num_tokens
                        pbar.update(1)
                        pbar.set_description(
                            f"{curr_epoch + 1}|{self.global_step}|Loss: {loss_to_log}"
                        )

                        # Log per-step metrics
                        if self.global_step % self._log_every_n_steps == 0:
                            time_per_step = time.perf_counter() - t0
                            log_dict = {
                                "loss": loss_to_log,
                                "lr": self._optimizer.param_groups[0]["lr"],
                                "tokens_per_second_per_gpu": num_tokens / time_per_step,
                            }
                            if (
                                self._device.type != "cpu"
                                and self._log_peak_memory_stats
                            ):
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

                        # Step the profiler
                        # Note we are stepping each batch, which might not include optimizer step in the trace
                        # if the schedule cycle doesn't align with gradient accumulation.
                        prof.step()

                        # If not last checkpoint
                        if (
                            self.global_step % self.save_every_n_steps == 0
                            and curr_epoch != self.total_epochs - 1
                        ):
                            self.save_checkpoint(epoch=curr_epoch, full_tensors=False)

                    if (
                        (idx + 1) // self._gradient_accumulation_steps
                    ) == self.max_steps_per_epoch:
                        break

                self.epochs_run += 1
                if curr_epoch in self._epochs_to_save:
                    start_save_checkpoint = time.perf_counter()
                    self._logger.info(
                        f"Starting checkpoint save for epoch {curr_epoch}..."
                    )
                    self.save_checkpoint(epoch=curr_epoch)
                    self._logger.info(
                        "Checkpoint saved in {:.2f} seconds.".format(
                            time.perf_counter() - start_save_checkpoint
                        )
                    )
                else:
                    self._log.info(
                        f"Skipping checkpoint save for epoch {curr_epoch}..."
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
    config.log_config(recipe_name="LoRAFinetuneRecipeSingleDevice", cfg=cfg)
    recipe = LoRAFinetuneRecipeSingleDevice(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())
