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

log = utils.get_logger("DEBUG")


class FullFinetuneRecipeDistributed(FTRecipeInterface):
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
        self._device = utils.get_device(device=cfg.device)
        self._dtype = training.get_dtype(cfg.dtype, device=self._device)

        if self._dtype == torch.float16:
            raise ValueError(
                "full fp16 training is not supported with this recipe. Please use bf16 or fp32 instead."
            )

        if (
            cfg.get("fsdp_cpu_offload", False)
            and cfg.optimizer.get("fused", False)
            and not utils.torch_version_ge("2.4.0")
        ):
            raise RuntimeError(
                "Using fused optimizer on CPU is only supported in PyTorch nightly."
            )

        # logging attributes
        self._output_dir = cfg.output_dir
        self.max_bsize = cfg.max_bsize
        # extract information from output_dir
        path_parts = self._output_dir.split("/")
        self.method = cfg.method
        self._log_every_n_steps = cfg.get("log_every_n_steps", 1)
        self._log_peak_memory_stats = cfg.get("log_peak_memory_stats", False)

        # Look for the date_run pattern in all path segments
        date_run_pattern = None
        for part in path_parts:
            match = re.search(r"(\d{8}_\d{6})_(.*)", part)
            if match:
                date_run_pattern = match
                break
        if not date_run_pattern:
            # Fallback if pattern not found
            log.warning(
                f"Could not extract date_run pattern from path: {self._output_dir}"
            )
            date_part = "unknown_date"
            run_name_part = "unknown_run"
        else:
            date_part = date_run_pattern.group(1)
            run_name_part = date_run_pattern.group(2)

        # Extract epoch and seed
        # TODO: never use regex on paths, use configs
        epoch_match = re.search(r"epoch_(\d+)", path_parts[-2])
        self.epoch = int(epoch_match.group(1) if epoch_match else "0")
        # Construct run_id using date, run_name, epoch and seed
        # TODO: we should just pass this in the config
        self.run_id = f"{date_part}_{run_name_part}"

        if self._log_peak_memory_stats and self._device.type != "cuda":
            log.info(
                "log_peak_memory_stats was set to True, however, training does not use cuda. Setting log_peak_memory_stats=False."
            )
            self._log_peak_memory_stats = False

        world_size, rank = training.get_world_size_and_rank()
        self._is_rank_zero = rank == 0

        # Training cfg
        self._resume_from_checkpoint = cfg.resume_from_checkpoint
        self._gradient_accumulation_steps = cfg.gradient_accumulation_steps
        self._optimizer_in_bwd = cfg.get("optimizer_in_bwd", False)
        self._clip_grad_norm = cfg.get("clip_grad_norm", None)

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
                log,
                "Hint: enable_activation_checkpointing is True, but enable_activation_offloading isn't. "
                "Enabling activation offloading should reduce memory further.",
            )
        # Initialize epsilon values for importance sampling
        self.epsilon_low_pos = cfg.get("epsilon_low_pos", 0.8)
        self.epsilon_low_neg = cfg.get("epsilon_low_neg", 0.8)
        self.epsilon_high_pos = cfg.get("epsilon_high_pos", 1.2)
        self.epsilon_high_neg = cfg.get("epsilon_high_neg", 1.2)
        self.use_reference = cfg.get("use_reference", False)


        # Add storage for precomputed reference logprobs
        self.reference_logprobs_cache = {}


        # These are public properties which are updated by the checkpoint loader
        # when ``resume_from_checkpoint`` is `True` or validated in tests
        self.seed = training.set_seed(seed=cfg.seed)
        self.epochs_run = 0
        self.total_epochs = cfg.epochs
        self.global_step = 0

        # NOTE: added by us
        self.save_checkpoints_interval = cfg.get("save_checkpoints", 1)
        self.max_seq_len = cfg.get("max_seq_len", None)
        self._max_validation_steps = int(
            cfg.get("samples_per_validation_steps") / (cfg.batch_size * world_size)
        )
        effective_batch_size = (
            cfg.batch_size * cfg.gradient_accumulation_steps * world_size
        )
        self.max_steps_per_epoch = int(
            cfg.get("samples_per_epoch") / effective_batch_size
        )
        if self._is_rank_zero:
            log.info(
                f"Setting max validation steps to {self._max_validation_steps} (samples_per_validation_steps / batch_size)"
            )
            log.info(
                f"Setting max steps per epoch to {self.max_steps_per_epoch} (samples_per_epoch / effective_batch_size)"
            )

    def load_checkpoint(self, cfg_checkpointer: DictConfig) -> Dict[str, Any]:
        """
        Extract the checkpoint state from file and validate. If resume_from_checkpoint
        is True, this also includes the recipe state.
        """
        self._checkpointer = config.instantiate(
            cfg_checkpointer,
            resume_from_checkpoint=self._resume_from_checkpoint,
        )
        checkpoint_dict = self._checkpointer.load_checkpoint()

        if self._resume_from_checkpoint:
            self._update_recipe_state(checkpoint_dict)
        return checkpoint_dict

    def _update_recipe_state(self, ckpt_dict: Dict[str, Any]) -> None:
        """
        Updates the recipe state from checkpoint.
        """
        try:
            if self.epochs_run != ckpt_dict[training.EPOCHS_KEY]:
                warn(
                    message=(
                        "Config value for epochs_run does not match the checkpoint value, "
                        f"using the config value: {self.epochs_run}"  # NOTE changed
                    )
                )

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
                        f"using the config value: {self.max_steps_per_epoch}"  # NOTE changed
                    )
                )
                # self.max_steps_per_epoch = ckpt_dict[training.MAX_STEPS_KEY]

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
    def load_model(self, cfg: DictConfig) -> None:
        checkpoint_dict = self.load_checkpoint(cfg_checkpointer=cfg.checkpointer)
            
        self._model = self._setup_model(
            cfg_model=cfg.model,
            enable_activation_checkpointing=self._enable_activation_checkpointing,
            enable_activation_offloading=self._enable_activation_offloading,
            custom_sharded_layers=cfg.get("custom_sharded_layers", None),
            fsdp_cpu_offload=cfg.get("fsdp_cpu_offload", False),
            reshard_after_forward=cfg.get("fsdp_reshard_after_forward", True),
            model_state_dict=checkpoint_dict[training.MODEL_KEY],
            ac_mode=cfg.get("ac_mode", None),
            ac_option=cfg.get("ac_option", None),
        )

    def setup(self, cfg: DictConfig) -> None:
        """
        Setup the recipe. This includes training state (if resume_from_checkpoint is True),
        model, tokenizer, loss, optimizer, lr scheduler, sampler, and dataloader.
        """
        if self._is_rank_zero:
            wandb_kwargs = {
                "epoch": self.epoch,  # Not resuming but creating new run with constructed ID
                "run_id": cfg.get("wandb_job_id", None),
                "project": cfg.get("wandb_project", None),
                "log_dir": cfg.metric_logger.get("log_dir", None),
                "seed": self.seed,
                "group": cfg.get("wandb_group", None),
            }
            self._metric_logger = config.instantiate(cfg.metric_logger, **wandb_kwargs)

            # Check if wandb is being used in the logger
            wandb_logger = None
            if (
                hasattr(self._metric_logger, "_wandb")
                and self._metric_logger._wandb.run
            ):
                # Direct WandBLogger
                wandb_logger = self._metric_logger
            elif hasattr(self._metric_logger, "loggers"):
                # HybridLogger - check if any of the contained loggers is a WandBLogger
                for logger in self._metric_logger.loggers:
                    if hasattr(logger, "_wandb") and logger._wandb.run:
                        wandb_logger = logger
                        break
            # log config with parameter override
            self._metric_logger.log_config(cfg)
        checkpoint_dict = self.load_checkpoint(cfg_checkpointer=cfg.checkpointer)

        self._compile = cfg.get("compile", False)
        self._model = self._setup_model(
            cfg_model=cfg.model,
            enable_activation_checkpointing=self._enable_activation_checkpointing,
            enable_activation_offloading=self._enable_activation_offloading,
            custom_sharded_layers=cfg.get("custom_sharded_layers", None),
            fsdp_cpu_offload=cfg.get("fsdp_cpu_offload", False),
            reshard_after_forward=cfg.get("fsdp_reshard_after_forward", True),
            model_state_dict=checkpoint_dict[training.MODEL_KEY],
            ac_mode=cfg.get("ac_mode", None),
            ac_option=cfg.get("ac_option", None),
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

        if self._compile:
            training.compile_loss(self._loss_fn, verbose=self._is_rank_zero)

        if self._loss_fn.__class__.__name__ == "CEWithChunkedOutputLoss":
            # set num_output_chunks for model
            self._model.set_num_output_chunks(self._loss_fn.num_output_chunks)

        utils.log_rank_zero(log, "Loss is initialized.")

        # sampler and dataloader depend on the tokenizer and loss_fn and should be
        # setup after both of these are initialized

        if isinstance(cfg.dataset.get("_component_", None), str):
            if (
                cfg.dataset.get("_component_", None)[0] == "["
                and cfg.dataset.get("_component_", None)[-1] == "]"
            ):
                cfg.dataset["_component_"] = OmegaConf.create(
                    cfg.dataset["_component_"]
                )

        collate_name = cfg.get("collate_fn", "torchtune.data.padded_collate_sft")
        if self.method == "reinforce":
            collate_name = cfg.get(
                "collate_fn", "torchtune.data.padded_collate_reinforce"
            )
        cfg.dataset["split"] = "train"  # NOTE: added by us
        self._sampler, self._dataloader = self._setup_data(
            cfg_dataset=cfg.dataset,
            shuffle=cfg.shuffle,
            batch_size=cfg.batch_size,
            collate_fn=collate_name,
        )

        # NOTE: added by us
        # validation dataloader
        cfg["validation_dataset"] = deepcopy(cfg.dataset)
        self._sampler_validation_list = []
        self._dataloader_validation_list = []
        if isinstance(cfg["validation_dataset"]["_component_"], ListConfig):
            portions = [
                dataset["portion"]
                for dataset in cfg["validation_dataset"]["_component_"]
            ]
            for dataset in cfg["validation_dataset"]["_component_"]:
                dataset["split"] = "test"
                sampler_validation, dataloader_validation = self._setup_data(
                    cfg_dataset=dataset,
                    shuffle=cfg.shuffle,
                    batch_size=cfg.batch_size,
                    collate_fn=collate_name,
                )
                self._sampler_validation_list.append(sampler_validation)
                self._dataloader_validation_list.append(dataloader_validation)
            # downsample the validation using the portion property in the dataset config
            for i in range(
                1, len(self._sampler_validation_list)
            ):  # hardcoding the first portion to be the training portion
                if portions[i] == 0:
                    sample_size = min(
                        1028, int(len(self._dataloader_validation_list[i].dataset))
                    )  # hardcoding the default sample size to 1028 if the portion is 0
                else:
                    sample_size = int(
                        len(self._dataloader_validation_list[0].dataset)
                        * portions[i]
                        / (1 - sum(portions[1:]))
                    )  # hardcoding the first portion to be the training portion

                if sample_size > len(self._dataloader_validation_list[i].dataset):
                    continue
                random.seed(42)
                subset = Subset(
                    self._dataloader_validation_list[i].dataset,
                    random.sample(
                        range(len(self._dataloader_validation_list[i].dataset)),
                        sample_size,
                    ),
                )
                collate_fn = _get_component_from_path(collate_name)
                self._dataloader_validation_list[i] = DataLoader(
                    subset,
                    batch_size=cfg.batch_size,
                    shuffle=False,
                    collate_fn=collate_fn,
                )
        else:
            cfg["validation_dataset"]["split"] = "test"
            sampler_validation, dataloader_validation = self._setup_data(
                cfg_dataset=cfg["validation_dataset"],
                shuffle=cfg.shuffle,
                batch_size=cfg.batch_size,  # TODO: have a separate batch size for validation
                collate_fn=collate_name,
            )
            self._sampler_validation_list.append(sampler_validation)
            self._dataloader_validation_list.append(dataloader_validation)

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
        # If we have a wandb logger, try to get the last step from its history
        if self._is_rank_zero and wandb_logger is not None:
            try:
                # Get the current run
                run = wandb_logger._wandb.run

                # Try to get the latest global_step by querying the summary
                last_step = run.step
                log.info(f"Found last step in wandb summary: {last_step}")
                if last_step > 0:
                    self.global_step = last_step
                    log.info(
                        f"Setting global_step to last wandb step: {self.global_step}"
                    )
                else:
                    log.info("No step information found in wandb run summary")
            except Exception as e:
                log.info(f"Failed to get last step from wandb run: {e}")
        else:
            if self._is_rank_zero:
                self.global_step = self.epoch * self._steps_per_epoch
        # Broadcast global_step from rank 0 to all other ranks to ensure consistency
        if torch.distributed.is_initialized():
            global_step_tensor = torch.tensor([self.global_step], device=self._device)
            torch.distributed.broadcast(global_step_tensor, src=0)
            self.global_step = int(global_step_tensor.item())
        # Setup lr scheduler
        self._lr_scheduler = self._setup_lr_scheduler(
            cfg_lr_scheduler=cfg.get("lr_scheduler", None),
            num_training_steps=self.total_epochs * self._steps_per_epoch,
            last_epoch=self.global_step - 1,
        )

        # Set up profiler, returns DummyProfiler (nullcontext object with no-op `step` method)
        # if cfg is missing profiler key or if `cfg.profiler.enabled = False`
        self._profiler = self._setup_profiler(cfg.get(PROFILER_KEY, None))

        # Used to ignore labels for loss computation
        self.ignore_labels_cache = torch.full(
            (cfg.batch_size, 1), self._loss_fn.ignore_index, device=self._device
        )

    def _setup_lr_scheduler(
        self,
        cfg_lr_scheduler: Optional[DictConfig],
        num_training_steps: int,
        last_epoch: int,
    ) -> Optional[Optimizer]:
        """
        Set up the learning rate scheduler based on the provided configuration.
        It supports both standard optimization and optimizer-in-backward cases.

        Args:
            cfg_lr_scheduler (Optional[DictConfig]): The learning rate scheduler configuration.
            num_training_steps (int): The total number of training steps.
            last_epoch (int): The index of the last epoch.

        Returns:
            lr_scheduler (Optional[Optimizer]): The learning rate scheduler.
        """
        if cfg_lr_scheduler is None:
            if self._is_rank_zero:
                log.info(
                    "No learning rate scheduler configured. Using constant learning rate."
                )
            return None

        if self._optimizer_in_bwd:
            # Use the first optimizer from the wrapper to represent the learning rate
            optimizer = next(iter(self._optim_ckpt_wrapper.optim_map.values()))
        else:
            # Standard case: use the single optimizer
            optimizer = self._optimizer

        # Instantiate the learning rate scheduler
        lr_scheduler = config.instantiate(
            cfg_lr_scheduler,
            optimizer,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch,
        )

        if self._optimizer_in_bwd:
            # Modify the scheduler for optimizer_in_bwd case
            self._optim_ckpt_wrapper.set_lr_scheduler(lr_scheduler)

        if self._is_rank_zero:
            log.info("Learning rate scheduler is initialized.")

        return lr_scheduler

    def _setup_profiler(
        self, cfg_profiler: Optional[DictConfig] = None
    ) -> Union[torch.profiler.profile, DummyProfiler]:
        """
        Parses the `profiler` section of top-level `cfg` and sets up profiler

        Args:
            cfg_profiler (Optional[DictConfig]): ``profiler`` section of the top-level ``cfg`` (the main config passed to
                `recipe.main`). Default None.

        Returns:
            profiler: Union[torch.profiler.profile, DummyProfiler] - DummyProfiler is a nullcontext with no-op methods
            for `start`, `stop`, and `step` that can be used in place of `torch.profiler.profile` if profiler is not enabled such
            that the instrumented training loop does not need to be changed profiling is disabled.

        The profiler config can be provided in configs under the `profiler` key with the following layout:

        .. code-block:: yaml
            profiler:
                enabled: bool

                #Output directory of trace artifacts
                output_dir: str

            #`torch.profiler.ProfilerActivity` types to trace
            cpu: bool
            cuda: bool

                #Trace options
                profile_memory: bool
                with_stack: bool
                record_shapes: bool
                with_flops: bool

            # `torch.profiler.schedule` options:
            # wait_steps -> wait, warmup_steps -> warmup, active_steps -> active, num_cycles -> repeat
            wait_steps: int
            warmup_steps: int
            active_steps: int
            num_cycles: int
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

        utils.log_rank_zero(
            log, f" Profiler config after instantiation: {profiler_cfg}"
        )
        if self._is_rank_zero:
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
        fsdp_cpu_offload: bool,
        reshard_after_forward: bool,
        model_state_dict: Dict[str, Any],
        custom_sharded_layers: Optional[List[str]] = None,
        ac_mode: Optional[str] = None,
        ac_option: Optional[int] = None,
    ) -> nn.Module:
        """
        Model initialization has some important considerations:
           a. To minimize GPU peak memory, we initialize the model on meta device with
              the right dtype
           b. All ranks calls ``load_state_dict`` without peaking CPU RAMs since
              full state dicts are loaded with ``torch.load(mmap=True)``
        """

        utils.log_rank_zero(
            log,
            "FSDP is enabled. Instantiating model and loading checkpoint on Rank 0 ...",
        )
        init_start = time.perf_counter()

        with training.set_default_dtype(self._dtype), torch.device("meta"):
            model = config.instantiate(cfg_model)

        if self._compile:
            training.compile_model(model, verbose=self._is_rank_zero)

        # We currently have two versions of activation checkpointing in this recipe
        # for testing and BC purposes. ``enable_activation_checkpointing`` controls
        # the older version of AC and this behavior is unchanged
        # ac_mode and ac_option together control selective AC. This is only enabled
        # when these are set AND ``enable_activation_checkpointing`` is set to False
        # We'll clean this up as soon as testing of AC is complete
        if (not enable_activation_checkpointing) and (ac_mode is not None):
            apply_selective_activation_checkpointing(
                model,
                ac_mode,
                ac_option,
            )

        # original activation checkpointing (full) - flip the condition above
        if enable_activation_checkpointing and ac_mode is None:
            training.set_activation_checkpointing(
                model, auto_wrap_policy={modules.TransformerSelfAttentionLayer}
            )

        # For FSDP sharding
        fsdp_shard_conditions = [
            partial(
                training.get_shard_conditions,
                names_to_match=custom_sharded_layers,
            )
        ]
        training.shard_model(
            model=model,
            shard_conditions=fsdp_shard_conditions,
            cpu_offload=fsdp_cpu_offload,
            reshard_after_forward=reshard_after_forward,
        )

        with training.set_default_dtype(self._dtype), self._device:
            for m in model.modules():
                # RoPE is not covered in state dict
                if hasattr(m, "rope_init"):
                    m.rope_init()

        # This method will convert the full model state dict into a sharded state
        # dict and load into the model
        training.load_from_full_model_state_dict(
            model,
            model_state_dict,
            self._device,
            self._is_rank_zero,
            strict=True,
            cpu_offload=fsdp_cpu_offload,
        )

        # activation offloading
        self.activations_handling_ctx = training.get_act_offloading_ctx_manager(
            model, enable_activation_offloading
        )

        # Ensure no params and buffers are on meta device
        training.validate_no_params_on_meta_device(model)

        utils.log_rank_zero(
            log,
            f"Instantiating model and loading checkpoint took {time.perf_counter() - init_start:.2f} secs",
        )
        if self._is_rank_zero:
            memory_stats = training.get_memory_stats(device=self._device)
            training.log_memory_stats(memory_stats)

        # synchronize before training begins
        torch.distributed.barrier()

        return model

    def _setup_optimizer(
        self,
        cfg_optimizer: DictConfig,
        optimizer_in_bwd: bool = False,
        opt_state_dict: Optional[Dict[str, Any]] = None,
    ) -> Optional[Optimizer]:
        if optimizer_in_bwd:
            # Maintain a dict of optims for every parameter.
            optim_dict = {
                param: config.instantiate(cfg_optimizer, [param])
                for param in self._model.parameters()
            }

            # Register optimizer step hooks on the model to run optimizer in backward.
            training.register_optim_in_bwd_hooks(
                model=self._model, optim_dict=optim_dict
            )
            # Create a wrapper for checkpoint save/load of optimizer states when running in backward.
            self._optim_ckpt_wrapper = training.create_optim_in_bwd_wrapper(
                model=self._model, optim_dict=optim_dict
            )
            # Load optimizer states for each param. If optimizer states are being restored in an optimizer in
            # backward run, these need to have been saved with the same setting. Cannot restore from runs that
            # did not use optimizer in backward.
            if opt_state_dict is not None:
                for param in opt_state_dict.keys():
                    try:
                        training.load_from_full_optimizer_state_dict(
                            self._optim_ckpt_wrapper.state_dict()[param],
                            opt_state_dict[param],
                            self._device,
                        )
                    except BaseException as e:
                        raise RuntimeError(
                            "Failed loading in-backward optimizer checkpoints."
                            "Please make sure run being restored from was using in-backward optimizer."
                        ) from e
            utils.log_rank_zero(log, "In-backward optimizers are set up.")
            return None
        else:
            optimizer = config.instantiate(cfg_optimizer, self._model.parameters())
            if opt_state_dict:
                training.load_from_full_optimizer_state_dict(
                    optimizer,
                    opt_state_dict,
                    self._device,
                )

            utils.log_rank_zero(log, "Optimizer is initialized.")
            return optimizer

    def _setup_data(
        self,
        cfg_dataset: DictConfig,
        shuffle: bool,
        batch_size: int,
        collate_fn: str,
    ) -> Tuple[DistributedSampler, DataLoader]:
        """
        All data related setup happens here. Currently this recipe only supports the
        DistributedSamplers with Map-style Datasets which fit into memory. Other samplers,
        iterable datasets and streaming datasets are not supported.
        """
        world_size, rank = training.get_world_size_and_rank()

        def convert_to_nested(config):
            """
            Converts a flattened dictionary (e.g., 'column_map.output')
            into a nested dictionary structure.

            Args:
                config (dict): A dictionary with flattened keys.

            Returns:
                dict: A dictionary with nested keys.
            """
            nested_config = {}

            for key, value in config.items():
                parts = key.split(".")  # Split the keys by '.'
                current = nested_config

                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]

                current[parts[-1]] = value

            return nested_config

        if isinstance(cfg_dataset.get("_component_", None), ListConfig):
            portions = []
            datasets = []
            for single_cfg_dataset in cfg_dataset["_component_"]:
                single_cfg_dataset = DictConfig(
                    convert_to_nested(deepcopy(single_cfg_dataset))
                )

                portion = (
                    single_cfg_dataset.pop("portion")
                    if "portion" in single_cfg_dataset
                    else None
                )
                portions.append(portion)
                datasets.append(
                    config.instantiate(
                        single_cfg_dataset,
                        self._tokenizer,
                    )
                )
            ds = ConcatDataset(datasets=datasets, portions=portions)
            packed = False
        else:
            new_single_cfg_dataset = DictConfig(
                convert_to_nested(deepcopy(cfg_dataset))
            )
            new_single_cfg_dataset.pop("portion", None)
            ds = config.instantiate(new_single_cfg_dataset, self._tokenizer)
            packed = new_single_cfg_dataset.get("packed", False)

        # if isinstance(cfg_dataset, ListConfig):
        #     datasets = [
        #         config.instantiate(single_cfg_dataset, self._tokenizer)
        #         for single_cfg_dataset in cfg_dataset
        #     ]
        #     ds = ConcatDataset(datasets=datasets)
        #     packed = False
        # else:
        #     ds = config.instantiate(cfg_dataset, self._tokenizer)
        #     packed = cfg_dataset.get("packed", False)

        # Instantiate collate_fn
        if "left_pad_sequence" in collate_fn:
            raise RuntimeError("left_pad_sequence collator is only for inference.")
        collate_fn = _get_component_from_path(collate_fn)

        sampler = DistributedSampler(
            ds, num_replicas=world_size, rank=rank, shuffle=shuffle, seed=self.seed
        )
        dataloader = DataLoader(
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

        utils.log_rank_zero(log, "Dataset and Sampler are initialized.")

        return sampler, dataloader

    def save_checkpoint(
        self,
        epoch: int,
    ) -> None:
        """
        Checkpoint the state of the recipe. The constructed checkpoint state dict
        contains the following information:
        - Model weights with key training.MODEL_KEY
        - Relevant recipe state if training is not complete

        Checkpointer will save the model weights and recipe state in
        different checkpoint files. To correctly resume training from an intermediate checkpoint,
        the model weights and recipe state must be provided.
        """
        # NOTE: added by us
        if self.save_checkpoints_interval:
            if epoch + 1 == self.total_epochs:
                pass
            elif (
                self.save_checkpoints_interval > 0
                and epoch % self.save_checkpoints_interval == 0
            ):
                pass
            else:
                return
        else:
            return

        # final dict passed onto the checkpointer
        checkpoint_dict = {}

        # NOTE: added by us - setting intermediate_checkpoint to True always, because we will resume
        # training from the last checkpoint
        # intermediate_checkpoint = epoch + 1 < self.total_epochs
        intermediate_checkpoint = True

        utils.log_rank_zero(
            log,
            "Saving checkpoint. This may take some time. Retrieving full model state dict...",
        )
        start = time.perf_counter()

        # To prevent GPU memory from spiking during checkpoint save,
        # we consolidate the full model and optim state dicts on CPU for rank 0
        cpu_state_dict = training.gather_cpu_state_dict(
            self._model.state_dict(),
            self._is_rank_zero,
            device=self._device,
        )

        utils.log_rank_zero(
            log,
            f"Getting full model state dict took {time.perf_counter() - start:.2f} secs",
        )

        if intermediate_checkpoint:
            start = time.perf_counter()
            utils.log_rank_zero(log, "Getting optimizer state dict...")
            if not self._optimizer_in_bwd:
                opt_state_dict = training.get_full_optimizer_state_dict(
                    self._optimizer,
                    self._is_rank_zero,
                    device=self._device,
                )
            else:
                opt_state_dict = {}
                for param, opt in self._optim_ckpt_wrapper.optim_map.items():
                    opt_state_dict[param] = training.get_full_optimizer_state_dict(
                        opt, self._is_rank_zero, device=self._device
                    )
            utils.log_rank_zero(
                log,
                f"Getting optimizer state dict took {time.perf_counter() - start:.2f} secs",
            )
        else:
            opt_state_dict = None

        # Now that we have the model and opt state dict, create the actual checkpoint dict
        # to be sent to the checkpointer and ultimately written to file

        if self._is_rank_zero:
            start = time.perf_counter()
            checkpoint_dict.update({training.MODEL_KEY: cpu_state_dict})

            # if training is in-progress, checkpoint the optimizer state and recipe state
            # as well.
            if intermediate_checkpoint:
                checkpoint_dict.update(
                    {
                        training.OPT_KEY: opt_state_dict,
                        training.SEED_KEY: self.seed,
                        training.EPOCHS_KEY: self.epochs_run,
                        training.TOTAL_EPOCHS_KEY: self.total_epochs,
                        training.MAX_STEPS_KEY: self.max_steps_per_epoch,
                    }
                )

            self._checkpointer.save_checkpoint(
                checkpoint_dict,
                epoch=epoch,
                intermediate_checkpoint=intermediate_checkpoint,
            )
            log.info(f"Saving checkpoint took {time.perf_counter() - start:.2f} secs")

        torch.distributed.barrier()

    def _loss_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Shape [b, s], needed for the loss not the model
        labels = batch.pop("labels")

        with self.activations_handling_ctx:
            logits = self._model(**batch)

        # Shift labels to compute loss
        # equivalent to doing labels[..., 1:] and logits[..., :-1, :]
        # But this way we dont need to slice the logits. We just add an ignore index to labels.
        labels = torch.hstack(
            (labels[..., 1:], self.ignore_labels_cache[: labels.shape[0]])
        )
        if not isinstance(logits, list):
            labels = labels.reshape(-1)
            logits = logits.reshape(-1, logits.size(-1))

        # Compute loss
        loss = self._loss_fn(logits, labels)
        # free logits otherwise it peaks backward memory
        del logits

        return loss

    # NOTE: added by us
    def _skip_max_seq_len_samples(self, batch: Dict[str, torch.Tensor]) -> bool:
        """
        Skip samples that are too long. This is needed for the training loop to handle
        samples that are too long to fit in the model.
        """
        if self.max_seq_len is None:
            return False
        return len(batch["tokens"][0]) > self.max_seq_len

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

        # NOTE: added by us - sample just once at the beginning of the epoch loop
        self._sampler.set_epoch(0)

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
                        if self.method == "reinforce":
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
            running_ent = 0
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
                if self.method == "reinforce":
                    reward = batch.pop("reward")
                else:
                    reward = 1

                with self.activations_handling_ctx:
                    logits = self._model(**batch)
                # calculate the entropy of the models responses

                # Shift labels to compute loss
                # equivalent to doing labels[..., 1:] and logits[..., :-1, :]
                # But this way we dont need to slice the logits. We just add an ignore index to labels.
                labels = torch.hstack(
                    (labels[..., 1:], self.ignore_labels_cache[: labels.shape[0]])
                )
                if not isinstance(logits, list):
                    labels = labels.reshape(-1)
                    logits = logits.reshape(-1, logits.size(-1))

                # Compute loss
                # Loss is normalized by default so we multiply by the number of tokens
                # This way we can normalize by the total number of tokens if we're accumulating gradients

                current_loss = (
                    self._loss_fn(logits, labels) * current_num_tokens * reward
                )
                #before you compute the entropy extract the single logit from the label 
               
                entropy = self._loss_fn.compute_entropy(logits,labels)
                # free logits otherwise it peaks backward memory
                del logits


                running_ent += entropy.detach()
                running_loss += current_loss
                del entropy

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
                        torch.distributed.all_reduce(running_ent)
                        # Manually scale the gradients from unnormalized loss by total # of tokens
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
                    if ( self._is_rank_zero):
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
                            "entropy": running_ent.item() / real_num_tokens,
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
                    num_tokens = 0
                    real_num_tokens = 0  # NOTE: added by us
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

    config.log_config(recipe_name="FullFinetuneRecipeDistributed", cfg=cfg)

    recipe = FullFinetuneRecipeDistributed(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())
