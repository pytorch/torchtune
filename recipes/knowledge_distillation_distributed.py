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
from torch.distributed import destroy_process_group, init_process_group

from torch.optim import Optimizer
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler
from torchtune import config, modules, training, utils
from torchtune.data import padded_collate_packed, padded_collate_sft
from torchtune.datasets import ConcatDataset
from torchtune.modules.peft import (
    AdapterModule,
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


class KDRecipeDistributed(FTRecipeInterface):
    """
    Knowledge distillation recipe for dense transformer-based LLMs such as Llama3. This recipe is optimized
    for single GPU training. Training on CPU is not supported.

    Features:
        - FSDP. Supported using PyTorch's FSDP APIs. CPU offload of parameters, gradients, and optimizer states
            is supported via ``fsdp_cpu_offload``. Resharding of parameters after the forward pass is
            done by default (corresponding to FULL_SHARD sharding strategy), but can be disabled by setting the config
            ``fsdp_reshard_after_forward`` to False (this corresponds to SHARD_GRAD_OP sharding strategy).
            DDP is currently not supported. Training on CPU is not supported.

        - Activation Checkpointing. This can be controlled using the ``activation_checkpointing``
            flag. Activation checkpointing helps reduce the memory footprint since we no longer keep
            activations in memory and instead recompute them during the backward pass. This is especially
            helpful for larger batch sizes when you're memory constrained. But these savings in memory
            come at the cost of training performance. In most cases training can slow-down quite a bit as
            a result of this activation recomputation.

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

    For a full list of example configs for this recipe, run ``tune ls`` on the command line. Each config
    has example commands for how to kick-off training.

    Args:
        cfg (DictConfig): OmegaConf object parsed from yaml file

    Raises:
        ValueError: If ``dtype`` is set to fp16.
        RuntimeError: If ``dtype`` is set to bf16 and the hardware does not support bf16.

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

        # Set up the backend for distributed training (NCCL, GLOO, etc.)
        self._enable_async_checkpointing = cfg.get("enable_async_checkpointing", False)
        self.fsdp_cpu_offload = cfg.get("fsdp_cpu_offload", False)
        self.distributed_backend = training.get_distributed_backend(
            cfg.device,
            offload_ops_to_cpu=self.fsdp_cpu_offload
            or self._enable_async_checkpointing,
        )
        init_process_group(self.distributed_backend)

        self.world_size, self.rank = utils.get_world_size_and_rank()

        self._is_rank_zero = self.rank == 0

        # logging attributes
        self._output_dir = cfg.output_dir
        self._log_every_n_steps = cfg.get("log_every_n_steps", 1)
        self._log_peak_memory_stats = cfg.get("log_peak_memory_stats", False)
        self._logger = utils.get_logger(cfg.log_level)

        # training attributes
        self._enable_async_checkpointing = cfg.get("enable_async_checkpointing", False)
        self._checkpoint_client = CheckpointClient(cfg)

        self._enable_activation_checkpointing = cfg.enable_activation_checkpointing

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
        self._gradient_accumulation_steps = cfg.gradient_accumulation_steps
        self._kd_ratio = cfg.get("kd_ratio", 0.5)
        self.save_every_n_steps = cfg.get("save_every_n_steps")

    def load_teacher_checkpoint(self, cfg: DictConfig) -> dict[str, Any]:
        """
        Extract the teacher checkpoint state from file.
        """
        teacher_checkpointer = config.instantiate(
            cfg.teacher_checkpointer,
        )

        teacher_checkpoint_client = CheckpointClient(cfg, teacher_checkpointer)
        checkpoint_dict = teacher_checkpoint_client.load_base_checkpoint()
        return checkpoint_dict

    def _update_recipe_state(self, ckpt_dict: dict[str, Any]) -> None:
        """
        Updates the recipe state from checkpoint.
        """
        try:
            self.epochs_run = ckpt_dict[training.EPOCHS_KEY]
            self.global_step = ckpt_dict[training.STEPS_KEY]

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
        if self._is_rank_zero:
            self._metric_logger = config.instantiate(cfg.metric_logger)
            config.log_config(recipe_name="KDRecipeDistributed", cfg=cfg)
            # log config with parameter override
            self._metric_logger.log_config(cfg)

        self._compile = cfg.get("compile", False)
        checkpoint_dict = self._checkpoint_client.load_base_checkpoint()
        teacher_checkpoint_dict = self.load_teacher_checkpoint(cfg=cfg)

        # set up model
        self._model = self._setup_model(
            cfg_model=cfg.model,
            enable_activation_checkpointing=cfg.enable_activation_checkpointing,
            fsdp_cpu_offload=cfg.get("fsdp_cpu_offload", False),
            reshard_after_forward=cfg.get("fsdp_reshard_after_forward", True),
            base_model_state_dict=checkpoint_dict[training.MODEL_KEY],
            lora_weights_state_dict=(
                checkpoint_dict[training.ADAPTER_KEY]
                if training.ADAPTER_KEY in checkpoint_dict
                else None
            ),
        )

        self._teacher_model = self._setup_teacher_model(
            model_cfg=cfg.teacher_model,
            custom_sharded_layers=cfg.get("custom_sharded_layers", None),
            fsdp_cpu_offload=cfg.get("fsdp_cpu_offload", False),
            reshard_after_forward=cfg.get("fsdp_reshard_after_forward", True),
            model_state_dict=teacher_checkpoint_dict[training.MODEL_KEY],
        )

        self._tokenizer = config.instantiate(cfg.tokenizer)
        utils.log_rank_zero(self._logger, "Tokenizer is initialized from file.")

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
                )

            if training.ADAPTER_KEY not in checkpoint_dict:
                raise ValueError(
                    "Adapter weights not found. Please ensure a valid adapter checkpoint is provided."
                )

            # Update the recipe state from the checkpoint state dict.
            self._update_recipe_state(checkpoint_dict)

        # initialize loss
        self._loss_fn = config.instantiate(cfg.loss)
        self._kd_loss_fn = config.instantiate(cfg.kd_loss)
        if self._compile:
            self._loss_fn = training.compile_loss(
                self._loss_fn, verbose=self._is_rank_zero
            )
            self._kd_loss_fn = training.compile_loss(
                self._kd_loss_fn, verbose=self._is_rank_zero
            )

        if self._loss_fn.__class__.__name__ == "CEWithChunkedOutputLoss":
            # set num_output_chunks for model
            self._model.set_num_output_chunks(self._loss_fn.num_output_chunks)
            self._teacher_model.set_num_output_chunks(self._loss_fn.num_output_chunks)
            # assert _loss_fn and _kd_loss_fn have the same num_output_chunks
            assert (
                self._loss_fn.num_output_chunks == self._kd_loss_fn.num_output_chunks
            ), "Number of output chunks for loss_fn and kd_loss_fn must be the same."
        elif getattr(self._loss_fn, "linear_projection", False):
            raise ValueError(
                "Linear losses are not supported yet for KD. Please use the deprecated CEWithChunkedOutputLoss."
            )

        utils.log_rank_zero(self._logger, "Loss is initialized.")

        # Dataloader depends on the tokenizer and loss_fn and should be
        # setup after all of these are setup
        self._dataloader = self._setup_data(
            cfg_dataset=cfg.dataset,
            shuffle=cfg.shuffle,
            batch_size=cfg.batch_size,
            dataloader_state_dict=checkpoint_dict.get(training.DATALOADER_KEY, None),
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

        if self.save_every_n_steps is None:
            self.save_every_n_steps = self._steps_per_epoch
            self.checkpoint_dir_prefix = "epoch"
        else:
            self.checkpoint_dir_prefix = "step"

        if (
            self._resume_from_checkpoint
            and self.global_step % self._steps_per_epoch == 0
        ):
            list(self._dataloader)

        # Learning rate scheduler can only be set up after number of steps
        # has been computed
        self._lr_scheduler = self._setup_lr_scheduler(
            cfg_lr_scheduler=cfg.lr_scheduler,
            num_training_steps=self.total_epochs * self._steps_per_epoch,
            last_epoch=self.global_step - 1,
        )

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

        utils.log_rank_zero(
            self._logger, f" Profiler config after instantiation: {profiler_cfg}"
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
        fsdp_cpu_offload: bool,
        reshard_after_forward: bool,
        base_model_state_dict: dict[str, Any],
        custom_sharded_layers: Optional[list[str]] = None,
        lora_weights_state_dict: Optional[dict[str, Any]] = None,
    ) -> nn.Module:
        """
        Model initialization has some important considerations:
          a. To minimize GPU peak memory, we initialize the model on meta device with
             the right dtype
          b. All ranks calls ``load_state_dict`` without peaking CPU RAMs since
             full state dicts are loaded with ``torch.load(mmap=True)``
          c. We register (pre-)forward hooks with ``fully_shard`` instead of wrapping `nn.Module`
        """

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

        utils.log_rank_zero(
            self._logger,
            "FSDP is enabled. Instantiating model and loading checkpoint on Rank 0 ...",
        )
        init_start = time.perf_counter()

        with training.set_default_dtype(self._dtype), torch.device("meta"):
            model = config.instantiate(cfg_model)

        self.adapter_params = get_adapter_params(model)
        set_trainable_params(model, self.adapter_params)

        if self._compile:
            training.compile_model(model, verbose=self._is_rank_zero)

        if enable_activation_checkpointing:
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

        if lora_weights_state_dict:
            lora_missing, lora_unexpected = training.load_from_full_model_state_dict(
                model,
                lora_weights_state_dict,
                self._device,
                cpu_offload=fsdp_cpu_offload,
            )
        else:
            lora_missing, lora_unexpected = None, None

        # Initializer for LoRA params and RoPE buffers
        with training.set_default_dtype(self._dtype), self._device:
            lora_device = "cpu" if fsdp_cpu_offload else self._device
            for m in model.modules():
                if isinstance(m, AdapterModule) and not lora_weights_state_dict:
                    # lora may not be covered in state dict
                    # if finetune for the 1st time
                    m.to_empty(device=lora_device)
                    m.initialize_parameters()
                # RoPE is not covered in state dict
                if hasattr(m, "rope_init"):
                    m.rope_init()

        base_missing, base_unexpected = training.load_from_full_model_state_dict(
            model,
            base_model_state_dict,
            self._device,
            cpu_offload=fsdp_cpu_offload,
        )
        for m in model.modules():
            if hasattr(m, "initialize_dora_magnitude"):
                m.initialize_dora_magnitude()

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
        # Ensure no params and buffers are on meta device
        training.validate_no_params_on_meta_device(model)

        utils.log_rank_zero(
            self._logger,
            f"Instantiating student model and loading checkpoint took {time.perf_counter() - init_start:.2f} secs",
        )
        if self._is_rank_zero:
            memory_stats = training.get_memory_stats(device=self._device)
            training.log_memory_stats(
                memory_stats, message="Memory stats after student model init:"
            )

        # synchronize before training begins
        torch.distributed.barrier()

        return model

    def _setup_teacher_model(
        self,
        model_cfg: DictConfig,
        custom_sharded_layers: Optional[list[str]],
        fsdp_cpu_offload: bool,
        reshard_after_forward: bool,
        model_state_dict: dict[str, Any],
    ) -> nn.Module:
        """
        Model initialization for teacher model has some important considerations:
          a. To minimize GPU peak memory, we initialize the model on meta device with
              the right dtype
           b. All ranks calls ``load_state_dict`` without peaking CPU RAMs since
              full state dicts are loaded with ``torch.load(mmap=True)``
        """

        utils.log_rank_zero(
            self._logger,
            "FSDP enabled. Instantiating teacher model and loading checkpoint on Rank 0 ...",
        )
        init_start = time.perf_counter()

        with training.set_default_dtype(self._dtype), torch.device("meta"):
            model = config.instantiate(model_cfg)

        if self._compile:
            training.compile_model(model, verbose=self._is_rank_zero)

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
            strict=True,
            cpu_offload=fsdp_cpu_offload,
        )

        # Put model in eval mode.
        # Note: This will not disable the dropout applied in SDPA,
        # see https://github.com/pytorch/pytorch/issues/124464
        model.eval()

        for p in model.parameters():
            p.requires_grad = False

        # Ensure no params and buffers are on meta device
        training.validate_no_params_on_meta_device(model)

        utils.log_rank_zero(
            self._logger,
            f"Instantiating teacher model and loading checkpoint took {time.perf_counter() - init_start:.2f} secs",
        )
        if self._is_rank_zero:
            memory_stats = training.get_memory_stats(device=self._device)
            training.log_memory_stats(
                memory_stats, message="Memory stats after teacher model init:"
            )

        # synchronize before training begins
        torch.distributed.barrier()

        return model

    def _setup_optimizer(
        self, cfg_optimizer: DictConfig, opt_state_dict: Optional[dict[str, Any]] = None
    ) -> Optimizer:
        optimizer = config.instantiate(cfg_optimizer, self._model.parameters())
        if opt_state_dict:
            training.load_from_full_optimizer_state_dict(
                self._model,
                optimizer,
                opt_state_dict,
                self._device,
            )

        utils.log_rank_zero(self._logger, "Optimizer is initialized.")
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

        utils.log_rank_zero(self._logger, "Learning rate scheduler is initialized.")
        return lr_scheduler

    def _setup_data(
        self,
        cfg_dataset: DictConfig,
        shuffle: bool,
        batch_size: int,
        dataloader_state_dict: Optional[dict[str, Any]] = None,
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

        sampler = StatefulDistributedSampler(
            ds,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=shuffle,
        )

        dataloader = StatefulDataLoader(
            dataset=ds,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=(
                partial(
                    padded_collate_sft,
                    padding_idx=self._tokenizer.pad_id,
                    ignore_idx=self._loss_fn.ignore_index,
                )
                if not packed
                else partial(
                    padded_collate_packed,
                )
            ),
            # dropping last avoids shape issues with compile + flex attention
            drop_last=True,
        )

        if dataloader_state_dict is not None:
            dataloader.load_state_dict(dataloader_state_dict)

        utils.log_rank_zero(self._logger, "Dataset and Sampler are initialized.")

        return dataloader

    def save_checkpoint(self, epoch: int, full_tensors: bool) -> None:
        training_progress_epoch = epoch
        if self.global_step % self._steps_per_epoch == 0:
            training_progress_epoch += 1

        self._checkpoint_client.save_checkpoint(
            model=self._model,
            optimizer=self._optimizer,
            training_progress=TrainingProgress(
                seed=self.seed,
                epochs_run=training_progress_epoch,
                total_epochs=self.total_epochs,
                steps_run=self.global_step,
                max_steps_per_epoch=self.max_steps_per_epoch,
                dataloader_state_dict=self._dataloader.state_dict(),
            ),
            epoch=epoch,
            full_tensors=full_tensors,
            dir_prefix=self.checkpoint_dir_prefix,
            adapter_config=self._adapter_config.copy(),
            adapter_only=self._save_adapter_weights_only,
        )

    def _loss_step(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Both are shape [b, s]
        tokens, labels = batch["tokens"], batch["labels"]

        # Get the attention mask and position ids from the dataset if they
        # exist. Currently, only sample packing in PackedDataset returns these
        mask = batch.get("mask", None)  # shape [b, s, s]
        input_pos = batch.get("input_pos", None)  # shape [b, s]

        # run model
        logits = self._model(tokens, mask=mask, input_pos=input_pos)

        if not isinstance(logits, list):
            labels = labels.reshape(-1)
            logits = logits.reshape(-1, logits.size(-1))

        # Compute KD loss
        with torch.no_grad():
            teacher_logits = self._teacher_model(tokens, mask=mask, input_pos=input_pos)

        # Compute kd loss
        kd_loss = self._kd_loss_fn(logits, teacher_logits, labels)

        # Compute loss
        loss = self._loss_fn(logits, labels)

        # free logits otherwise it peaks backward memory
        del logits
        del teacher_logits

        return loss, kd_loss

    def train(self) -> None:
        """
        The core training loop.
        """
        # clean up before training begins
        training.cleanup_before_training()

        # zero out the gradients before starting training
        self._optimizer.zero_grad()

        # Initialize tokens count and running loss (for grad accumulation)
        t0 = time.perf_counter()
        running_class_loss = 0
        running_kd_loss = 0
        num_tokens = 0

        self._profiler.start()
        # self.epochs_run should be non-zero when we're resuming from a checkpoint
        for curr_epoch in range(self.epochs_run, self.total_epochs):
            # Update the sampler to ensure data is correctly shuffled across epochs
            # in case shuffle is True

            pbar = tqdm(total=self._steps_per_epoch, disable=not (self.rank == 0))
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
                    self._is_rank_zero
                    and curr_epoch == 0
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

                class_loss, kd_loss = self._loss_step(batch)
                running_class_loss += class_loss * current_num_tokens
                running_kd_loss += kd_loss * current_num_tokens
                current_loss = (
                    1 - self._kd_ratio
                ) * class_loss + self._kd_ratio * kd_loss
                current_loss.backward()

                # Step with optimizer
                if (idx + 1) % self._gradient_accumulation_steps == 0:
                    # Get total number of tokens across all ranks to normalize gradients
                    torch.distributed.all_reduce(num_tokens)
                    # This will ensure that the logged loss matches what we're optimizing
                    torch.distributed.all_reduce(running_class_loss)
                    torch.distributed.all_reduce(running_kd_loss)
                    # Manually scale the gradients from unnormalized loss by total # of tokens
                    # We multiply by world_size to undo FSDP2 gradient normalization.
                    training.scale_grads(self._model, self.world_size / num_tokens)
                    class_loss_to_log = running_class_loss.detach().item() / num_tokens
                    kd_loss_to_log = running_kd_loss.detach().item() / num_tokens
                    self._optimizer.step()
                    self._optimizer.zero_grad(set_to_none=True)
                    self._lr_scheduler.step()
                    # Update the number of steps when the weights are updated
                    self.global_step += 1

                    class_loss_to_log = class_loss.detach().item()
                    kd_loss_to_log = kd_loss.detach().item()
                    loss_to_log = (
                        1 - self._kd_ratio
                    ) * class_loss_to_log + self._kd_ratio * kd_loss_to_log
                    pbar.update(1)
                    pbar.set_description(
                        f"{curr_epoch + 1}|{self.global_step}|Loss: {loss_to_log}"
                    )

                    # Log per-step metrics
                    if (
                        self.global_step % self._log_every_n_steps == 0
                        and self._is_rank_zero
                    ):
                        time_per_step = time.perf_counter() - t0
                        log_dict = {
                            "loss": loss_to_log,
                            "class_loss": class_loss_to_log,
                            "kd_loss": kd_loss_to_log,
                            "lr": self._optimizer.param_groups[0]["lr"],
                            "tokens_per_second_per_gpu": num_tokens
                            / (time_per_step * self.world_size),
                        }
                        if self._log_peak_memory_stats:
                            log_dict.update(
                                training.get_memory_stats(device=self._device)
                            )
                        self._metric_logger.log_dict(
                            log_dict,
                            step=self.global_step,
                        )

                    # If not last checkpoint
                    if (
                        self.global_step % self.save_every_n_steps == 0
                        and curr_epoch != self.total_epochs - 1
                    ):
                        self.save_checkpoint(epoch=curr_epoch, full_tensors=False)

                    # Reset running stats for the next step
                    running_class_loss = 0
                    running_kd_loss = 0
                    num_tokens = 0
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

                # Step the profiler
                # Note we are stepping each batch, which might not include optimizer step in the trace
                # if the schedule cycle doesn't align with gradient accumulation.
                self._profiler.step()

            self.epochs_run += 1

        self._profiler.stop()

        # Save final non-distributed ckpt
        self.save_checkpoint(epoch=curr_epoch, full_tensors=True)

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
    if cfg.get("fsdp_cpu_offload", False):
        # Utilize all available CPU cores for intra-op parallelism. This provides ~2x
        # speed up when benchmarking fused AdamW on CPU
        training.set_torch_num_threads()

    recipe = KDRecipeDistributed(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())
