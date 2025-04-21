# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
import time
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union
from warnings import warn

import torch
from omegaconf import DictConfig, ListConfig
from torch import nn
from torch.distributed import destroy_process_group, init_process_group
from torch.distributed._tensor import DTensor
from torch.distributed.tensor.parallel import parallelize_module
from torch.optim import Optimizer
from torchao.float8 import precompute_float8_dynamic_scale_for_fsdp
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler
from torchtune import config, modules, rlhf, training, utils
from torchtune.data import CROSS_ENTROPY_IGNORE_IDX, padded_collate_dpo
from torchtune.datasets import ConcatDataset
from torchtune.recipe_interfaces import FTRecipeInterface
from torchtune.rlhf import ChosenRejectedOutputs
from torchtune.training import disable_dropout, DummyProfiler, PROFILER_KEY
from torchtune.training.checkpointing._checkpoint_client import (
    CheckpointClient,
    TrainingProgress,
)
from torchtune.training.lr_schedulers import get_lr
from torchtune.training.quantization import (
    convert_to_float8_training,
    is_fp8_tensorwise_scaling,
)
from tqdm import tqdm

log = utils.get_logger("DEBUG")


class FullDPORecipeDistributed(FTRecipeInterface):
    """
    Full DPO finetuning recipe for dense transformer-based LLMs such as Llama2. This recipe supports
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

        - Gradient Clipping. Gradient clipping is supported using the ``clip_grad_norm`` flag. By default,
            ``clip_grad_norm`` is set to ``None``. If you only want to log the grad norm, you can set
            ``clip_grad_norm='inf'``.

        - Logging. Terminal, Disk, WandB and TensorBoard are all supported.

    The following losses are supported in this recipe:
        - :class:`~torchtune.modules.rlhf.loss.DPOLoss`: Direct Preference Optimization (DPO).
        - :class:`~torchtune.rlhf.loss.RSOPLoss`: Rejection Sampling Optimization (RSO).



    For a full list of example configs for this recipe, run ``tune ls`` on the command line. Each config
    has example commands for how to kick-off training.

    Args:
        cfg (DictConfig): OmegaConf object parsed from yaml file

    Raises:
        ValueError: If ``dtype`` is set to fp16.
        RuntimeError: If ``dtype`` is set to bf16 and the hardware does not support bf16.
        RuntimeError: If ``enable_activation_offloading`` is True and device is not CUDA.
        RuntimeError: If ``enable_activation_offloading`` is True and ``enable_activation_checkpointing`` is False.
    """

    def __init__(self, cfg: DictConfig) -> None:
        device_type = cfg.device
        self._device = utils.get_device(device=device_type)
        self._dtype = training.get_dtype(cfg.dtype, device=self._device)

        if self._dtype == torch.float16:
            raise ValueError(
                "full fp16 training is not supported with this recipe. Please use bf16 or fp32 instead."
            )

        # Set up the backend for distributed training (NCCL, GLOO, etc.)
        self._enable_async_checkpointing = cfg.get("enable_async_checkpointing", False)
        self.fsdp_cpu_offload = cfg.get("fsdp_cpu_offload", False)
        self.distributed_backend = training.get_distributed_backend(
            device_type,
            offload_ops_to_cpu=self.fsdp_cpu_offload
            or self._enable_async_checkpointing,
        )
        init_process_group(self.distributed_backend)

        # Initialize distributed variables
        self.world_size, self.rank = utils.get_world_size_and_rank()
        self._is_rank_zero = self.rank == 0
        self.tp_plan = cfg.get("tensor_parallel_plan", None)
        self.tp_degree = cfg.get("tensor_parallel_dim", 1)
        if self.tp_degree > 1 and self.tp_plan is None:
            raise ValueError(
                "Tensor Parallel plan needs to be provided when tensor parallel is enabled."
            )
        data_shard = cfg.get("data_parallel_shard_dim", -1)  # -1 means to infer
        data_replicate = cfg.get("data_parallel_replicate_dim", 1)

        # Set up n-d device mesh
        self.parallel_dims = training.ParallelDims(
            dp_replicate=data_replicate,
            dp_shard=data_shard,
            tp=self.tp_degree,
            world_size=self.world_size,
        )
        self.world_mesh = self.parallel_dims.build_mesh(device_type=device_type)
        if self.parallel_dims.dp_enabled:
            dp_mesh = self.world_mesh["dp"]
            self.dp_degree, self.dp_rank = (
                dp_mesh.size(),
                dp_mesh.get_local_rank(),
            )
        else:
            self.dp_degree, self.dp_rank = 1, 0

        # logging attributes
        self._output_dir = cfg.output_dir
        self._log_every_n_steps = cfg.get("log_every_n_steps", 1)
        self._log_peak_memory_stats = cfg.get("log_peak_memory_stats", False)
        if self._log_peak_memory_stats and device_type != "cuda":
            log.info(
                "log_peak_memory_stats was set to True, however, training does not use cuda. Setting log_peak_memory_stats=False."
            )
            self._log_peak_memory_stats = False

        # Training cfg
        self._resume_from_checkpoint = cfg.resume_from_checkpoint
        self._gradient_accumulation_steps = cfg.gradient_accumulation_steps
        self._clip_grad_norm = cfg.get("clip_grad_norm", None)
        self._checkpoint_client = CheckpointClient(cfg)
        self._enable_fp8_training = cfg.get("enable_fp8_training", False)
        self._fp8_recipe_name = cfg.get("fp8_recipe_name", None)

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
        elif self._enable_activation_checkpointing:
            utils.log_rank_zero(
                log,
                "Hint: enable_activation_checkpointing is True, but enable_activation_offloading isn't. "
                "Enabling activation offloading should reduce memory further.",
            )

        # These attributes constitute the recipe state and are updated by ``load_checkpoint``
        # when ``resume_from_checkpoint`` is ``True``
        self.seed = training.set_seed(
            seed=cfg.seed, debug_mode=cfg.get("cudnn_deterministic_mode", None)
        )
        self.epochs_run = 0
        self.total_epochs = cfg.epochs
        self.max_steps_per_epoch = cfg.max_steps_per_epoch
        self.global_step = 0

    def _load_ref_checkpoint(self, cfg_ref_checkpointer: DictConfig) -> Dict[str, Any]:
        """
        Extract the reference model checkpoint state from file.
        """
        _ref_checkpointer = config.instantiate(
            cfg_ref_checkpointer, should_load_recipe_state=False
        )
        checkpoint_dict = _ref_checkpointer.load_checkpoint()
        return checkpoint_dict[training.MODEL_KEY]

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

    def setup(self, cfg: DictConfig) -> None:
        """
        Setup the recipe state. This includes recipe state (if resume_from_checkpoint is True),
        model, tokenizer, loss, optimizer, learning rate scheduler, sampler, and dataloader.
        """
        if self.fsdp_cpu_offload:
            # Utilize all available CPU cores for intra-op parallelism. This provides ~2x
            # speed up when benchmarking fused AdamW on CPU
            training.set_torch_num_threads()

        if self._is_rank_zero:
            self._metric_logger = config.instantiate(cfg.metric_logger)

            # log config with parameter override
            self._metric_logger.log_config(cfg)

        # Load the base model
        checkpoint_dict = self._checkpoint_client.load_base_checkpoint()
        ref_checkpoint_dict = self._load_ref_checkpoint(cfg.ref_checkpointer)

        if self._resume_from_checkpoint:
            # If async checkpointing is enabled, intermediate checkpoints are saved asynchronously
            # using the DistributedCheckpointer.
            # Therefore the recipe needs to load the distributed checkpoint to restore the training
            # progress.
            if self._enable_async_checkpointing:
                try:
                    checkpoint_dict = (
                        self._checkpoint_client.load_distributed_checkpoint(
                            self._model,
                            self._optimizer,
                        )
                    )
                except Exception as e:
                    log.warning(
                        f"Failed to load distributed checkpoint: {e}. Training will start from the base checkpoint."
                    )

            # Update the recipe state from the checkpoint state dict.
            self._update_recipe_state(checkpoint_dict)

        self._compile = cfg.get("compile", False)
        self._model = self._setup_model(
            cfg_model=cfg.model,
            model_state_dict=checkpoint_dict[training.MODEL_KEY],
            is_reference_model=False,
            enable_activation_checkpointing=self._enable_activation_checkpointing,
            enable_activation_offloading=self._enable_activation_offloading,
            fsdp_cpu_offload=cfg.get("fsdp_cpu_offload", False),
            reshard_after_forward=cfg.get("fsdp_reshard_after_forward", True),
            custom_sharded_layers=cfg.get("custom_sharded_layers", None),
        )

        self._ref_model = self._setup_model(
            cfg_model=cfg.model,
            model_state_dict=ref_checkpoint_dict,
            is_reference_model=True,
            fsdp_cpu_offload=False,
            reshard_after_forward=False,
            custom_sharded_layers=None,
        )

        self._tokenizer = config.instantiate(cfg.tokenizer)

        self._optimizer = self._setup_optimizer(
            cfg_optimizer=cfg.optimizer,
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

        if self._is_rank_zero:
            log.info("Loss is initialized.")

        # sampler and dataloader depend on the tokenizer and loss_fn and should be
        # setup after all of these are initialized
        self._dataloader = self._setup_data(
            cfg_dataset=cfg.dataset,
            shuffle=cfg.shuffle,
            batch_size=cfg.batch_size,
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

        if self._is_rank_zero:
            log.info(f" Profiler config after instantiation: {profiler_cfg}")

            self.profiler_profile_memory = profiler_cfg.get("profile_memory", False)
            if profiler_cfg["enabled"]:
                self.profiler_wait_steps = profiler_cfg["wait_steps"]
                self.profiler_warmup_steps = profiler_cfg["warmup_steps"]
                self.profiler_active_steps = profiler_cfg["active_steps"]

        return profiler

    def _setup_model(
        self,
        cfg_model: DictConfig,
        model_state_dict: Dict[str, Any],
        is_reference_model: bool = False,
        enable_activation_checkpointing: bool = False,
        enable_activation_offloading: bool = False,
        fsdp_cpu_offload: bool = False,
        reshard_after_forward: bool = True,
        custom_sharded_layers: Optional[List[str]] = None,
    ) -> nn.Module:
        """
        Model initialization has some important considerations:
           a. To minimize GPU peak memory, we initialize the model on meta device with
              the right dtype
           b. All ranks calls ``load_state_dict`` without peaking CPU RAMs since
              full state dicts are loaded with ``torch.load(mmap=True)``

        Args:
            cfg_model: Model configuration
            model_state_dict: Model state dictionary to load
            is_reference_model: Whether this is a reference model (inference-only)
            enable_activation_checkpointing: Whether to enable activation checkpointing
            enable_activation_offloading: Whether to enable activation offloading
            fsdp_cpu_offload: Whether to offload FSDP parameters to CPU (only for base model)
            reshard_after_forward: Whether to reshard parameters after forward pass (only for base model)
            custom_sharded_layers: Custom layers to shard (only for base model)

        Returns:
            Initialized model
        """
        model_type = "reference model" if is_reference_model else "model"
        utils.log_rank_zero(
            log,
            f"Distributed training is enabled. Instantiating {model_type} and loading checkpoint on Rank 0 ...",
        )
        init_start = time.perf_counter()

        with training.set_default_dtype(self._dtype), torch.device("meta"):
            model = config.instantiate(cfg_model)

        if self._compile:
            training.compile_model(model, verbose=self._is_rank_zero)

        # Only apply FP8 training to the base model, not the reference model
        if not is_reference_model and self._enable_fp8_training:
            # Requires https://github.com/pytorch/pytorch/pull/148922
            if torch.__version__ < "2.8.0.dev20250318":
                raise RuntimeError(
                    "Float8 fine-tuning requires PyTorch 2.8.0.dev20250318 or later."
                )
            if self.tp_plan is not None:
                raise ValueError(
                    "FP8 training does not support tensor parallelism yet. "
                    "This will be enabled in the near future."
                )
            model = convert_to_float8_training(model, self._fp8_recipe_name)

        # Apply tensor parallelism to the model (for both base and reference models)
        if self.parallel_dims.tp_enabled:
            if (
                not is_reference_model
                and not self.parallel_dims.dp_enabled
                and fsdp_cpu_offload
            ):
                raise ValueError(
                    "Tensor parallelism is not supported with FSDP CPU offloading when data parallelism is disabled."
                )
            # Use the local number (num_heads, num_kv_heads, embed_dim) to account for tensor parallel
            model = training.prepare_mha_for_tp(model, self.world_mesh["tp"])
            if self.tp_plan is not None:
                self.tp_plan = config.instantiate(
                    self.tp_plan,
                    model=model,
                )
            parallelize_module(
                model,
                self.world_mesh["tp"],
                parallelize_plan=self.tp_plan,
            )

        # Apply activation checkpointing if enabled (only for base model)
        if not is_reference_model and enable_activation_checkpointing:
            training.set_activation_checkpointing(
                model, auto_wrap_policy={modules.TransformerSelfAttentionLayer}
            )

        # Apply Fully Sharded Data Parallelism to the model (only for base model)
        if not is_reference_model and self.parallel_dims.dp_shard_enabled:
            fsdp_shard_conditions = [
                partial(
                    training.get_shard_conditions,
                    names_to_match=custom_sharded_layers,
                )
            ]

            if self.parallel_dims.dp_replicate_enabled:
                dp_mesh_dim_names = ("dp_replicate", "dp_shard")
            else:
                dp_mesh_dim_names = ("dp_shard",)

            training.shard_model(
                model=model,
                shard_conditions=fsdp_shard_conditions,
                cpu_offload=fsdp_cpu_offload,
                reshard_after_forward=reshard_after_forward,
                dp_mesh=self.world_mesh[dp_mesh_dim_names],
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

        # Set up activation offloading context for the base model
        if not is_reference_model:
            self.activations_handling_ctx = training.get_act_offloading_ctx_manager(
                model, enable_activation_offloading
            )

        # Ensure no params and buffers are on meta device
        training.validate_no_params_on_meta_device(model)

        utils.log_rank_zero(
            log,
            f"Instantiating {model_type} and loading checkpoint took {time.perf_counter() - init_start:.2f} secs",
        )

        if self._is_rank_zero:
            memory_stats = training.get_memory_stats(device=self._device)
            training.log_memory_stats(memory_stats)

        # disabling dropout if found - non-determinism leads to issues in e.g. comparing logprobs
        # between ref policy and current policy
        disable_dropout(model)

        # For reference model, set to eval mode and disable gradients
        if is_reference_model:
            for p in model.parameters():
                p.requires_grad = False
            model.eval()

        # synchronize before training begins
        torch.distributed.barrier()

        return model

    def _setup_optimizer(
        self,
        cfg_optimizer: DictConfig,
        opt_state_dict: Optional[Dict[str, Any]] = None,
    ) -> Optimizer:
        optimizer = config.instantiate(cfg_optimizer, self._model.parameters())
        if opt_state_dict:
            training.load_from_full_optimizer_state_dict(
                self._model,
                optimizer,
                opt_state_dict,
                self._device,
            )

        utils.log_rank_zero(log, "Optimizer and loss are initialized.")
        return optimizer

    def _setup_lr_scheduler(
        self,
        cfg_lr_scheduler: Optional[DictConfig],
        num_training_steps: int,
        last_epoch: int,
    ) -> Optional[Optimizer]:
        """
        Set up the learning rate scheduler based on the provided configuration.

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

        # Instantiate the learning rate scheduler
        lr_scheduler = config.instantiate(
            cfg_lr_scheduler,
            self._optimizer,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch,
        )

        if self._is_rank_zero:
            log.info("Learning rate scheduler is initialized.")

        return lr_scheduler

    def _setup_data(
        self,
        cfg_dataset: DictConfig,
        shuffle: bool,
        batch_size: int,
    ) -> StatefulDataLoader:
        """
        All data related setup happens here. This recipe currently supports only
        map-style datasets. If a state_dict is provided (meaning we are resuming a training run),
        it is loaded into the dataloader.
        """

        if isinstance(cfg_dataset, ListConfig):
            datasets = [
                config.instantiate(single_cfg_dataset, tokenizer=self._tokenizer)
                for single_cfg_dataset in cfg_dataset
            ]
            ds = ConcatDataset(datasets=datasets)
        else:
            ds = config.instantiate(cfg_dataset, tokenizer=self._tokenizer)

        sampler = StatefulDistributedSampler(
            ds, num_replicas=self.dp_degree, rank=self.dp_rank, shuffle=shuffle
        )

        dataloader = StatefulDataLoader(
            dataset=ds,
            batch_size=batch_size,
            sampler=sampler,
            # dropping last avoids shape issues with compile + flex attention
            drop_last=True,
            collate_fn=partial(
                padded_collate_dpo,
                padding_idx=self._tokenizer.pad_id,
                ignore_idx=CROSS_ENTROPY_IGNORE_IDX,
            ),
        )

        if self._is_rank_zero:
            log.info("Dataset and Sampler are initialized.")

        return dataloader

    def save_checkpoint(
        self,
        epoch: int,
    ) -> None:
        """
        Checkpoint the state of the recipe using the CheckpointClient.
        """
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
        )

    def concatenated_forward(
        self,
        model: nn.Module,
        batch: Tuple[torch.Tensor, torch.Tensor],
        activations_handling: Optional[bool] = True,
    ) -> ChosenRejectedOutputs:
        """
        Run forward pass of the model with chosen and rejected samples concatenated.

        Args:
            model (nn.Module): The model to be used for the forward pass.
            batch (Tuple[torch.Tensor, torch.Tensor]): Tuple of input_ids and labels.

        Returns:
            Dataclass of chosen log probs, rejected log probs, chosen logits, rejected logits.
        """
        concatenated_input_ids, concatenated_labels = batch
        concatenated_input_ids = concatenated_input_ids.to(self._device)
        concatenated_labels = concatenated_labels.to(self._device)

        # formed by concatenating an equal number of "chosen" and "rejected".
        len_chosen = concatenated_input_ids.shape[0] // 2

        if activations_handling:
            with self.activations_handling_ctx:
                all_logits = model(concatenated_input_ids)
        else:
            all_logits = model(concatenated_input_ids)

        chosen_log_probs = rlhf.get_batch_log_probs(
            all_logits[:len_chosen],
            concatenated_labels[:len_chosen],
            return_average_logprobs=False,
        )

        rejected_log_probs = rlhf.get_batch_log_probs(
            all_logits[len_chosen:],
            concatenated_labels[len_chosen:],
            return_average_logprobs=False,
        )

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        return ChosenRejectedOutputs(
            chosen_log_probs, rejected_log_probs, chosen_logits, rejected_logits
        )

    def train(self) -> None:
        """
        The core training loop. Supports training on subsets of the dataset using the
        ``max_steps_per_epoch``.
        """
        # clean up before training begins
        training.cleanup_before_training()

        # zero out the gradients before starting training
        self._optimizer.zero_grad()

        # Initialize tokens count and running loss (for grad accumulation)
        t0 = time.perf_counter()

        # Running metrics
        running_loss = torch.tensor(0.0, device=self._device)
        running_metrics = {
            "rewards/chosen": torch.tensor(0.0, device=self._device),
            "rewards/rejected": torch.tensor(0.0, device=self._device),
            "rewards/accuracies": torch.tensor(0.0, device=self._device),
            "log_probs/chosen": torch.tensor(0.0, device=self._device),
            "log_probs/rejected": torch.tensor(0.0, device=self._device),
            "logits/chosen": torch.tensor(0.0, device=self._device),
            "logits/rejected": torch.tensor(0.0, device=self._device),
        }
        num_tokens = torch.tensor(0, device=self._device)

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

                # batch is input_ids, labels
                num_tokens += torch.tensor(batch[0].numel(), device=self._device)
                policy_chosen_rejected_outputs = self.concatenated_forward(
                    self._model, batch
                )

                policy_chosen_logits_mean = (
                    policy_chosen_rejected_outputs.chosen_logits.detach().mean()
                )
                policy_rejected_logits_mean = (
                    policy_chosen_rejected_outputs.rejected_logits.detach().mean()
                )

                # deleting logits here helps reduce (peak) memory usage - we only need them for metric logging
                del (
                    policy_chosen_rejected_outputs.chosen_logits,
                    policy_chosen_rejected_outputs.rejected_logits,
                )

                with torch.no_grad():
                    reference_chosen_rejected_outputs = self.concatenated_forward(
                        self._ref_model, batch, activations_handling=False
                    )

                del (
                    reference_chosen_rejected_outputs.chosen_logits,
                    reference_chosen_rejected_outputs.rejected_logits,
                )

                loss, chosen_rewards, rejected_rewards = self._loss_fn(
                    policy_chosen_rejected_outputs, reference_chosen_rejected_outputs
                )
                reward_accuracies = (chosen_rewards > rejected_rewards).float()

                loss = loss.mean()

                loss = loss / self._gradient_accumulation_steps

                # Update running metrics
                running_loss += loss
                scaling_factor = (
                    1 / self._gradient_accumulation_steps
                )  # to average out between grad_acc steps
                running_metrics["rewards/chosen"] += (
                    scaling_factor * chosen_rewards.mean()
                )
                running_metrics["rewards/rejected"] += (
                    scaling_factor * rejected_rewards.mean()
                )
                running_metrics["rewards/accuracies"] += (
                    scaling_factor * reward_accuracies.mean()
                )
                running_metrics["log_probs/chosen"] += (
                    scaling_factor
                    * policy_chosen_rejected_outputs.chosen_logps.detach().mean()
                )
                running_metrics["log_probs/rejected"] += (
                    scaling_factor
                    * policy_chosen_rejected_outputs.rejected_logps.detach().mean()
                )
                running_metrics["logits/chosen"] += (
                    scaling_factor * policy_chosen_logits_mean
                )
                running_metrics["logits/rejected"] += (
                    scaling_factor * policy_rejected_logits_mean
                )

                loss.backward()

                # Step with optimizer
                if (idx + 1) % self._gradient_accumulation_steps == 0:
                    # Accumulate running metrics across all devices
                    torch.distributed.all_reduce(
                        running_loss, op=torch.distributed.ReduceOp.AVG
                    )
                    torch.distributed.all_reduce(num_tokens)

                    for key in running_metrics:
                        torch.distributed.all_reduce(
                            running_metrics[key], op=torch.distributed.ReduceOp.AVG
                        )
                    if self._clip_grad_norm is not None:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self._model.parameters(),
                            max_norm=float(self._clip_grad_norm),
                        )
                        # If sharded, collect the DTensor here
                        if isinstance(grad_norm, DTensor):
                            grad_norm = grad_norm.full_tensor()
                    self._optimizer.step()
                    self._optimizer.zero_grad(set_to_none=True)

                    # Update the number of steps when the weights are updated
                    self.global_step += 1
                    # If float8 training is enabled, perform a single all-reduce to compute the
                    # scale for all float8 parameters efficiently instead of doing many small
                    # all-reduces for each parameter
                    if (
                        self._enable_fp8_training
                        and is_fp8_tensorwise_scaling(self._fp8_recipe_name)
                        and self.dp_degree > 1
                    ):
                        precompute_float8_dynamic_scale_for_fsdp(self._model)

                    # Step the learning rate scheduler
                    if self._lr_scheduler is not None:
                        self._lr_scheduler.step()
                    loss_to_log = running_loss.detach().item()
                    pbar.update(1)
                    pbar.set_description(
                        f"{curr_epoch + 1}|{self.global_step}|Loss: {loss_to_log}"
                    )

                    # Log per-step metrics
                    if (
                        self.global_step % self._log_every_n_steps == 0
                        and self._is_rank_zero
                    ):
                        self._log_training_metrics(
                            loss_to_log=loss_to_log,
                            running_metrics=running_metrics,
                            num_tokens=num_tokens,
                            time_per_step=time.perf_counter() - t0,
                            grad_norm=(
                                grad_norm if self._clip_grad_norm is not None else None
                            ),
                        )

                    # Reset running stats for the next step
                    running_loss = torch.tensor(0.0, device=self._device)
                    running_metrics = {
                        key: torch.tensor(0.0, device=self._device)
                        for key in running_metrics
                    }
                    num_tokens = torch.tensor(0, device=self._device)

                    t0 = time.perf_counter()

                    # Step profiler
                    # Note that this is called within gradient accumulation block, hence
                    # will include multiple forward / backward passes if gradient accumulation > 1
                    self._profiler.step()

            self.epochs_run += 1
            self.save_checkpoint(epoch=curr_epoch)

        self._profiler.stop()

    def _log_training_metrics(
        self,
        loss_to_log: float,
        running_metrics: Dict[str, torch.Tensor],
        num_tokens: torch.Tensor,
        time_per_step: float,
        grad_norm: Optional[torch.Tensor] = None,
    ) -> None:
        """Log training metrics to the metric logger."""
        log_dict = {
            "loss": loss_to_log,
            "lr": get_lr(self._optimizer),
            "tokens_per_second_per_gpu": num_tokens / (time_per_step * self.world_size),
            "rewards/chosen": running_metrics["rewards/chosen"].cpu(),
            "rewards/rejected": running_metrics["rewards/rejected"].cpu(),
            "rewards/accuracies": running_metrics["rewards/accuracies"].cpu(),
            "rewards/margins": (
                running_metrics["rewards/chosen"] - running_metrics["rewards/rejected"]
            ).cpu(),
            "log_probs/chosen": running_metrics["log_probs/chosen"].cpu(),
            "log_probs/rejected": running_metrics["log_probs/rejected"].cpu(),
            "logits/chosen": running_metrics["logits/chosen"].cpu(),
            "logits/rejected": running_metrics["logits/rejected"].cpu(),
        }

        if grad_norm is not None:
            log_dict.update({"grad_norm": grad_norm})

        if self._log_peak_memory_stats:
            log_dict.update(training.get_memory_stats(device=self._device))

        self._metric_logger.log_dict(
            log_dict,
            step=self.global_step,
        )

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
    config.log_config(recipe_name="FullDPORecipeDistributed", cfg=cfg)
    recipe = FullDPORecipeDistributed(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())
