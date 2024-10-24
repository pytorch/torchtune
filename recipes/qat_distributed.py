# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import time

from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union
from warnings import warn

import torch
from omegaconf import DictConfig, ListConfig

from torch import nn
from torch.distributed import destroy_process_group, init_process_group

from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler
from torchtune import config, modules, training, utils
from torchtune.data import padded_collate_packed, padded_collate_sft
from torchtune.datasets import ConcatDataset
from torchtune.recipe_interfaces import FTRecipeInterface
from torchtune.training import DummyProfiler, PROFILER_KEY
from torchtune.training.activations import apply_selective_activation_checkpointing

from tqdm import tqdm

log = utils.get_logger("DEBUG")


class QATRecipeDistributed(FTRecipeInterface):
    """
    Quantization-aware training (QAT) recipe for dense transformer-based LLMs such as Llama2.
    This recipe supports distributed training and can be run on a single node (1 to 8 GPUs).
    Only compatible with PyTorch 2.4+.

    Features:
        - Quantization-aware training (QAT). Perform fake quantization on weights and/or activations
            during finetuning, with the goal of ultimately producing a quantized model with minimal
            accuracy degradation. This recipe produces an unquantized model in the original dtype,
            which can then be quantized separately.

        - Delayed fake quantization. Optionally specify the step after which fake quantization occurs.
            Empirically, allowing the model to finetune without fake quantization initially allows the
            weight and activation values to stabilize before fake quantizing them, potentially leading
            to improved quantized accuracy. This can be specified through ``fake_quant_after_n_steps``.

        - FSDP. Supported using PyTorch's FSDP APIs. CPU offload of parameters, gradients, and optimizer states
            is supported via the ``fsdp_cpu_offload``. Resharding of parameters after the forward pass is
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
        self._log_every_n_steps = cfg.get("log_every_n_steps", 1)
        self._log_peak_memory_stats = cfg.get("log_peak_memory_stats", False)

        # _is_rank_zero is used primarily for logging. In the future, the logger
        # should directly take care of this
        _, rank = training.get_world_size_and_rank()
        self._is_rank_zero = rank == 0

        # Training cfg
        self._resume_from_checkpoint = cfg.resume_from_checkpoint
        self._gradient_accumulation_steps = cfg.gradient_accumulation_steps
        self._fsdp_sharding_strategy = torch.distributed.fsdp.ShardingStrategy[
            cfg.get("fsdp_sharding_strategy", "FULL_SHARD")
        ]
        self._fake_quant_after_n_steps = cfg.get("fake_quant_after_n_steps", None)
        self._quantizer_mode = None

        # These are public properties which are updated by the checkpoint loader
        # when ``resume_from_checkpoint`` is `True` or validated in tests
        self.seed = training.set_seed(seed=cfg.seed)
        self.epochs_run = 0
        self.total_epochs = cfg.epochs
        self.max_steps_per_epoch = cfg.max_steps_per_epoch
        self.global_step = 0

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
        Setup the recipe. This includes training state (if resume_from_checkpoint is True),
        model, tokenizer, loss, optimizer, sampler, and dataloader.
        """
        if self._is_rank_zero:
            self._metric_logger = config.instantiate(cfg.metric_logger)

            # log config with parameter override
            self._metric_logger.log_config(cfg)

        checkpoint_dict = self.load_checkpoint(cfg_checkpointer=cfg.checkpointer)

        self._model_compile = cfg.get("compile", False)
        self._model = self._setup_model(
            cfg_model=cfg.model,
            enable_activation_checkpointing=cfg.enable_activation_checkpointing,
            custom_sharded_layers=cfg.get("custom_sharded_layers", None),
            fsdp_cpu_offload=cfg.get("fsdp_cpu_offload", False),
            reshard_after_forward=cfg.get("fsdp_reshard_after_forward", True),
            model_state_dict=checkpoint_dict[training.MODEL_KEY],
            ac_mode=cfg.get("ac_mode", None),
            ac_option=cfg.get("ac_option", None),
            quantizer_cfg=cfg.get("quantizer", None),
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
        backend = os.environ.get("TORCH_COMPILE_BACKEND", "inductor")
        if self._loss_fn.__class__.__name__ == "CEWithChunkedOutputLoss":
            # set num_output_chunks for model
            self._model.set_num_output_chunks(self._loss_fn.num_output_chunks)
            if self._model_compile:
                log.info("Compiling loss with torch.compile...")
                # For CEWithChunkedOutputLoss, if we compile the entire class
                # we lose the benefits from the chunked loss.
                # Therefore, we only compile the cross entropy function + upcasting
                self._loss_fn.compute_cross_entropy = torch.compile(
                    self._loss_fn.compute_cross_entropy, backend=backend
                )
        else:
            if self._model_compile:
                log.info("Compiling loss with torch.compile...")
                self._loss_fn = torch.compile(self._loss_fn, backend=backend)
        log.info("Loss is initialized.")

        # sampler and dataloader depend on the tokenizer and loss_fn and should be
        # setup after both of these are initialized
        self._sampler, self._dataloader = self._setup_data(
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

        # Set up profiler, returns DummyProfiler (nullcontext object with no-op `step` method)
        # if cfg is missing profiler key or if `cfg.profiler.enabled = False`
        self._profiler = self._setup_profiler(cfg.get(PROFILER_KEY, None))

        # Used to ignore labels for loss computation
        self.ignore_labels_cache = torch.full(
            (cfg.batch_size, 1), self._loss_fn.ignore_index, device=self._device
        )

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
        enable_activation_checkpointing: bool,
        fsdp_cpu_offload: bool,
        reshard_after_forward: bool,
        model_state_dict: Dict[str, Any],
        custom_sharded_layers: Optional[List[str]] = None,
        ac_mode: Optional[str] = None,
        ac_option: Optional[int] = None,
        quantizer_cfg: Optional[DictConfig] = None,
    ) -> nn.Module:
        """
        Model initialization has some important considerations:
           a. To minimize GPU peak memory, we initialize the model on meta device with
              the right dtype
           b. All ranks calls ``load_state_dict`` without peaking CPU RAMs since
              full state dicts are loaded with ``torch.load(mmap=True)``
        """

        if self._is_rank_zero:
            log.info(
                "FSDP is enabled. Instantiating model and loading checkpoint on Rank 0 ..."
            )
            init_start = time.perf_counter()

        with training.set_default_dtype(self._dtype), torch.device("meta"):
            model = config.instantiate(cfg_model)

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
            model, model_state_dict, self._device, self._is_rank_zero, strict=True
        )

        # Ensure no params and buffers are on meta device
        training.validate_no_params_on_meta_device(model)

        if self._is_rank_zero:
            log.info(
                f"Instantiating model and loading checkpoint took {time.perf_counter() - init_start:.2f} secs"
            )
            memory_stats = training.get_memory_stats(device=self._device)
            training.log_memory_stats(memory_stats)

        # synchronize before training begins
        torch.distributed.barrier()

        return model

    def _setup_optimizer(
        self, cfg_optimizer: DictConfig, opt_state_dict: Optional[Dict[str, Any]] = None
    ) -> Optimizer:
        optimizer = config.instantiate(cfg_optimizer, self._model.parameters())
        if opt_state_dict:
            training.load_from_full_optimizer_state_dict(
                optimizer,
                opt_state_dict,
                self._device,
            )

        if self._is_rank_zero:
            log.info("Optimizer is initialized.")
        return optimizer

    def _setup_data(
        self,
        cfg_dataset: DictConfig,
        shuffle: bool,
        batch_size: int,
    ) -> Tuple[DistributedSampler, DataLoader]:
        """
        All data related setup happens here. Currently this recipe only supports the
        DistributedSamplers with Map-style Datasets which fit into memory. Other samplers,
        iterable datasets and streaming datasets are not supported.
        """
        world_size, rank = training.get_world_size_and_rank()

        if isinstance(cfg_dataset, ListConfig):
            datasets = [
                config.instantiate(single_cfg_dataset, tokenizer=self._tokenizer)
                for single_cfg_dataset in cfg_dataset
            ]
            ds = ConcatDataset(datasets=datasets)
            packed = False
        else:
            ds = config.instantiate(cfg_dataset, tokenizer=self._tokenizer)
            packed = cfg_dataset.get("packed", False)

        sampler = DistributedSampler(
            ds, num_replicas=world_size, rank=rank, shuffle=shuffle, seed=0
        )
        dataloader = DataLoader(
            dataset=ds,
            batch_size=batch_size,
            sampler=sampler,
            # dropping last avoids shape issues with compile + flex attention
            drop_last=True,
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
        )

        if self._is_rank_zero:
            log.info("Dataset and Sampler are initialized.")

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
        # final dict passed onto the checkpointer
        checkpoint_dict = {}

        intermediate_checkpoint = epoch + 1 < self.total_epochs
        # To prevent GPU memory from spiking during checkpoint save,
        # we consolidate the full model and optim state dicts on CPU for rank 0
        cpu_state_dict = training.get_full_model_state_dict(
            self._model,
            self._is_rank_zero,
        )

        if intermediate_checkpoint:
            opt_state_dict = training.get_full_optimizer_state_dict(
                self._optimizer,
                self._is_rank_zero,
            )
        else:
            opt_state_dict = None

        # Now that we have the model and opt state dict, create the actual checkpoint dict
        # to be sent to the checkpointer and ultimately written to file
        if self._is_rank_zero:

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

    def train(self) -> None:
        """
        The core training loop.
        """
        # clean up before training begins
        training.cleanup_before_training()

        _, rank = training.get_world_size_and_rank()

        # zero out the gradients before starting training
        self._optimizer.zero_grad()

        # Initialize tokens count and running loss (for grad accumulation)
        t0 = time.perf_counter()
        running_loss = 0
        num_tokens = 0

        self._profiler.start()
        # self.epochs_run should be non-zero when we're resuming from a checkpoint
        for curr_epoch in range(self.epochs_run, self.total_epochs):

            # Update the sampler to ensure data is correctly shuffled across epochs
            # in case shuffle is True
            self._sampler.set_epoch(curr_epoch)

            pbar = tqdm(total=self._steps_per_epoch, disable=not (rank == 0))
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
                ):
                    torch.cuda.memory._record_memory_history()

                # Both are shape [b, s]
                tokens, labels = batch["tokens"], batch["labels"]
                # Get the attention mask and position ids from the dataset if they
                # exist. Currently, only sample packing in PackedDataset returns these
                mask = batch.get("mask", None)  # shape [b, s, s]
                input_pos = batch.get("input_pos", None)  # shape [b, s]

                # Optionally wait N steps before enabling fake quant
                if self._fake_quant_after_n_steps is not None:
                    if self.global_step == 0:
                        log.info(
                            "Step 0: Disabling fake quant, will re-enable in step %s"
                            % self._fake_quant_after_n_steps
                        )
                        disable_fq = training.quantization._get_disable_fake_quant(
                            self._quantizer_mode
                        )
                        self._model.apply(disable_fq)
                    elif self.global_step == self._fake_quant_after_n_steps:
                        log.info(
                            "Step %s: Enabling fake quant"
                            % self._fake_quant_after_n_steps
                        )
                        enable_fq = training.quantization._get_enable_fake_quant(
                            self._quantizer_mode
                        )
                        self._model.apply(enable_fq)

                tokens = tokens.to(self._device)
                num_tokens += tokens.numel()
                labels = labels.to(self._device)
                mask = mask.to(self._device) if mask is not None else None
                input_pos = (
                    input_pos.to(self._device) if input_pos is not None else None
                )

                logits = self._model(tokens, mask=mask, input_pos=input_pos)

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

                loss = loss / self._gradient_accumulation_steps
                running_loss += loss
                loss.backward()

                # Step with optimizer
                if (idx + 1) % self._gradient_accumulation_steps == 0:
                    self._optimizer.step()
                    self._optimizer.zero_grad(set_to_none=True)

                    # Update the number of steps when the weights are updated
                    self.global_step += 1

                    loss_to_log = running_loss.item()
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
                            "lr": self._optimizer.param_groups[0]["lr"],
                            "tokens_per_second_per_gpu": num_tokens / time_per_step,
                        }
                        if self._log_peak_memory_stats:
                            log_dict.update(
                                training.get_memory_stats(device=self._device)
                            )
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
            "Distributed QAT recipe should be run via a distributed launcher."
            "If using tune CLI, please specify --nnodes 1 and --nproc_per_node [num_gpus]"
        )
    init_process_group(backend="gloo" if cfg.device == "cpu" else "nccl")
    if cfg.get("fsdp_cpu_offload", False):
        # Utilize all available CPU cores for intra-op parallelism. This provides ~2x
        # speed up when benchmarking fused AdamW on CPU
        training.set_torch_num_threads()

    config.log_config(recipe_name="QATRecipeDistributed", cfg=cfg)

    recipe = QATRecipeDistributed(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())
