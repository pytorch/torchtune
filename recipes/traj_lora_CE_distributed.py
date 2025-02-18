# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
import sys
import time
import os

from functools import partial
from typing import Any, Dict, List, Optional, Tuple
from warnings import warn

import torch
from omegaconf import DictConfig, ListConfig
from torchtune.modules.loss import CEWithChunkedOutputLoss


from torch import nn
from torch.distributed import (
    destroy_process_group,
    init_process_group,
    all_gather,
    get_world_size,
)
from torch.optim import Optimizer
import random
from torch.utils.data import DataLoader, DistributedSampler
from torchtune import config, modules, rlhf, training, utils
from torchtune.data import CROSS_ENTROPY_IGNORE_IDX, padded_collate_dpo, padded_collate_traj_CE
from torchtune.datasets import ConcatDataset
from torchtune.modules.peft import (
    disable_adapter,
    DoRALinear,
    get_adapter_params,
    get_adapter_state_dict,
    get_merged_lora_ckpt,
    load_dora_magnitudes,
    LoRALinear,
    set_trainable_params,
    validate_missing_and_unexpected_for_lora,
)
from torchtune.recipe_interfaces import FTRecipeInterface
from torchtune.rlhf.loss import SimPOLoss
from tqdm import tqdm
import torch.distributed as dist


log = utils.get_logger("DEBUG")

"""
Summary of changes:
    - Same as usual
    - no need to add a `real_num_tokens` variable this time: 
            the `num_tokens` variable is already being updated correctly

"""


class LoRADPORecipeDistributed(FTRecipeInterface):
    """
    Distributed LoRA DPO recipe for dense transformer-based LLMs such as Llama2. This recipe supports
    distributed training and can be run on a single node (1 to 8 GPUs). This is based on HF's DPOTrainer
    in the TRL library: https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py#L65

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

                Total Batch Size = batch_size * number of GPUs * gradient accumulation steps.

            For example: with batch_size=1, nproc_per_node=2 and gradient_accumulation_steps=32 we get a
            total batch size of 64.

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

    The following losses are supported in this recipe:
        - :class:`~torchtune.rlhf.loss.DPOLoss`: Direct Preference Optimization (DPO).
        - :class:`~torchtune.rlhf.loss.RSOPLoss`: Rejection Sampling Optimization (RSO).
        - :class:`~torchtune.rlhf.loss.SimPOLoss`: Simple Preference Optimization (SimPO).

    For a full list of example configs for this recipe, run ``tune ls`` on the command line. Each config
    has example commands for how to kick-off training.

    Args:
        cfg (DictConfig): OmegaConf object parsed from yaml file

    Raises:
        ValueError: If ``dtype`` is set to fp16.
        ValueError: If world_size is 1
        RuntimeError: If ``dtype`` is set to bf16 and the hardware does not support bf16.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self._device = utils.get_device(device=cfg.device)
        self._dtype = training.get_dtype(cfg.dtype, device=self._device)

        if self._dtype == torch.float16:
            raise ValueError(
                "full fp16 training is not supported with this recipe. Please use bf16 or fp32 instead."
            )

        world_size, rank = training.get_world_size_and_rank()

        self._is_rank_zero = rank == 0

        # logging attributes
        self._output_dir = cfg.output_dir
        self._log_every_n_steps = cfg.get("log_every_n_steps", 1)
        self._log_peak_memory_stats = cfg.get("log_peak_memory_stats", False)
        self.ce_loss=CEWithChunkedOutputLoss(num_output_chunks=6)
        self.reg_lambda=cfg.reg_lambda
        if self._log_peak_memory_stats and self._device.type != "cuda":
            log.info(
                "log_peak_memory_stats was set to True, however, training does not use cuda. Setting log_peak_memory_stats=False."
            )
            self._log_peak_memory_stats = False
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
        self.seed = training.set_seed(seed=cfg.seed)
        self.epochs_run = 0
        self.total_epochs = cfg.epochs
        self.max_steps_per_epoch = cfg.max_steps_per_epoch
        self.global_step = 0
        self._resume_from_checkpoint = cfg.resume_from_checkpoint
        self._save_adapter_weights_only = cfg.get("save_adapter_weights_only", False)
        self._gradient_accumulation_steps = cfg.gradient_accumulation_steps

        # NOTE: added by us
        self.save_checkpoints_interval = cfg.get("save_checkpoints", 1)
        self.max_seq_len = cfg.get("max_seq_len", None)
        self._max_validation_steps = int(
            cfg.get("samples_per_validation_steps") / (cfg.batch_size * world_size)
        )
        log.info(
            f"Setting max validation steps to {self._max_validation_steps} (samples_per_validation_steps / batch_size)"
        )
        assert self.max_steps_per_epoch is None
        # NOTE (smurty): effective_batch_size also needs to be multipled by the number of GPUs!
        effective_batch_size = (
            cfg.batch_size * cfg.gradient_accumulation_steps * world_size
        )
        self.max_steps_per_epoch = int(
            cfg.get("samples_per_epoch") / effective_batch_size
        )
        log.info(
            f"Setting max steps per epoch to {self.max_steps_per_epoch} (samples_per_epoch / effective_batch_size)"
        )

    def load_checkpoint(self, cfg_checkpointer: DictConfig) -> Dict[str, Any]:
        """
        Extract the checkpoint state from file and validate. This includes the
        base model weights. If resume_from_checkpoint is True, this also includes
        the adapter weights and recipe state
        """
        self._checkpointer = config.instantiate(
            cfg_checkpointer,
            resume_from_checkpoint=self._resume_from_checkpoint,
        )
        checkpoint_dict = self._checkpointer.load_checkpoint()

        # When resuming from checkpoint for LoRA, the recipe expects the adapter weights
        # and recipe state to be present. The keys should match up with what ``save_checkpoint``
        # used to create these intermediate checkpoints
        if self._resume_from_checkpoint:
            if training.ADAPTER_KEY not in checkpoint_dict:
                raise ValueError(
                    "Adapter weights not found. Please ensure a valid adapter checkpoint is provided."
                )
            # _update_recipe_state will throw an exception if the recipe state is not corrctly loaded
            # no need to check here
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
        Setup the recipe state. This includes recipe state (if resume_from_checkpoint is True),
        model, tokenizer, loss, optimizer, learning rate scheduler, sampler, and dataloader.
        """
        if self._is_rank_zero:
            self._metric_logger = config.instantiate(cfg.metric_logger)

            # log config with parameter override
            self._metric_logger.log_config(cfg)

        utils.log_rank_zero(log, "metric logger is initialized.")

        checkpoint_dict = self.load_checkpoint(cfg_checkpointer=cfg.checkpointer)

        self._model = self._setup_model(
            cfg_model=cfg.model,
            enable_activation_checkpointing=cfg.enable_activation_checkpointing,
            enable_activation_offloading=self._enable_activation_offloading,
            fsdp_cpu_offload=cfg.get("fsdp_cpu_offload", False),
            reshard_after_forward=cfg.get("fsdp_reshard_after_forward", True),
            base_model_state_dict=checkpoint_dict[training.MODEL_KEY],
            lora_weights_state_dict=(
                checkpoint_dict[training.ADAPTER_KEY]
                if self._resume_from_checkpoint
                else None
            ),
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

        self._loss_fn = config.instantiate(cfg.loss)

        utils.log_rank_zero(log, "Loss is initialized.")

        # NOTE: no collate_func in this recipe

        # sampler and dataloader depend on the tokenizer and loss_fn and should be
        # setup after all of these are setup
        cfg.dataset["split"] = "train"  # NOTE: added by us
        self._sampler, self._dataloader = self._setup_data(
            cfg_dataset=cfg.dataset,
            shuffle=cfg.shuffle,
            batch_size=cfg.batch_size,
        )

        # NOTE: added by us
        # validation dataloader
        cfg["validation_dataset"] = deepcopy(cfg.dataset)
        cfg["validation_dataset"]["split"] = "validation"
        self._sampler_validation, self._dataloader_validation = self._setup_data(
            cfg_dataset=cfg["validation_dataset"],
            shuffle=cfg.shuffle,
            batch_size=cfg.batch_size,  # TODO: have a separate batch size for validation
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

        # Learning rate scheduler can only be set up after number of steps
        # has been computed
        self._lr_scheduler = self._setup_lr_scheduler(
            cfg_lr_scheduler=cfg.lr_scheduler,
            num_training_steps=self.total_epochs * self._steps_per_epoch,
            last_epoch=self.global_step - 1,
        )

    def _setup_model(
        self,
        cfg_model: DictConfig,
        enable_activation_checkpointing: bool,
        enable_activation_offloading: bool,
        fsdp_cpu_offload: bool,
        reshard_after_forward: bool,
        base_model_state_dict: Dict[str, Any],
        custom_sharded_layers: Optional[List[str]] = None,
        lora_weights_state_dict: Optional[Dict[str, Any]] = None,
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

        init_start = time.perf_counter()
        # activation offloading
        

        utils.log_rank_zero(
            log,
            "FSDP is enabled. Instantiating model and loading checkpoint on Rank 0 ...",
        )

        with training.set_default_dtype(self._dtype), torch.device("meta"):
            model = config.instantiate(cfg_model)

        # NOTE: added by us
        if self.max_seq_len is not None:
            model.max_seq_len = self.max_seq_len

        self.activations_handling_ctx = training.get_act_offloading_ctx_manager(
            model, enable_activation_offloading
        )

        self.adapter_params = get_adapter_params(model)
        set_trainable_params(model, self.adapter_params)

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
                self._is_rank_zero,
                cpu_offload=fsdp_cpu_offload,
            )
        else:
            lora_missing, lora_unexpected = None, None

        # Initialize LoRA params and RoPE buffers
        with training.set_default_dtype(self._dtype), self._device:
            lora_device = "cpu" if fsdp_cpu_offload else self._device
            for m in model.modules():
                if (
                    isinstance(m, LoRALinear) or isinstance(m, DoRALinear)
                ) and not lora_weights_state_dict:
                    # lora may not be covered in state dict
                    # if finetune for the 1st time
                    m.lora_a.to_empty(device=lora_device)
                    m.lora_b.to_empty(device=lora_device)
                    m.initialize_parameters()
                # RoPE is not covered in state dict
                if hasattr(m, "rope_init"):
                    m.rope_init()

        base_missing, base_unexpected = training.load_from_full_model_state_dict(
            model,
            base_model_state_dict,
            self._device,
            self._is_rank_zero,
            cpu_offload=fsdp_cpu_offload,
        )
        is_dora = False
        for m in model.modules():
            if hasattr(m, "initialize_dora_magnitude"):
                is_dora = True
                m.initialize_dora_magnitude()
        if is_dora:
            load_dora_magnitudes(model)
        validate_missing_and_unexpected_for_lora(
            lora_attn_modules=self._lora_attn_modules,
            apply_lora_to_mlp=self._apply_lora_to_mlp,
            apply_lora_to_output=self._apply_lora_to_output,
            base_missing=base_missing,
            base_unexpected=base_unexpected,
            lora_missing=lora_missing,
            lora_unexpected=lora_unexpected,
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
        self, cfg_optimizer: DictConfig, opt_state_dict: Optional[Dict[str, Any]] = None
    ) -> Optimizer:
        optimizer = config.instantiate(cfg_optimizer, self._model.parameters())
        if opt_state_dict:
            training.load_from_full_optimizer_state_dict(
                optimizer,
                opt_state_dict,
                self._device,
            )

        utils.log_rank_zero(log, "Optimizer and loss are initialized.")
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

        utils.log_rank_zero(log, "Learning rate scheduler is initialized.")
        return lr_scheduler

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
        else:
            ds = config.instantiate(cfg_dataset, tokenizer=self._tokenizer)

        sampler = DistributedSampler(
            ds, num_replicas=world_size, rank=rank, shuffle=shuffle, seed=0
        )

        dataloader = DataLoader(
            dataset=ds,
            batch_size=batch_size,
            sampler=sampler,
            # dropping last avoids shape issues with compile + flex attention
            drop_last=True,
            collate_fn=partial(
                padded_collate_traj_CE,
                padding_idx=self._tokenizer.pad_id,
                ignore_idx=CROSS_ENTROPY_IGNORE_IDX,
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
        - Merged weights with key MODEL_KEY
        - Adapter weights with key ADAPTER_KEY
        - Relevant recipe state if training is not complete
        - If the `self._save_adapter_weights_only` option is True, the checkpointer will save only the adapter weights

        Checkpointer will save the merged weights, adapter weights and recipe state in
        different checkpoint files. To correctly resume from training, the adapter weights
        and recipe state must be provided along with the base model weights."""

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

        intermediate_checkpoint = epoch + 1 < self.total_epochs
        # To prevent GPU memory from spiking during checkpoint save,
        # we consolidate the full model and optim state dicts on CPU for rank 0
        state_dict = self._model.state_dict()
        if self._save_adapter_weights_only:
            state_dict = get_adapter_state_dict(state_dict, device=None)

        cpu_state_dict = training.gather_cpu_state_dict(
            state_dict,
            self._is_rank_zero,
            device=self._device,
        )
        if intermediate_checkpoint:
            opt_state_dict = training.get_full_optimizer_state_dict(
                self._optimizer,
                self._is_rank_zero,
                device=self._device,
            )
        else:
            opt_state_dict = None

        # Now that we have the model and opt state dict, create the actual checkpoint dict
        # to be sent to the checkpointer and ultimately written to file
        if self._is_rank_zero:
            if self._save_adapter_weights_only:
                adapter_state_dict = cpu_state_dict
            else:
                # Filter out the adapter keys and weights from the model state dict. These will
                # be saved separately
                adapter_state_dict = get_adapter_state_dict(cpu_state_dict)

                # merge the adapter weights and base weights to create the model checkpoint
                merged_state_dict = get_merged_lora_ckpt(
                    cpu_state_dict,
                    rank=self._lora_rank,
                    alpha=self._lora_alpha,
                )
                checkpoint_dict.update({training.MODEL_KEY: merged_state_dict})
            checkpoint_dict.update({training.ADAPTER_KEY: adapter_state_dict})

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
                adapter_only=self._save_adapter_weights_only,
            )

    def concatenated_forward(
        self, 
        model: nn.Module, 
        input_ids, 
        labels
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
    Run forward pass of the model with chosen and rejected samples concatenated.
    
    Args:
        model (nn.Module): The model to be used for the forward pass.
        input_ids: Input token IDs
        labels: Corresponding labels
    
    Returns:
        Tuple of log probs and logits
    """
        concatenated_input_ids = input_ids.to(self._device).unsqueeze(0)
        concatenated_labels = labels.to(self._device).unsqueeze(0)
    
        
    
        with self.activations_handling_ctx:
            all_logits = model(concatenated_input_ids)

        all_log_probs = rlhf.get_batch_log_probs(logits=all_logits, labels=concatenated_labels, return_average_logprobs=True)
    
        return (all_log_probs, all_logits)

    # NOTE: added by us
    def _skip_max_seq_len_samples(self, input_ids):
        max_inp=0
        for inp in input_ids:
            if len(inp)>max_inp:
                max_inp=len(inp)
        
        if max_inp>6000:
            return True
        else:
            return False

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
        num_tokens = 0

        # self.epochs_run should be non-zero when we're resuming from a checkpoint
        for curr_epoch in range(self.epochs_run, self.total_epochs):

            # Update the sampler to ensure data is correctly shuffled across epochs
            # in case shuffle is True
            self._sampler.set_epoch(curr_epoch)
            self._sampler_validation.set_epoch(curr_epoch)  # NOTE: added by us

            # NOTE: added by us
            # ------ Validation Step ------ #
            self._model.eval()

            with torch.no_grad():
                running_val_loss = 0
                running_reward_accuracy = 0

                num_eval_steps = (
                    min(self._max_validation_steps, len(self._dataloader_validation))
                    if self._max_validation_steps is not None
                    else len(self._dataloader_validation)
                )
                # NOTE: added by us
                # start a counter for samples that are too long
                max_len_samples = 0

                pbar_val = tqdm(total=num_eval_steps, desc="Validation")
                # NOTE: added by us - counter to account for samples that are too long
                idx = 0

                policy_chosen_sum = torch.zeros(1, device=self._device)
                reference_chosen_sum = torch.zeros(1, device=self._device)

                for _, batch in enumerate(self._dataloader_validation):
                    if self._max_validation_steps is not None and idx == self._max_validation_steps:
                        break

                    input_ids, labels, ce_label = batch
                    if self._skip_max_seq_len_samples(input_ids):
                        max_len_samples += 1
                        continue


                    local_len = len(input_ids)
                    local_len_tensor = torch.tensor([local_len], device=self._device, dtype=torch.int64)
                    world_size = dist.get_world_size()
                    all_lens = [torch.zeros(1, device=self._device, dtype=torch.int64) for _ in range(world_size)]
                    dist.barrier() 
                    dist.all_gather(all_lens, local_len_tensor)
                    max_len = max(t.item() for t in all_lens)

                    dist.barrier() 
                    del all_lens, local_len_tensor

                    policy_chosen_sum.zero_()
                    reference_chosen_sum.zero_()

                    reg_index=random.randint(0,len(input_ids)-1)
                    dist.barrier() 

                    for index in range(max_len):
                        if index<local_len:
                            inp=input_ids[index]
                            gnd=labels[index]
                        else:
                            inp=torch.tensor([128000, 220, 128001])
                            gnd=torch.tensor([128000, 220, 128001])
                        log_policy_probs, policy_logits = self.concatenated_forward(
                                self._model, inp, gnd
                            )
                        if index==reg_index:
                            sft_policy_logits=policy_logits
                            sft_policy_labels=labels[index]
                        del policy_logits

                        with torch.no_grad(), disable_adapter(self._model):
                            reference_log_probs, reference_logits = self.concatenated_forward(
                                self._model, inp, gnd
                            )

                            del reference_logits
                        del inp, gnd
                        torch.cuda.empty_cache()
                        policy_chosen_sum += log_policy_probs
                        reference_chosen_sum += reference_log_probs
                    

                    loss= self._loss_fn(
                    policy_chosen_sum,
                    reference_chosen_sum,
                    labels=ce_label[0].unsqueeze(0).to(self._device)
                    )

                    
                    loss = loss.mean()

                    running_val_loss += loss
                    pbar_val.update(1)
                    pbar_val.set_description(
                        f"{self.epochs_run+1}|{self.global_step}|Validation Loss: {running_val_loss / (idx + 1)}"
                    )
                    idx += 1

                mean_val_loss = running_val_loss / (idx + 1)
                gathered_mean_val_loss = [
                    torch.zeros_like(mean_val_loss) for _ in range(get_world_size())
                ]

                all_gather(gathered_mean_val_loss, mean_val_loss)


                mean_val_loss = torch.tensor(gathered_mean_val_loss).mean().cpu()

                if self._is_rank_zero:
                    self._metric_logger.log_dict(
                        {
                            "val_loss": mean_val_loss,

                        },
                        step=self.global_step,
                    )
                pbar_val.close()
                print("Number of samples that were too long: ", max_len_samples)

            # ------ Training Epoch ------ #
            # Initialize tokens count and running loss (for grad accumulation)
            t0 = time.perf_counter()
            num_tokens = 0
            running_loss = 0
            positive_num_tokens = 0
            negative_num_tokens=0
            max_len_samples = 0
            self._model.train()  # NOTE: added by us

            pbar = tqdm(
                total=self._steps_per_epoch, disable=not (rank == 0), desc="Training"
            )
            # NOTE: added by us - counter to account for samples that are too long
            idx = 0
            dummy_tensor = torch.tensor([128000, 220, 128001], device=self._device)
            for _, batch in enumerate(self._dataloader):
                if (
                    self.max_steps_per_epoch is not None
                    and (idx // self._gradient_accumulation_steps)
                    == self.max_steps_per_epoch
                ):
                    break

                input_ids, labels, ce_label = batch

                if self._skip_max_seq_len_samples(input_ids):
                    max_len_samples += 1
                    continue


                local_len = len(input_ids)
                local_len_tensor = torch.tensor([local_len], device=self._device, dtype=torch.int64)
                    
                world_size = dist.get_world_size()
                all_lens = [torch.zeros(1, device=self._device, dtype=torch.int64) for _ in range(world_size)]
                dist.barrier() 
                dist.all_gather(all_lens, local_len_tensor)
                max_len = max(t.item() for t in all_lens)
                del local_len_tensor, all_lens

                policy_chosen_sum = torch.zeros(1, device=self._device)
                reference_chosen_sum = torch.zeros(1, device=self._device)


                # batch is input_ids, labels
                reg_index=random.randint(0,len(input_ids)-1)
                dist.barrier()

                for index in range(max_len):
                    if index<local_len:
                        inp=input_ids[index]
                        gnd=labels[index]
                    else:
                        inp=dummy_tensor
                        gnd=dummy_tensor
                    log_policy_probs, policy_logits = self.concatenated_forward(
                                self._model, inp, gnd
                        )
                    if index==reg_index:
                        sft_policy_logits=policy_logits
                        sft_policy_labels=labels[index]
                    del policy_logits

                    with torch.no_grad(), disable_adapter(self._model):
                        reference_log_probs, reference_logits = self.concatenated_forward(
                                self._model, inp, gnd
                        )

                        del reference_logits
                    del inp, gnd

                    policy_chosen_sum += log_policy_probs
                    reference_chosen_sum += reference_log_probs

                    dist.barrier() 

                loss = self._loss_fn(
                    policy_chosen_sum,
                    reference_chosen_sum,
                    labels=ce_label[0].unsqueeze(0).to(self._device)
                )


                loss = loss.mean()


                loss = loss / self._gradient_accumulation_steps
                # ce_loss=ce_loss / self._gradient_accumulation_steps
                running_loss += loss
                dist.barrier() 
                loss.backward()
                

                # Step with optimizer
                if (idx + 1) % self._gradient_accumulation_steps == 0:
                    self._optimizer.step()
                    self._optimizer.zero_grad(set_to_none=True)

                    if self._lr_scheduler is not None:
                        self._lr_scheduler.step()
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
                            "idx": index,
                            "lr": self._optimizer.param_groups[0]["lr"],
                            "positive_tokens_per_second_per_gpu": positive_num_tokens / time_per_step,
                            "negative_tokens_per_second_per_gpu": negative_num_tokens / time_per_step,
                            # "log_probs/rejected": policy_rejected_sum.detach().mean().cpu(),
                            # "log_probs/chosen": policy_chosen_sum.detach().mean().cpu(),
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

                idx += 1  # NOTE: added by us

            self.epochs_run += 1
            self.save_checkpoint(epoch=curr_epoch)

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
    init_process_group(backend="gloo" if cfg.device == "cpu" else "nccl")

    config.log_config(recipe_name="LoRADPORecipeDistributed", cfg=cfg)

    recipe = LoRADPORecipeDistributed(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())