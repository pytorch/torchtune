# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
import time

from functools import partial
from typing import Any, Dict, Optional, Tuple
from warnings import warn

import torch
from omegaconf import DictConfig

from torch import nn
from torch.distributed import destroy_process_group, init_process_group
from torch.distributed.fsdp import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    StateDictType,
)
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler
from torchtune import config, modules, utils
from torchtune.modules.peft.peft_utils import (
    get_adapter_params,
    get_merged_lora_ckpt,
    set_trainable_params,
    validate_state_dict_for_lora,
)
from torchtune.recipe_interfaces import FTRecipeInterface

from torchtune.utils.distributed import validate_no_params_on_meta_device
from tqdm import tqdm

log = utils.get_logger("DEBUG")


def memory_stats_log(msg: str) -> str:
    return f"""
    Memory Stats {msg}:
    Memory Allocated: {torch.cuda.memory_allocated() / 1000**3:.2f} GB
    Memory Reserved: {torch.cuda.memory_reserved() / 1000**3:.2f} GB
    Peak Memory: {torch.cuda.max_memory_allocated() / 1000**3:.2f} GB
    """


class LoRAFinetuneDistributedRecipe(FTRecipeInterface):
    """
    Distributed LoRA finetuning recipe for dense transformer-based LLMs such as Llama2.

    This recipe supports:
        - FSDP and activation checkpointing. This is enabled by default but is
            configurable.
        - Mixed precision training - fp32, fp16 and bf16 are supported.
        - Checkpointing of full model weights and optionally of optimizer state (for
            checkpoints created during training).
        - Logging to terminal, WandB, or TensorBoard.

    Assumptions:
        - Training happens on CUDA (CPU training is not supported)
        - Checkpoints are ONLY saved at epoch boundaries. In case of failure, work done
            in ongoing epoch is lost.
        - Datasets are Map-style and data fits in memory (not streamed).

    The following configs can be used to run this recipe:
        >>> tune ls
        RECIPE                         CONFIG
        lora_finetune_distributed        alpaca_llama2_lora_finetune_distributed

    Args:
        cfg (DictConfig): OmegaConf object parsed from yaml file

    Raises:
        ValueError: If world_size is 1 and enable_fsdp is True
    """

    def __init__(self, cfg: DictConfig) -> None:
        self._device = utils.get_device(device=cfg.device)
        self._dtype = utils.get_dtype(dtype=cfg.dtype)

        world_size, rank = utils.get_world_size_and_rank()
        if world_size == 1 and cfg.enable_fsdp:
            raise ValueError("enable_fsdp should be False when world_size is 1")

        # _is_rank_zero is used primarily for logging. In the future, the logger
        # should directly take care of this
        self._is_rank_zero = rank == 0

        # logging attributes
        self._output_dir = cfg.output_dir
        self._log_every_n_steps = cfg.log_every_n_steps if cfg.log_every_n_steps else 1

        # training attributes
        self._enable_fsdp = cfg.enable_fsdp
        self._enable_activation_checkpointing = cfg.enable_activation_checkpointing

        # These attributes consitute the recipe state and are updated by ``load_checkpoint``
        # when ``resume_from_checkpoint`` is ``True``
        self.seed = utils.set_seed(seed=cfg.seed)
        self.epochs_run = 0
        self.total_epochs = cfg.epochs
        self.max_steps_per_epoch = cfg.max_steps_per_epoch
        self.total_training_steps = 0

        self._resume_from_checkpoint = cfg.resume_from_checkpoint

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
            if utils.ADAPTER_KEY not in checkpoint_dict:
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
        if not (
            utils.SEED_KEY in ckpt_dict
            and utils.TOTAL_EPOCHS_KEY in ckpt_dict
            and utils.MAX_STEPS_KEY in ckpt_dict
        ):
            raise KeyError(
                "Checkpoint does not contain the required keys needed for updating recipe state."
                "Are you sure you passed in the right recipe checkpoint?"
            )
        # If seed, total_epoch or max_steps_per_epoch don't match,
        # warn the user and overwrite
        if (
            self.seed != ckpt_dict[utils.SEED_KEY]
            or self.total_epochs != ckpt_dict[utils.TOTAL_EPOCHS_KEY]
            or self.max_steps_per_epoch != ckpt_dict[utils.MAX_STEPS_KEY]
        ):
            warn(
                message="""Configured value for seed, epochs or max_steps_per_epoch
                does not match the value stored in checkpoint."""
            )
        self.seed = utils.set_seed(seed=ckpt_dict[utils.SEED_KEY])
        self.epochs_run = ckpt_dict[utils.EPOCHS_KEY]
        self.total_epochs = ckpt_dict[utils.TOTAL_EPOCHS_KEY]
        self.max_steps_per_epoch = ckpt_dict[utils.MAX_STEPS_KEY]

    def setup(self, cfg: DictConfig) -> None:
        """
        Setup the recipe state. This includes recipe state (if resume_from_checkpoint is True),
        model, tokenizer, loss, optimizer, learning rate scheduler, sampler, and dataloader.
        """
        if self._is_rank_zero:
            self._metric_logger = config.instantiate(cfg.metric_logger)

        checkpoint_dict = self.load_checkpoint(cfg_checkpointer=cfg.checkpointer)

        self._model = self._setup_model(
            cfg_model=cfg.model,
            enable_fsdp=cfg.enable_fsdp,
            enable_activation_checkpointing=cfg.enable_activation_checkpointing,
            base_model_state_dict=checkpoint_dict[utils.MODEL_KEY],
            lora_weights_state_dict=(
                checkpoint_dict[utils.ADAPTER_KEY]
                if self._resume_from_checkpoint
                else None
            ),
        )
        self._tokenizer = config.instantiate(cfg.tokenizer)

        self._optimizer = self._setup_optimizer(
            cfg_optimizer=cfg.optimizer,
            opt_state_dict=checkpoint_dict[utils.OPT_KEY]
            if self._resume_from_checkpoint
            else None,
        )

        self._loss_fn = config.instantiate(cfg.loss)

        # sampler and dataloader depend on the tokenizer and loss_fn and should be
        # setup after all of these are setup
        self._sampler, self._dataloader = self._setup_data(
            cfg_dataset=cfg.dataset,
            shuffle=cfg.shuffle,
            batch_size=cfg.batch_size,
        )

        # Finally update the recipe state which can only be correctly set after all of the
        # other components have been initialized and updated.

        # Number of training steps in each epoch depends on the number of batches produced
        # by the dataloader and the max_steps_per_epoch param set by the user and is used
        # for logging and tracking training state. This should be computed after the dataloader
        # has been setup
        steps_per_epoch = len(self._dataloader)
        if self.max_steps_per_epoch is not None and self.max_steps_per_epoch < len(
            self._dataloader
        ):
            steps_per_epoch = self.max_steps_per_epoch
            self.total_training_steps = self.epochs_run * steps_per_epoch

        # Learning rate scheduler can only be set up after number of steps
        # has been computed
        self._lr_scheduler = self._setup_lr_scheduler(
            cfg_lr_scheduler=cfg.lr_scheduler,
            num_training_steps=self.total_epochs * steps_per_epoch,
            last_epoch=self.total_training_steps - 1,
        )

    def _setup_model(
        self,
        cfg_model: DictConfig,
        enable_fsdp: bool,
        enable_activation_checkpointing: bool,
        base_model_state_dict: Dict[str, Any],
        lora_weights_state_dict: Optional[Dict[str, Any]] = None,
    ) -> nn.Module:
        # Model initialization has two scenarios:
        #
        # 1. FSDP is enabled:
        #    a. To minimize GPU peak memory, we load the model on CPU with the right
        #       dtype. To ensure that we don't instantiate ``world_size`` number of models,
        #       we initialize on meta_device for all ranks other thank rank 0.
        #    b. Rank 0 is also responsible for calling ``load_state_dict`` and loading the
        #       model weights from checkpoint.
        #    c. While wrapping the model with FSDP, we set ``sync_module_states``
        #       to TRUE and broadcast module params and buffers from rank 0.
        #    d. The ``device_id`` param ensures that the FSDP initialization happens on
        #       the correct device.
        #
        # 2. FSDP is disabled:
        #    a. We instantiate the model on the correct device with the right dtype.

        if enable_fsdp:
            if self._is_rank_zero:
                log.info("FSDP is enabled. Instantiating Model on CPU for Rank 0 ...")
                init_start = time.perf_counter()

                with utils.set_default_dtype(self._dtype):
                    model = config.instantiate(cfg_model)

                log.info(
                    f"Model instantiation took {time.perf_counter() - init_start:.2f} secs"
                )

                # Load both the base model weights and (if available) the adapter weights. Both
                # of this should happen only on Rank 0
                model.load_state_dict(base_model_state_dict, strict=False)
                if lora_weights_state_dict:
                    model.load_state_dict(lora_weights_state_dict, strict=False)

            else:
                # For non-zero ranks, load the model on meta device
                with utils.set_default_dtype(self._dtype), torch.device("meta"):
                    model = config.instantiate(cfg_model)

        else:
            if self._is_rank_zero:
                log.info("FSDP is Disabled. Instantiating Model ...")
                init_start = time.perf_counter()

            # FSDP is disabled, directly instantiate model on correct device with the right dtype
            with utils.set_default_dtype(self._dtype), torch.device(self._device):
                model = config.instantiate(cfg_model)

            model.load_state_dict(base_model_state_dict, strict=False)
            if lora_weights_state_dict:
                model.load_state_dict(lora_weights_state_dict, strict=False)

            if self._is_rank_zero:
                log.info(f"Model init took {time.perf_counter() - init_start:.2f} secs")

        # The model contains LoRA params which won't have any matching keys in
        # the state dict. As a result, we need to load with strict=False.
        # Before loading the state dict, ensure the state dict keys for the base
        # model and adapters (if available) match the keys in the full LoRA model
        # This is a good sanity check to prevent silent errors
        validate_state_dict_for_lora(
            lora_attn_modules=cfg_model.lora_attn_modules,
            apply_lora_to_mlp=cfg_model.apply_lora_to_mlp,
            apply_lora_to_output=cfg_model.apply_lora_to_output,
            full_model_state_dict_keys=model.state_dict().keys(),
            lora_state_dict_keys=(
                lora_weights_state_dict.keys()
                if lora_weights_state_dict is not None
                else None
            ),
            base_model_state_dict_keys=base_model_state_dict.keys(),
        )

        # LoRA hyper-params needed for merging weights while saving checkpoints
        self._lora_rank = cfg_model.lora_rank
        self._lora_alpha = cfg_model.lora_alpha

        # Note: this needs to be set before wrapping with FSDP
        self.adapter_params = get_adapter_params(model)
        set_trainable_params(model, self.adapter_params)

        if enable_fsdp:
            model = FSDP(
                module=model,
                auto_wrap_policy=utils.lora_fsdp_wrap_policy(
                    modules_to_wrap={modules.TransformerDecoderLayer}
                ),
                sharding_strategy=torch.distributed.fsdp.ShardingStrategy.FULL_SHARD,
                device_id=self._device,
                # this recipe does not currently support mixed precision training
                mixed_precision=None,
                # Ensure we broadcast params and buffers from rank 0
                sync_module_states=True,
                # Initialize empty modules on all non-zero ranks
                param_init_fn=(
                    lambda module: module.to_empty(
                        device=torch.device("cuda"), recurse=False
                    )
                    if not self._is_rank_zero
                    else None
                ),
            )

        # Ensure no params and buffers are on meta device
        validate_no_params_on_meta_device(model)

        if enable_activation_checkpointing:
            utils.set_activation_checkpointing(
                model, auto_wrap_policy={modules.TransformerDecoderLayer}
            )
        if self._is_rank_zero:
            log.info(memory_stats_log("Memory Stats after model init:"))
        return model

    def _setup_optimizer(
        self, cfg_optimizer: DictConfig, opt_state_dict: Optional[Dict[str, Any]] = None
    ) -> Optimizer:
        optimizer = config.instantiate(cfg_optimizer, self._model.parameters())
        if opt_state_dict:
            # Note: technically we should check _contains_fsdp for
            # just the state dict of the adapter cfg, but should be equivalent
            opt_state_dict = utils.transform_opt_state_dict(
                opt_state_dict, self._model, optimizer
            )
            optimizer.load_state_dict(opt_state_dict)

        if self._is_rank_zero:
            log.info("Optimizer and loss are initialized.")
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
        if self._is_rank_zero:
            log.info("Learning rate scheduler is initialized.")
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
        world_size, rank = utils.get_world_size_and_rank()
        ds = config.instantiate(cfg_dataset, tokenizer=self._tokenizer)
        sampler = DistributedSampler(
            ds, num_replicas=world_size, rank=rank, shuffle=shuffle, seed=0
        )

        dataloader = DataLoader(
            dataset=ds,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=partial(
                utils.padded_collate,
                padding_idx=self._tokenizer.pad_id,
                ignore_idx=self._loss_fn.ignore_index,
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
        - Merged weights with key MODEL_KEY
        - Adapter weights with key ADAPTER_KEY
        - Relevant recipe state if training is not complete

        Checkpointer will save the merged weights, adapter weights and recipe state in
        different checkpoint files. To correctly resume from training, the adapter weights
        and recipe state must be provided along with the base model weights.
        """
        # final dict passed onto the checkpointer
        checkpoint_dict = {}

        cpu_state_dict = None
        opt_state_dict = None

        # in case of FSDP, to prevent GPU memory from spiking during checkpoint save,
        # we consolidate the full model and optim state dicts on CPU for rank 0
        if self._enable_fsdp:

            FSDP.set_state_dict_type(
                self._model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
                FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
            )

            cpu_state_dict = self._model.state_dict()
            opt_state_dict = FSDP.optim_state_dict(self._model, self._optimizer)

        # if FSDP is disabled, copy over the state dict from rank 0 to CPU
        else:
            if self._is_rank_zero:
                cpu_state_dict = {
                    k: v.cpu() for k, v in self._model.state_dict().items()
                }

            opt_state_dict = self._optimizer.state_dict()

        # Now that we have the model and opt state dict, create the actual checkpoint dict
        # to be sent to the checkpointer and ultimately written to file
        if self._is_rank_zero:

            # Filter out the adapter keys and weights from the model state dict. These will
            # be saved separately
            adapter_key_filter = lambda x: x in self.adapter_params
            adapter_state_dict = {
                k: v for k, v in cpu_state_dict.items() if adapter_key_filter(k)
            }
            checkpoint_dict.update({utils.ADAPTER_KEY: adapter_state_dict})

            # merge the adapter weights and base weights to create the model checkpoint
            merged_state_dict = get_merged_lora_ckpt(
                cpu_state_dict,
                rank=self._lora_rank,
                alpha=self._lora_alpha,
            )
            checkpoint_dict.update({utils.MODEL_KEY: merged_state_dict})

            # if training is in-progress, checkpoint the optimizer state as well
            if epoch + 1 < self.total_epochs:
                checkpoint_dict.update(
                    {
                        utils.OPT_KEY: opt_state_dict,
                        utils.SEED_KEY: self.seed,
                        utils.EPOCHS_KEY: self.epochs_run,
                        utils.TOTAL_EPOCHS_KEY: self.total_epochs,
                        utils.MAX_STEPS_KEY: self.max_steps_per_epoch,
                    }
                )

            self._checkpointer.save_checkpoint(
                checkpoint_dict,
                epoch=epoch,
                intermediate_checkpoint=(epoch + 1 < self.total_epochs),
            )

    def train(self) -> None:
        """
        The core training loop.
        """
        _, rank = utils.get_world_size_and_rank()

        # self.epochs_run should be non-zero when we're resuming from a checkpoint
        for curr_epoch in range(self.epochs_run, self.total_epochs):

            # Update the sampler to ensure data is correctly shuffled across epochs
            # in case shuffle is True
            self._sampler.set_epoch(curr_epoch)

            for idx, batch in enumerate(
                pbar := tqdm(self._dataloader, disable=not (rank == 0))
            ):
                if (
                    self.max_steps_per_epoch is not None
                    and idx == self.max_steps_per_epoch
                ):
                    break
                self.total_training_steps += 1
                self._optimizer.zero_grad()

                input_ids, labels = batch
                input_ids = input_ids.to(self._device)
                labels = labels.to(self._device)

                logits = self._model(input_ids)
                # Shift so that tokens < n predict n
                logits = logits[..., :-1, :].contiguous()
                labels = labels[..., 1:].contiguous()
                logits = logits.transpose(1, 2)
                # Compute loss
                loss = self._loss_fn(logits, labels)

                pbar.set_description(f"{curr_epoch+1}|{idx+1}|Loss: {loss.item()}")

                if (
                    self.total_training_steps % self._log_every_n_steps == 0
                    and self._is_rank_zero
                ):
                    pbar.set_description(f"{curr_epoch+1}|{idx+1}|Loss: {loss.item()}")
                    self._metric_logger.log_dict(
                        {
                            "loss": loss.item(),
                            "lr": self._optimizer.param_groups[0]["lr"],
                            "gpu_resources": torch.cuda.memory_allocated(),
                        },
                        step=self.total_training_steps,  # Each step is unique, not limited to each epoch
                    )

                loss.backward()
                self._optimizer.step()
                self._lr_scheduler.step()
                if self.total_training_steps % 100 == 0 and self._is_rank_zero:
                    log.info(memory_stats_log("Memory Stats:"))

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
        - Parameters specified in ``alpaca_llama2_lora_finetune_distributed.yaml``
        - Overwritten by arguments from the command-line
    """
    if not utils.is_distributed():
        raise RuntimeError(
            "Distributed finetune recipe should be run via a distributed launcher."
            "If using tune CLI, please specify --nnodes 1 and --nproc_per_node [num_gpus]"
        )

    init_process_group(backend="gloo" if cfg.device == "cpu" else "nccl")

    recipe = LoRAFinetuneDistributedRecipe(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())
