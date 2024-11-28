# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Original from lora_finetune_distributed.py

import os
import pprint
import sys
import time
from collections import defaultdict
from functools import partial
from typing import Any, Dict, Tuple

import torch
from omegaconf import DictConfig, ListConfig
from torch import nn
from torch.distributed import destroy_process_group, init_process_group
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader, DistributedSampler
from torchtune import config, modules, training, utils
from torchtune.data import padded_collate_sft
from torchtune.datasets import ConcatDataset
from torchtune.modules.peft import get_merged_lora_ckpt
from torchtune.recipe_interfaces import EvalRecipeInterface
from torchtune.training.metrics import compute_classification_metrics
from tqdm import tqdm

log = utils.get_logger("DEBUG")


class EvalRecipeDistributed(EvalRecipeInterface):
    """
    Distributed LoRA finetuning recipe for dense transformer-based LLMs such as Llama2. This recipe supports
    distributed training and can be run on a single node (1 to 8 GPUs).

    Features:
        - FSDP. Supported using PyTorch's FSDP APIs. This can be parameterized using the
            ``fsdp_sharding_strategy`` config option. You can pass any value supported by
            torch.distributed.fsdp.ShardingStrategy
            (https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.ShardingStrategy).
            For example, in your config, simply pass ``fsdp_sharding=NO_SHARD`` for DDP.

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

        _, rank = training.get_world_size_and_rank()

        # _is_rank_zero is used primarily for logging. In the future, the logger
        # should directly take care of this
        self._is_rank_zero = rank == 0

        # logging attributes
        # self._output_dir = cfg.output_dir
        # self._log_every_n_steps = cfg.get("log_every_n_steps", 1)
        # self._log_peak_memory_stats = cfg.get("log_peak_memory_stats", False)

        self.global_step = 0
        self._resume_from_checkpoint = cfg.resume_from_checkpoint

        self._fsdp_sharding_strategy = torch.distributed.fsdp.ShardingStrategy[
            cfg.get("fsdp_sharding_strategy", "FULL_SHARD")
        ]

    def load_checkpoint(self, cfg_checkpointer: DictConfig) -> Dict[str, Any]:
        """
        Extract the checkpoint state from file and validate. This includes the
        base model weights. If resume_from_checkpoint is True, this also includes
        the adapter weights and recipe state
        """
        self._checkpointer = config.instantiate(
            cfg_checkpointer,
        )
        checkpoint_dict = self._checkpointer.load_checkpoint()

        return checkpoint_dict

    def setup(self, cfg: DictConfig) -> None:
        """
        Setup the recipe state. This includes recipe state (if resume_from_checkpoint is True),
        model, tokenizer, loss, optimizer, learning rate scheduler, sampler, and dataloader.
        """
        if self._is_rank_zero:
            if cfg.get("metric_logger"):
                self._metric_logger = config.instantiate(cfg.metric_logger)

        # log config with parameter override
        # self._metric_logger.log_config(cfg)

        checkpoint_dict = self.load_checkpoint(cfg_checkpointer=cfg.checkpointer)

        self.checkpoint_epoch = cfg.get("checkpoint_epoch", 0)
        print(f"checkpoint_epoch: {self.checkpoint_epoch}")

        # Merge adapter weights with base model weights
        # find faster way to do this
        if training.ADAPTER_KEY in checkpoint_dict:
            if training.ADAPTER_CONFIG not in checkpoint_dict:
                raise ValueError(
                    "Adapter weights found in checkpoint but adapter config not found."
                )
            tmp_state_dict = {
                **checkpoint_dict[training.MODEL_KEY],
                **checkpoint_dict[training.ADAPTER_KEY],
            }

            # cast to float32 since bfloat16 has really slow calculations for torch.matmul
            # model will cast back to bfloat16 later on
            if self._dtype == torch.bfloat16:
                tmp_state_dict = {
                    k: v.to(torch.float32) for k, v in tmp_state_dict.items()
                }

            merged_state_dict = get_merged_lora_ckpt(
                tmp_state_dict,
                rank=checkpoint_dict[training.ADAPTER_CONFIG]["r"],
                alpha=checkpoint_dict[training.ADAPTER_CONFIG]["lora_alpha"],
            )

            checkpoint_dict.update({training.MODEL_KEY: merged_state_dict})

        self._model = self._setup_model(
            cfg_model=cfg.model,
            base_model_state_dict=checkpoint_dict[training.MODEL_KEY],
        )
        self._tokenizer = config.instantiate(cfg.tokenizer)

        self._loss_fn = config.instantiate(cfg.loss)

        self._val_batch_size = cfg.val_batch_size

        self._val_sampler, self._val_dataloader = self._setup_data(
            cfg_dataset=cfg.val_dataset,
            shuffle=False,
            batch_size=cfg.val_batch_size,
        )

        self.eval_prefix = cfg.get("eval_prefix", "eval")

        self._steps_per_val_epoch = len(self._val_dataloader)

    def _setup_model(
        self,
        cfg_model: DictConfig,
        base_model_state_dict: Dict[str, Any],
    ) -> nn.Module:
        """
        Model initialization has some important considerations:
           a. To minimize GPU peak memory, we load the model on CPU with the right
              dtype. To ensure that we don't instantiate ``world_size`` number of models,
              we initialize on meta_device for all ranks other than rank 0.
           b. Rank 0 is also responsible for calling ``load_state_dict`` and loading the
              model weights from checkpoint.
           c. While wrapping the model with FSDP, we set ``sync_module_states``
              to TRUE and broadcast module params and buffers from rank 0.
           d. The ``device_id`` param ensures that the FSDP initialization happens on
              the correct device.
        """

        if self._is_rank_zero:
            log.info("FSDP is enabled. Instantiating Model on CPU for Rank 0 ...")
            init_start = time.perf_counter()

            with training.set_default_dtype(self._dtype):
                model = config.instantiate(cfg_model)

            log.info(
                f"Model instantiation took {time.perf_counter() - init_start:.2f} secs"
            )

            model.load_state_dict(base_model_state_dict, strict=False)

        else:
            # For non-zero ranks, load the model on meta device
            with training.set_default_dtype(self._dtype), torch.device("meta"):
                model = config.instantiate(cfg_model)

        if self._dtype == torch.bfloat16:
            model = model.to(torch.bfloat16)

        model = FSDP(
            module=model,
            auto_wrap_policy=training.lora_fsdp_wrap_policy(
                modules_to_wrap={modules.TransformerSelfAttentionLayer}
            ),
            sharding_strategy=self._fsdp_sharding_strategy,
            device_id=self._device,
            # this recipe does not currently support mixed precision training
            mixed_precision=None,
            # Ensure we broadcast params and buffers from rank 0
            sync_module_states=True,
            # Initialize empty modules on all non-zero ranks
            param_init_fn=(
                lambda module: (
                    module.to_empty(device=torch.device("cuda"), recurse=False)
                    if not self._is_rank_zero
                    else None
                )
            ),
        )

        # Ensure no params and buffers are on meta device
        training.validate_no_params_on_meta_device(model)

        # synchronize before training begins
        torch.distributed.barrier()

        return model

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
                config.instantiate(single_cfg_dataset, self._tokenizer)
                for single_cfg_dataset in cfg_dataset
            ]
            ds = ConcatDataset(datasets=datasets)
            packed = False
        else:
            ds = config.instantiate(cfg_dataset, self._tokenizer)
            packed = cfg_dataset.get("packed", False)

        sampler = DistributedSampler(
            ds, num_replicas=world_size, rank=rank, shuffle=shuffle, seed=0
        )

        dataloader = DataLoader(
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
                else None
            ),
        )

        if self._is_rank_zero:
            log.info("Dataset and Sampler are initialized.")

        return sampler, dataloader

    ############################
    # Validation loop, derived from training loop
    ############################
    @torch.no_grad()
    def evaluate(self) -> None:
        """
        The evaluation loop.
        """
        _, rank = training.get_world_size_and_rank()
        running_loss = 0.0
        running_metrics = defaultdict(float)

        pbar = tqdm(total=self._steps_per_val_epoch, disable=not (rank == 0))

        for idx, batch in enumerate(self._val_dataloader):
            with torch.no_grad():
                # Both are shape [b, s]
                tokens, labels = batch["tokens"], batch["labels"]
                # Get the attention mask and position ids from the dataset if they
                # exist. Currently, only sample packing in PackedDataset returns these
                mask = batch.get("mask", None)  # shape [b, s, s]
                input_pos = batch.get("input_pos", None)  # shape [b, s]

                tokens = tokens.to(self._device)
                labels = labels.to(self._device)
                mask = mask.to(self._device) if mask is not None else None
                input_pos = (
                    input_pos.to(self._device) if input_pos is not None else None
                )

                logits = self._model(tokens, mask=mask, input_pos=input_pos)
                # Shift labels to compute loss
                # equivalent to doing labels[..., 1:] and logits[..., :-1, :]
                # But this way we dont need to slice the logits. We just add an ignore index to labels.
                # labels = torch.hstack(
                #     (labels[..., 1:], self.ignore_labels_cache[: labels.shape[0]])
                # )
                # Shift so that tokens < n predict n
                logits = logits[..., :-1, :].contiguous()
                labels = labels[..., 1:].contiguous()
                # if not isinstance(logits, list):
                #     labels = labels.reshape(-1)
                #     logits = logits.reshape(-1, logits.size(-1))
                logits = logits.transpose(1, 2)

                # Compute loss
                loss = self._loss_fn(logits, labels)
                running_loss += loss

                logits = logits.transpose(1, 2)

                metrics_dict, running_metrics = compute_classification_metrics(
                    split="",
                    tokens=tokens,
                    labels=labels,
                    logits=logits,
                    batch=batch,
                    tokenizer=self._tokenizer,
                    running_metrics=running_metrics,
                )

                # free logits otherwise it peaks backward memory
                del logits

                pbar.update(1)
                pbar.set_description(
                    f"{idx+1}| {pprint.pformat(metrics_dict, compact=True)}"
                )

        # average the losses across all batches

        # Log validation metrics
        if rank == 0:
            log_dict = {
                "avg_loss": running_loss / len(self._val_dataloader),
                **metrics_dict,
            }
            log.info("Evaluation Metrics")
            log.info(log_dict)
            # if self._log_peak_memory_stats:
            #     log_dict.update(training.get_memory_stats(device=self._device))
            if self._metric_logger:
                self._metric_logger.log_dict(
                    log_dict,
                    prefix=self.eval_prefix,
                    epoch=self.checkpoint_epoch,
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
    if not training.is_distributed():
        raise RuntimeError(
            "Distributed finetune recipe should be run via a distributed launcher."
            "If using tune CLI, please specify --nnodes 1 and --nproc_per_node [num_gpus]"
        )
    os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
    init_process_group(backend="gloo" if cfg.device == "cpu" else "nccl")

    # config.log_config(
    #     recipe_name="LoRAClassificationEvalRecipeDistributed", cfg=cfg
    # )

    recipe = EvalRecipeDistributed(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.evaluate()
    recipe.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())
