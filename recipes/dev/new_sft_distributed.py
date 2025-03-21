# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
import time
from tqdm import tqdm
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.distributed import (
    destroy_process_group,
    init_device_mesh,
    init_process_group,
)
from torch.distributed._tensor import DTensor
from torch.utils.data import DataLoader, DistributedSampler
from torchtune import config, modules, training, utils
from torchtune.config._utils import _get_component_from_path
from torchtune.data import padded_collate_packed
from torchtune.datasets import ConcatDataset


log = utils.get_logger("DEBUG")

@dataclass
class RecipeState:
    device = "cuda"
    dtype = "bf16"
    seed = None
    compile = False

    # distributed attributes
    enable_async_checkpointing = False
    fsdp_cpu_offload = False
    fsdp_reshard_after_forward = True
    custom_sharded_layers = None
    tensor_parallel_plan = None
    tensor_parallel_dim = 1

    # logging attributes
    output_dir = "."
    log_every_n_steps = 1
    log_peak_memory_stats = False
    eval_every_n_steps = 1
    save_every_n_steps = 1

    # training attributes
    resume_from_checkpoint = False
    gradient_accumulation_steps = 1
    clip_grad_norm = None
    enable_activation_checkpointing = False
    enable_activation_offloading = False
    shuffle = False
    batch_size = 1
    num_training_steps = 1

    # artifacts
    model: Any
    tokenizer: Any
    optimizer: Any
    loss: Any
    checkpointer: Any
    metric_logger: Any
    dataset: Any
    eval_dataset: Any = None
    collate_fn: Any = torchtune.data.padded_collate_sft
    profiler: Any = None


def setup_env(state):
    state.device = training.get_device(state.device)
    state.dtype = training.get_dtype(state.dtype, device=state.device, options=["fp32", "bf16"])

    # ---- Initialize distributed ---- #
    cpu_offload = state.fsdp_cpu_offload or state.enable_async_checkpointing
    distributed_backend = training.get_distributed_backend(state.device, offload_ops_to_cpu=cpu_offload)
    init_process_group(distributed_backend)
    if state.fsdp_cpu_offload:
        training.set_torch_num_threads()

    state.world_size, state.rank = training.get_world_size_and_rank()
    state.data_parallel_dim = state.world_size // state.tensor_parallel_dim
    state.device_mesh = training.init_device_mesh(
        state.device.type,
        mesh_shape=(state.data_parallel_dim, state.tensor_parallel_dim),
        mesh_dim_names=("dp", "tp"),
    )
    state.data_parallel_size = device_mesh["dp"].size()
    state.data_parallel_rank = device_mesh["dp"].get_local_rank()

    state.seed = training.set_seed(state.seed, debug_mode=state.cudnn_deterministic_mode)

    # ---- Setup Checkpointing ---- #
    state.checkpointer = training.Checkpointer(state.checkponter)
    state.checkpoint_dict = state.checkpointer.load_base_checkpoint()
    state.total_steps = state.num_training_steps
    state.step = 0

    # ---- Initialize logging ---- #
    state.metric_logger = config.instantiate(state.metric_logger, log_rank=0)
    state.metric_logger.log_config(state)
    state.tqdm = tqdm(total=state.num_training_steps, disable=state.rank > 0)
    state.profiler = training.get_profiler(**state.profiler)


def setup_model(state):
    utils.log_rank_zero(log, "Distributed training is enabled. Instantiating model and loading checkpoint on Rank 0",)
    init_start = time.perf_counter()

    with training.set_default_dtype(state.dtype), torch.device("meta"):
        state.model = config.instantiate(state.model)

    if state.compile:
        training.compile_model(state.model, verbose=state.rank==0)

    # ---- distributed and/or offload model ---- #
    # tensor parallel
    if self.tensor_parallel_dim > 1:
        tp_parallelize_module(state.model, state.device_mesh["tp"], state.tensor_parallel_plan)

    # activation checkpointing
    if enable_activation_checkpointing:
        training.set_activation_checkpointing(model, auto_wrap_policy={modules.TransformerSelfAttentionLayer})

    # sharded data parallel
    if self.data_parallel_dim > 1:
        data_parallelize_module(
            model, 
            state.custom_sharded_layers,
            state.fsdp_cpu_offload,
            state.fsdp_reshard_after_forward,
            state.device_mesh
        )

    # init buffers not stored in state dict
    init_buffers(model, dtype, device)

    # DTensor state dict
    training.load_from_full_model_state_dict(
        state.model,
        state.checkpoint_dict[training.MODEL_KEY],
        state.device,
        strict=True,
        cpu_offload=state.fsdp_cpu_offload,
    )

    # activation offloading
    state.activation_offloading = training.get_act_offloading_manager(state.model, state.enable_activation_offloading)

    # ensure no params and buffers are on meta device
    training.validate_no_params_on_meta_device(model)

    if state.rank == 0:
        memory_stats = training.get_memory_stats(device=state.device)
        training.log_memory_stats(memory_stats)

    # synchronize before training begins
    torch.distributed.barrier()
    utils.log_rank_zero(log, f"Loading model and checkpoint took {time.perf_counter() - init_start:.2f} secs")

    # ---- setup tokenizer ---- #
    state.tokenizer = config.instantiate(state.tokenizer)


def setup_optimizer(state):
    # ---- setup optimizer ---- #
    state.optimizer = config.instantiate(state.optimizer, state.model.parameters())
    opt_state_dict = state.checkpoint_dict.get(training.OPT_KEY, None)
    if opt_state_dict:
        training.load_from_full_optimizer_state_dict(state.model, state.optimizer, opt_state_dict, state.device)
    utils.log_rank_zero(log, "Optimizer is initialized.")

    # ---- setup lr scheduler ---- #
    if state.lr_scheduler is not None:
        lr_scheduler = config.instantiate(
            state.lr_scheduler,
            state.optimizer,
            num_training_steps=state.num_training_steps,
            last_epoch=state.step - 1,
        )
        utils.log_rank_zero(log, "Learning rate scheduler is initialized.")

    
def setup_loss(state):
    state.loss = config.instantiate(state.loss)
    if state.compile:
        training.compile_loss(state.loss, verbose=state.rank==0)
    if isinstance(state.loss, torchtune.modules.loss.CEWithChunkedOutputLoss)
        state.model.set_num_output_chunks(state.loss.num_output_chunks)
    utils.log_rank_zero(log, "Loss is initialized.")

    # ---- cache ignore labels ---- #
    state.ignore_labels_cache = torch.full((state.batch_size, 1), state.loss.ignore_index, device=state.device)


def setup_data(state, dataset):
    # [todo] add util?
    if isinstance(dataset, ListConfig):
        datasets = [config.instantiate(ds, state.tokenizer) for ds in dataset]
        ds = ConcatDataset(datasets=datasets)
        packed = getattr(ds, "packed", False)
    else:
        ds = config.instantiate(dataset, state.tokenizer)
        packed = dataset.get("packed", False)

    # Instantiate collate_fn
    if "left_pad_sequence" in state.collate_fn:
        raise RuntimeError("left_pad_sequence collator is only for inference.")
    collate_fn = _get_component_from_path(state.collate_fn)

    state.sampler = DistributedSampler(
        ds, num_replicas=state.dp_size, rank=state.dp_rank, shuffle=state.shuffle, seed=0
    )
    state.dataloader = DataLoader(
        dataset=ds,
        batch_size=state.batch_size,
        sampler=state.sampler,
        # dropping last avoids shape issues with compile + flex attention
        drop_last=True,
        collate_fn=(
            partial(
                collate_fn,
                padding_idx=state.tokenizer.pad_id,
                ignore_idx=state.loss.ignore_index,
            )
            if not packed
            else padded_collate_packed
        ),
    )

    utils.log_rank_zero(log, "Dataset and Sampler are initialized.")


def resume_training(state)
    state.checkpointer.update_state(state)
    state.checkpointer.load_distributed_checkpoint(state.model, state.optimizer)


def eval(state):
    # [todo] set model.eval, copy "train batch" section, copy "log section"


def train(state):
    training.cleanup_before_training()
    state.optimizer.zero_grad()

    with state.profiler() as profiler:
        for state.step in range(state.step, state.total_steps):
            # ---- train batch ---- #
            with GradientAccumulation(state.gradient_accumulation_steps, state.model, state.data_parallel_size) as grad_acc:
                for _ in range(grad_acc.steps):
                    batch = next(state.dataloader)
                    utils.batch_to_device(batch, state.device)

                    # Shape [b, s], needed for the loss not the model
                    labels = batch.pop("labels")
                    num_tokens = (labels != loss.ignore_index).sum()

                    with self.activation_offloading:
                        logits = state.model(**batch)

                    # Shift labels to compute loss
                    # equivalent to doing labels[..., 1:] and logits[..., :-1, :]
                    # But this way we dont need to slice the logits. We just add an ignore index to labels.
                    labels = torch.hstack((labels[..., 1:], self.ignore_labels_cache[: labels.shape[0]]))
                    if not isinstance(logits, list):
                        labels = labels.reshape(-1)
                        logits = logits.reshape(-1, logits.size(-1))

                    # Loss is normalized by the number of tokens for accumulating gradients
                    loss = state.loss(logits, labels)
                    grad_acc.scale_loss(loss, num_tokens)

                    # free logits otherwise it peaks backward memory
                    del logits
                    loss.backward()

            # ---- optimizer step ---- #
            if state.clip_grad_norm is not None:
                max_norm = float(state.clip_grad_norm)
                grad_norm = torch.nn.utils.clip_grad_norm_(state.model.parameters(), max_norm=max_norm)
                if isinstance(grad_norm, DTensor):
                    grad_norm = grad_norm.full_tensor()
            state.optimizer.step()
            state.optimizer.zero_grad(set_to_none=True)
            if state.lr_scheduler is not None:
                state.lr_scheduler.step()

            # ---- logging ---- #
            state.tqdm.update(1)
            state.tqdm.set_description(f"{curr_epoch + 1}|{state.step}|Loss: {loss}") # [todo] no epoch
            if state.step % state.log_every_n_steps == 0 and state.rank==0:
                time_per_step = time.perf_counter() - t0
                log_dict = {
                    "loss": grad_acc.avg_loss,
                    "lr": training.get_lr(state.optimizer)
                    "tokens_per_second_per_gpu": grad_acc.num_tokens / (time_per_step * state.world_size),
                }
                if state.log_peak_memory_stats:
                    log_dict.update(training.get_memory_stats(device=state.device))
                if state.clip_grad_norm is not None:
                    log_dict.update({"grad_norm": grad_norm})
                state.metric_logger.log_dict(log_dict, step=state.step)
            profiler.step()

            # ---- checkpointing ---- #
            if state.step % state.save_every_n_steps == 0:
                state.checkpointer.save_checkpoint(
                    model=state.model,
                    optimizer=state.optimizer,
                    training_progress=TrainingProgress(
                        seed=state.seed,
                        epochs_run=state.epochs_run,
                        total_epochs=state.total_epochs,
                    ),
                    step=state.step,
                )

            # ---- run eval ---- #
            if state.step % state.eval_every_n_steps == 0:
                eval(state)


def cleanup(state):
    if state.rank==0:
        state.metric_logger.close()
    destroy_process_group()


@config.parse
def recipe_main(cfg: DictConfig) -> None:
    """
    Entry point for the recipe.

    Configurable parameters are read in the following order:
        - Parameters specified in config (see available configs through ``tune ls``)
        - Overwritten by arguments from the command-line
    """
    config.log_config(recipe_name="FullFinetuneRecipeDistributed", cfg=cfg)
    state = TrainingState(cfg=cfg)
    setup_env(state)
    setup_model(state)
    setup_optimizer(state)
    setup_loss(state)
    setup_data(state, state.dataset)
    setup_data(state, state.eval_dataset)
    if state.resume_from_checkpoint:
        resume_training(state)
    train(state)
    cleanup(state)

if __name__ == "__main__":
    sys.exit(recipe_main())
