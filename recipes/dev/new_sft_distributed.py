# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
import time
from functools import partial
from typing import Any

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
from tqdm import tqdm


log = utils.get_logger("DEBUG")


@dataclass
class SFTRecipe(Recipe):
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

    def setup_env(self):
        self.device = training.get_device(self.device)
        self.dtype = training.get_dtype(
            self.dtype, device=self.device, options=["fp32", "bf16"]
        )

        # ---- Initialize distributed ---- #
        cpu_offload = self.fsdp_cpu_offload or self.enable_async_checkpointing
        distributed_backend = training.get_distributed_backend(
            self.device, offload_ops_to_cpu=cpu_offload
        )
        init_process_group(distributed_backend)
        if self.fsdp_cpu_offload:
            training.set_torch_num_threads()

        self.world_size, self.rank = training.get_world_size_and_rank()
        self.data_parallel_dim = self.world_size // self.tensor_parallel_dim
        self.device_mesh = init_device_mesh(
            self.device.type,
            mesh_shape=(self.data_parallel_dim, self.tensor_parallel_dim),
            mesh_dim_names=("dp", "tp"),
        )
        self.data_parallel_size = device_mesh["dp"].size()
        self.data_parallel_rank = device_mesh["dp"].get_local_rank()

        self.seed = training.set_seed(
            self.seed, debug_mode=self.cudnn_deterministic_mode
        )

        # ---- Setup Checkpointing ---- #
        self.checkpointer = training.Checkpointer(self.checkponter)
        self.checkpoint_dict = self.checkpointer.load_base_checkpoint()
        self.total_steps = self.num_training_steps
        self.step = 0

        # ---- Initialize logging ---- #
        self.metric_logger = config.instantiate(self.metric_logger, log_rank=0)
        self.metric_logger.log_config(self.config)
        self.tqdm = tqdm(total=self.num_training_steps, disable=self.rank > 0)
        self.profiler = training.get_profiler(**self.profiler)

    def setup_model(self):
        utils.log_rank_zero(
            log,
            "Distributed training is enabled. Instantiating model and loading checkpoint on Rank 0",
        )
        init_start = time.perf_counter()

        with training.set_default_dtype(self.dtype), torch.device("meta"):
            self.model = config.instantiate(self.model)

        if self.compile:
            training.compile_model(self.model, verbose=self.rank == 0)

        # ---- distributed and/or offload model ---- #
        # tensor parallel
        if self.tensor_parallel_dim > 1:
            tp_parallelize_module(
                self.model, self.device_mesh["tp"], self.tensor_parallel_plan
            )

        # activation checkpointing
        if enable_activation_checkpointing:
            training.set_activation_checkpointing(
                model, auto_wrap_policy={modules.TransformerSelfAttentionLayer}
            )

        # sharded data parallel
        if self.data_parallel_dim > 1:
            data_parallelize_module(
                model,
                self.custom_sharded_layers,
                self.fsdp_cpu_offload,
                self.fsdp_reshard_after_forward,
                self.device_mesh,
            )

        # init buffers not stored in state dict
        init_buffers(model, dtype, device)

        # DTensor state dict
        training.load_from_full_model_state_dict(
            self.model,
            self.checkpoint_dict[training.MODEL_KEY],
            self.device,
            strict=True,
            cpu_offload=self.fsdp_cpu_offload,
        )

        # activation offloading
        self.activation_offloading = training.get_act_offloading_manager(
            self.model, self.enable_activation_offloading
        )

        # ensure no params and buffers are on meta device
        training.validate_no_params_on_meta_device(model)

        if self.rank == 0:
            memory_stats = training.get_memory_stats(device=self.device)
            training.log_memory_stats(memory_stats)

        # synchronize before training begins
        torch.distributed.barrier()
        utils.log_rank_zero(
            log,
            f"Loading model and checkpoint took {time.perf_counter() - init_start:.2f} secs",
        )

        # ---- setup tokenizer ---- #
        self.tokenizer = config.instantiate(self.tokenizer)

    def setup_optimizer(self):
        # ---- setup optimizer ---- #
        self.optimizer = config.instantiate(self.optimizer, self.model.parameters())
        opt_state_dict = self.checkpoint_dict.get(training.OPT_KEY, None)
        if opt_state_dict:
            training.load_from_full_optimizer_state_dict(
                self.model, self.optimizer, opt_state_dict, self.device
            )
        utils.log_rank_zero(log, "Optimizer is initialized.")

        # ---- setup lr scheduler ---- #
        if self.lr_scheduler is not None:
            lr_scheduler = config.instantiate(
                self.lr_scheduler,
                self.optimizer,
                num_training_steps=self.num_training_steps,
                last_epoch=self.step - 1,
            )
            utils.log_rank_zero(log, "Learning rate scheduler is initialized.")

    def setup_loss(self):
        self.loss = config.instantiate(self.loss)
        if self.compile:
            training.compile_loss(self.loss, verbose=self.rank == 0)
        if isinstance(self.loss, torchtune.modules.loss.CEWithChunkedOutputLoss):
            self.model.set_num_output_chunks(self.loss.num_output_chunks)
        utils.log_rank_zero(log, "Loss is initialized.")

        # ---- cache ignore labels ---- #
        self.ignore_labels_cache = torch.full(
            (self.batch_size, 1), self.loss.ignore_index, device=self.device
        )

    def setup_data(self, dataset):
        # [todo] add util?
        if isinstance(dataset, ListConfig):
            datasets = [config.instantiate(ds, self.tokenizer) for ds in dataset]
            ds = ConcatDataset(datasets=datasets)
            packed = getattr(ds, "packed", False)
        else:
            ds = config.instantiate(dataset, self.tokenizer)
            packed = dataset.get("packed", False)

        # Instantiate collate_fn
        if "left_pad_sequence" in self.collate_fn:
            raise RuntimeError("left_pad_sequence collator is only for inference.")
        collate_fn = _get_component_from_path(self.collate_fn)

        self.sampler = DistributedSampler(
            ds,
            num_replicas=self.dp_size,
            rank=self.dp_rank,
            shuffle=self.shuffle,
            seed=0,
        )
        self.dataloader = DataLoader(
            dataset=ds,
            batch_size=self.batch_size,
            sampler=self.sampler,
            # dropping last avoids shape issues with compile + flex attention
            drop_last=True,
            collate_fn=(
                partial(
                    collate_fn,
                    padding_idx=self.tokenizer.pad_id,
                    ignore_idx=self.loss.ignore_index,
                )
                if not packed
                else padded_collate_packed
            ),
        )

        utils.log_rank_zero(log, "Dataset and Sampler are initialized.")

    def resume_training(self):
        self.checkpointer.update_state()
        self.checkpointer.load_distributed_checkpoint(self.model, self.optimizer)

    def eval(self):
        # [todo] set model.eval, copy "train batch" section, copy "log section"
        pass

    def train(self):
        training.cleanup_before_training()
        self.optimizer.zero_grad()

        with self.profiler() as profiler:
            for step in range(self.step, self.total_steps):
                self.step = step
                # ---- train batch ---- #
                with GradientAccumulation(
                    self.gradient_accumulation_steps,
                    self.model,
                    self.data_parallel_size,
                ) as grad_acc:
                    for _ in range(grad_acc.steps):
                        batch = next(self.dataloader)
                        utils.batch_to_device(batch, self.device)

                        # Shape [b, s], needed for the loss not the model
                        labels = batch.pop("labels")
                        num_tokens = (labels != loss.ignore_index).sum()

                        with self.activation_offloading:
                            logits = self.model(**batch)

                        # Shift labels to compute loss
                        # equivalent to doing labels[..., 1:] and logits[..., :-1, :]
                        # But this way we dont need to slice the logits. We just add an ignore index to labels.
                        labels = torch.hstack(
                            (
                                labels[..., 1:],
                                self.ignore_labels_cache[: labels.shape[0]],
                            )
                        )
                        if not isinstance(logits, list):
                            labels = labels.reshape(-1)
                            logits = logits.reshape(-1, logits.size(-1))

                        # Loss is normalized by the number of tokens for accumulating gradients
                        loss = self.loss(logits, labels)
                        grad_acc.scale_loss(loss, num_tokens)

                        # free logits otherwise it peaks backward memory
                        del logits
                        loss.backward()

                # ---- optimizer step ---- #
                if self.clip_grad_norm is not None:
                    max_norm = float(self.clip_grad_norm)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=max_norm
                    )
                    if isinstance(grad_norm, DTensor):
                        grad_norm = grad_norm.full_tensor()
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                # ---- logging ---- #
                self.tqdm.update(1)
                self.tqdm.set_description(
                    f"{curr_epoch + 1}|{self.step}|Loss: {loss}"
                )  # [todo] no epoch
                if self.step % self.log_every_n_steps == 0 and self.rank == 0:
                    time_per_step = time.perf_counter() - t0
                    log_dict = {
                        "loss": grad_acc.avg_loss,
                        "lr": training.get_lr(self.optimizer),
                        "tokens_per_second_per_gpu": grad_acc.num_tokens
                        / (time_per_step * self.world_size),
                    }
                    if self.log_peak_memory_stats:
                        log_dict.update(training.get_memory_stats(device=self.device))
                    if self.clip_grad_norm is not None:
                        log_dict.update({"grad_norm": grad_norm})
                    self.metric_logger.log_dict(log_dict, step=self.step)
                profiler.step()

                # ---- checkpointing ---- #
                if self.step % self.save_every_n_steps == 0:
                    self.checkpointer.save_checkpoint(
                        model=self.model,
                        optimizer=self.optimizer,
                        training_progress=TrainingProgress(
                            seed=self.seed,
                            epochs_run=self.epochs_run,
                            total_epochs=self.total_epochs,
                        ),
                        step=self.step,
                    )

                # ---- run eval ---- #
                if self.step % self.eval_every_n_steps == 0:
                    self.eval()

    def cleanup(self):
        if self.rank == 0:
            self.metric_logger.close()
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
    trainer = TrainingRecipe(cfg=cfg)
    trainer.setup_env()
    trainer.setup_model()
    trainer.setup_optimizer()
    trainer.setup_loss()
    trainer.setup_data(trainer.dataset)
    trainer.setup_data(trainer.eval_dataset)
    if trainer.resume_from_checkpoint:
        trainer.resume_training()
    trainer.train()
    trainer.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())
