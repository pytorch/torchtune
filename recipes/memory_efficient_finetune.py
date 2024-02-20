# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import sys

from functools import partial
from typing import Any, Dict, Optional, Tuple
from warnings import warn

import torch

from recipes.interfaces import FTRecipeInterface
from recipes.params.memory_efficient_finetune import MemEfficientFTParams

from torch import nn
from torch.cuda.amp import GradScaler
from torch.distributed import init_process_group

from torch.distributed.optim.apply_optimizer_in_backward import (
    _apply_optimizer_in_backward,
)
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler

from torchtune import datasets, models, modules, utils
from torchtune.utils.constants import (
    EPOCHS_KEY,
    MAX_STEPS_KEY,
    MODEL_KEY,
    OPT_KEY,
    SEED_KEY,
    TOTAL_EPOCHS_KEY,
)

from tqdm import tqdm


log = utils.get_logger("DEBUG")


def print_memory_summary(prefix, device):
    rank = (
        torch.distributed.get_rank()
        if torch.distributed.is_available() and torch.distributed.is_initialized()
        else 0
    )
    if rank == 0:
        peak_memory_active = torch.cuda.memory_stats().get("active_bytes.all.peak", 0)
        print(
            f"{prefix}, GPU peak memory allocation: {torch.cuda.max_memory_allocated(device) // 1e9}GB, "
            f"GPU peak memory reserved: {torch.cuda.max_memory_reserved(device) // 1e9}GB, "
            f"GPU peak memory active: {peak_memory_active // 1e9}GB"
        )
    torch.cuda.reset_peak_memory_stats(device)


class MemEfficientFTRecipe(FTRecipeInterface):
    """
    Full finetuning recipe for dense transformer-based LLMs such as Llama2.

    This recipe supports:
        - FSDP and activation checkpointing. This is enabled by default but can be
            configured using the ``enable_fsdp`` and ``enable_activation_checkpointing`` flags.
        - Mixed precision training - fp32, fp16 and bf16 are supported.
        - Checkpointing of model weights, optimizer state and the recipe state (epoch and seed).
        - Resuming from checkpoints saved using the ``save_checkpoint`` functionality.
        - Logging to terminal. WandB and TensorBoard are currently not supported.

    Assumptions:
        - Training is launched with the Tune CLI (recommended) which uses TorchRun under the
            hood. Setting up the env variables is handled by TorchRun.
        - Training happens on CUDA (CPU training is not supported)
        - Checkpoints are ONLY saved at epoch boundaries. Mid-epoch checkpointing is NOT supported.
        - Datasets are Map-style and data fits in memory (not streamed).
    """

    def __init__(self, params: MemEfficientFTParams) -> None:
        self.params = params
        self._device = utils.get_device(device=params.device)
        self._dtype = utils.get_dtype(dtype=params.dtype)

        # logging attributes
        self._output_dir = params.output_dir
        self._metric_logger = utils.get_metric_logger(
            metric_logger_type=params.metric_logger_type,
            project=params.project,
            log_dir=params.output_dir,
        )
        self._log_every_n_steps = (
            params.log_every_n_steps if params.log_every_n_steps else 1
        )

        # _is_rank_zero is used primarily for logging. In the future, the logger
        # should directly take care of this
        _, rank = utils.get_world_size_and_rank()
        self._is_rank_zero = rank == 0

        # Training params
        self._resume_from_checkpoint = params.resume_from_checkpoint
        self._enable_fsdp = params.enable_fsdp
        self._gradient_accumulation_steps = params.gradient_accumulation_steps

        # These are public properties which are updated by the checkpoint loader
        # when ``resume_from_checkpoint`` is `True` or validated in tests
        self.seed = utils.set_seed(seed=params.seed)
        self.epochs_run = 0
        self.total_epochs = params.epochs
        self.max_steps_per_epoch = params.max_steps_per_epoch
        self.total_training_steps = 0

    def load_checkpoint(self, ckpt_path: str):
        """
        Extract the checkpoint state from file and validate.
        """
        ckpt_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        utils.validate_checkpoint(ckpt_dict, self._resume_from_checkpoint)
        return ckpt_dict

    def setup(self, params: MemEfficientFTParams) -> None:
        """
        Sets up the recipe state correctly. This includes setting recipe attributes based
        on the ``resume_from_checkpoint`` flag.
        """

        ckpt_dict = self.load_checkpoint(ckpt_path=params.model_checkpoint)

        # If we're resuming from checkpoint, the recipe's state should be updated before
        # initializing the training components. This ensures that the seed is correctly
        # propagated to the relevant components
        if self._resume_from_checkpoint:
            self._update_recipe_state(ckpt_dict)

        # ``_setup_model`` handles initialization and loading the state dict. This method
        # should be called before ``_setup_optimizer`` since transforming the optimizer
        # state dict requires the model
        self._model = self._setup_model(
            model=params.model,
            enable_fsdp=params.enable_fsdp,
            cpu_offload=params.cpu_offload,
            enable_activation_checkpointing=params.enable_activation_checkpointing,
            model_state_dict=ckpt_dict[MODEL_KEY],
        )
        print_memory_summary("After model init", torch.cuda.current_device())

        self._tokenizer = self._setup_tokenizer(
            tokenizer=params.tokenizer, tokenizer_checkpoint=params.tokenizer_checkpoint
        )

        # _setup_optimizer should take in ckpt_dict only if training is resumed from
        # checkpoint. Transforming the opt state dict is handled by this method
        self._optimizer = self._setup_optimizer(
            optimizer=params.optimizer,
            lr=params.lr,
            opt_state_dict=ckpt_dict[OPT_KEY] if self._resume_from_checkpoint else None,
        )

        self._loss_fn = self._setup_loss(loss=params.loss)

        # sampler and dataloader depend on the tokenizer and loss_fn and should be
        # setup after both of these are initialized
        self._sampler, self._dataloader = self._setup_data(
            dataset=params.dataset,
            train_on_input=params.train_on_input,
            shuffle=params.shuffle,
            batch_size=params.batch_size,
        )

        # training setup
        self._autocast = utils.get_autocast(self._dtype, self._device)
        self._grad_scaler = None
        if self._dtype == torch.float16:
            self._grad_scaler = utils.get_gradient_scaler(fsdp=params.enable_fsdp)
        else:
            self._grad_scaler = GradScaler(enabled=False)

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
        self.total_training_steps = self.epochs_run * self._steps_per_epoch

    def _update_recipe_state(self, ckpt_dict: Dict[str, Any]) -> None:
        """
        Updates the recipe state from checkpoint.
        """
        # If seed, total_epoch or max_steps_per_epoch don't match,
        # warn the user and overwrite
        if (
            self.seed != ckpt_dict[SEED_KEY]
            or self.total_epochs != ckpt_dict[TOTAL_EPOCHS_KEY]
            or self.max_steps_per_epoch != ckpt_dict[MAX_STEPS_KEY]
        ):
            warn(
                message="""Configured value for seed, epochs or max_steps_per_epoch
                does not match the value stored in checkpoint."""
            )
        self.seed = utils.set_seed(seed=ckpt_dict[SEED_KEY])
        self.epochs_run = ckpt_dict[EPOCHS_KEY]
        self.total_epochs = ckpt_dict[TOTAL_EPOCHS_KEY]
        self.max_steps_per_epoch = ckpt_dict[MAX_STEPS_KEY]

    def _setup_model(
        self,
        model: str,
        enable_fsdp: bool,
        cpu_offload: bool,
        enable_activation_checkpointing: bool,
        model_state_dict: Dict[str, Any],
    ) -> nn.Module:
        """
        Set up the model including enabling FSDP and activation checkpointing. For this recipe,
        ``enable_fsdp`` should always be ``True``. This is currently a configurable flag for
        running tests on CPUs.
        """
        prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        model = models.get_model(model, device=self._device)
        torch.set_default_dtype(prev_dtype)
        # assert not enable_fsdp
        print(f"RV: enable_fsdp is {enable_fsdp}, cpu_offload is {cpu_offload}")
        model = (
            utils.wrap_fsdp(
                model=model,
                device=self._device,
                dtype=self._dtype,
                strategy="FULL_SHARD",
                cpu_offload=cpu_offload,
                auto_wrap_policy={modules.TransformerDecoderLayer},
            )
            if enable_fsdp
            else model
        )
        # assert not enable_activation_checkpointing
        enable_activation_checkpointing = True
        if enable_activation_checkpointing:
            utils.set_activation_checkpointing(
                model, auto_wrap_policy={modules.TransformerDecoderLayer}
            )

        for k in model_state_dict:
            assert model_state_dict[k].dtype == torch.bfloat16

        model.load_state_dict(model_state_dict)

        if self._is_rank_zero:
            log.info("Model is initialized.")
        return model

    def _setup_tokenizer(
        self, tokenizer: str, tokenizer_checkpoint: str
    ) -> modules.Tokenizer:
        """
        Unlike ```setup_model```, this takes in the checkpoint and loads the sentencepiece
        tokenizer model. This is related to how the tokenizer is implemented and should
        change in a future iteration.
        """
        tokenizer = models.get_tokenizer(tokenizer, path=tokenizer_checkpoint)

        if self._is_rank_zero:
            log.info("Tokenizer is initialized from file.")
        return tokenizer

    def _setup_optimizer(
        self, optimizer: str, lr: float, opt_state_dict: Optional[Dict[str, Any]] = None
    ) -> Optimizer:
        """
        Set up the optimizer. This method also handles transforing the state dict
        for FSDP.
        """
        optim_str = optimizer
        if "bnb" in optimizer:
            assert optimizer == "bnb_paged_adamw", f"Only bnb_paged_adamw supported"
            import bitsandbytes as bnb
            optimizer = bnb.optim.PagedAdamW8bit(self._model.parameters(), lr=lr)
        else:
            optimizer = modules.get_optimizer(optimizer, self._model, lr)
        if opt_state_dict:
            opt_state_dict = utils.transform_opt_state_dict(
                opt_state_dict, self._model, optimizer
            )
            optimizer.load_state_dict(opt_state_dict)

        if not self.params.optim_in_bwd:
            print(f"RV: NOT optimizer in backward")
            return optimizer

        assert optim_str in ["SGD", "bnb_paged_adamw"], f"Only SGD and bnb_paged_adamw supported"
        if optim_str == "SGD":
            print(f"RV: optimizer in backward is SGD")
            _apply_optimizer_in_backward(optimizer_class=torch.optim.SGD, params=self._model.parameters(), optimizer_kwargs={"lr": 2e-5})
        else:
            import bitsandbytes as bnb
            print(f"RV: optim in backward is paged adamW")
            # optimizer = bnb.optim.PagedAdamW8bit(self._model.parameters(), lr=lr)
            # optimizer = AnyPrecisionAdamW(self._model.parameters())
            _apply_optimizer_in_backward(
                optimizer_class=bnb.optim.PagedAdamW,
                params=self._model.parameters(),
                optimizer_kwargs={"lr": lr},
            )
        if self._is_rank_zero:
            log.info("Optimizer is initialized.")
        return optimizer

    def _setup_loss(self, loss: str) -> nn.Module:
        loss_fn = modules.get_loss(loss)

        if self._is_rank_zero:
            log.info("Loss is initialized.")

        return loss_fn

    def _setup_data(
        self, dataset: str, shuffle: bool, batch_size: int, train_on_input: bool
    ) -> Tuple[DistributedSampler, DataLoader]:
        """
        All data related setup happens here. Currently this recipe only supports the
        DistributedSamplers with Map-style Datasets which fit into memory. Other samplers,
        iterable datasets and streaming datasets are not supported.
        """
        world_size, rank = utils.get_world_size_and_rank()
        ds = datasets.get_dataset(
            dataset,
            split="train",
            tokenizer=self._tokenizer,
            train_on_input=train_on_input,
        )
        sampler = DistributedSampler(
            ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=0,
        )
        dataloader = DataLoader(
            dataset=ds,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=partial(
                utils.padded_collate,
                padding_idx=self._tokenizer.pad_id,
                ignore_idx=self._loss_fn.ignore_index,  # TODO support loss without ignore_index
            ),
        )

        if self._is_rank_zero:
            log.info("Dataset and Sampler are initialized.")

        return sampler, dataloader

    def save_checkpoint(self, epoch: int) -> None:
        """
        Checkpoint the relevant state of a recipe.

        This makes use of the `save_checkpoint` utility which is responsible for
        writing the checkpoint dictionary to file. The contents of the dict are dictated
        by whether training is complete or not.

        If training is ongoing, optimizer state, seed and epochs_run are saved along with the
        model weights.
        """
        os.makedirs(self._output_dir, exist_ok=True)
        output_loc = f"{self._output_dir}/model_{epoch}.ckpt"
        ckpt_dict = {MODEL_KEY: self._model}

        # if training is in-progress, checkpoint the optimizer state as well
        if epoch + 1 < self.total_epochs:
            ckpt_dict.update(
                {
                    OPT_KEY: self._optimizer,
                    SEED_KEY: self.seed,
                    EPOCHS_KEY: self.epochs_run,
                    TOTAL_EPOCHS_KEY: self.total_epochs,
                    MAX_STEPS_KEY: self.max_steps_per_epoch,
                }
            )
        utils.save_checkpoint(ckpt_dict, output_loc)

        if self._is_rank_zero:
            log.info(
                f"Model checkpoint of size {os.path.getsize(output_loc) >> 20} MB saved to {output_loc}"
            )

    def _should_update_weights(self, curr_step: int) -> bool:
        """
        Determines whether the weights should be updated on the current step or not.
        True is returned either if we've accumulated gradients for enough steps or if this
        is the last step in the epoch.
        """
        should_update_weights = (
            curr_step + 1
        ) % self._gradient_accumulation_steps == 0 or (
            curr_step + 1
        ) == self._steps_per_epoch
        return should_update_weights

    def train(self) -> None:
        """
        The core training loop. Supports training on subsets of the dataset using the
        ``max_steps_per_epoch``.
        """
        _, rank = utils.get_world_size_and_rank()
        print_memory_summary("Before first zero_grad", torch.cuda.current_device())
        # zero out the gradients before starting training
        self._optimizer.zero_grad()
        print_memory_summary(
            "After first zero_grad, before first forward", torch.cuda.current_device()
        )

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
                    and (idx // self._gradient_accumulation_steps)
                    == self.max_steps_per_epoch
                ):
                    break

                input_ids, labels = batch
                input_ids = input_ids.to(self._device)
                labels = labels.to(self._device)

                with self._autocast:
                    logits = self._model(input_ids)
                    # Shift so that tokens < n predict n
                    logits = logits[..., :-1, :].contiguous()
                    labels = labels[..., 1:].contiguous()
                    logits = logits.transpose(1, 2)
                    # Compute loss
                    loss = self._loss_fn(logits, labels)
                    print_memory_summary(
                        "After FWD / loss compute", torch.cuda.current_device()
                    )

                # Note: We're always logging the loss before normalizing it
                # Check if this is the norm or not
                pbar.set_description(f"{curr_epoch+1}|{idx+1}|Loss: {loss.item()}")

                if self.total_training_steps % self._log_every_n_steps == 0:
                    self._metric_logger.log_dict(
                        {
                            "loss": loss.item(),
                            "lr": self._optimizer.param_groups[0]["lr"],
                            "gpu_resources": torch.cuda.memory_allocated(),
                        },
                        step=self.total_training_steps,
                    )

                # Does loss normalization need to happen within autocast context?
                loss = loss / self._gradient_accumulation_steps
                self._grad_scaler.scale(loss).backward()
                print_memory_summary("After BWD", torch.cuda.current_device())
                # Offload model to CPU
                # self._model.cpu()

                if self._should_update_weights(idx):
                    self._grad_scaler.step(self._optimizer)
                    print_memory_summary(
                        "After optim step", torch.cuda.current_device()
                    )
                    self._grad_scaler.update()
                    self._optimizer.zero_grad(set_to_none=True)
                    print_memory_summary("After zero_grad", torch.cuda.current_device())

                    # Update the number of steps when the weights are updated
                    self.total_training_steps += 1

                # self._model.cpu()

            self.epochs_run += 1
            self.save_checkpoint(epoch=curr_epoch)

    def cleanup(self) -> None:
        self._metric_logger.close()


class AnyPrecisionAdamW(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 3e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        use_kahan_summation: bool = False,
        momentum_dtype: torch.dtype = torch.bfloat16,
        variance_dtype: torch.dtype = torch.bfloat16,
        compensation_buffer_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        """
        AnyPrecisionAdamW: a flexible precision AdamW optimizer
        with optional Kahan summation for high precision weight updates.
        Allows direct control over momentum, variance and auxiliary compensation
        buffer dtypes.
        Optional Kahan summation is used to offset precision reduction for
        the weight updates. This allows full training in BFloat16 (can be equal or
        better than FP32 results in many cases) due to high precision weight upates.

        This optimizer is the same AnyPrecision that was previously residing
        in Torch.DistX, now moved to TorchMM for easier install:
        https://github.com/pytorch/torchdistx/blob/main/src/python
        /torchdistx/optimizers/anyprecision_optimizer.py

        Kahan summation overview:
        https://en.wikipedia.org/wiki/Kahan_summation_algorithm

        "In numerical analysis, the Kahan summation algorithm, also known as compensated
        summation, significantly reduces the numerical error in the total obtained by
        adding a sequence of finite-precision floating-point numbers, compared to the
        obvious approach. This is done by keeping a separate running compensation
        (a variable to accumulate small errors), in effect extending the precision of
        the sum by the precision of the compensation variable."

        Args:
            params (iterable): iterable of parameters to optimize or dicts defining
                parameter groups
            lr (float, optional): learning rate (default: 1e-3)
            betas (Tuple[float, float], optional): coefficients used for computing
                running averages of gradient and its square (default: (0.9, 0.999))
            eps (float, optional): term added to the denominator to improve
                numerical stability (default: 1e-8)
            weight_decay (float, optional): weight decay coefficient (default: 1e-2)

            # Any Precision specific
            use_kahan_summation = creates auxiliary buffer to ensure high precision
            model param updates (default: False)
            momentum_dtype = dtype for momentum  (default: BFloat32)
            variance_dtype = dtype for uncentered variance (default: BFloat16)
            compensation_buffer_dtype  = dtype for Kahan summation
                                         buffer (default: BFloat16). Only used if
                                         ``use_kahan_summation=True``.

            # Usage
            This optimizer implements optimizer states, and Kahan summation
            for high precision updates, all in user controlled dtypes.
            Defaults are variance in BF16, Momentum in FP32.
            This can be run in FSDP mixed precision, amp, or full precision,
            depending on what training pipeline you wish to work with.

            Setting to use_kahan_summation = False, and changing momentum and
            variance dtypes to FP32, reverts this to a standard AdamW optimizer.
        """
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            use_kahan_summation=use_kahan_summation,
            momentum_dtype=momentum_dtype,
            variance_dtype=variance_dtype,
            compensation_buffer_dtype=compensation_buffer_dtype,
        )

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Any = None) -> None:
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        if closure is not None:
            with torch.enable_grad():
                # to fix linter, we do not keep the returned loss for use atm.
                closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            use_kahan_summation = group["use_kahan_summation"]

            momentum_dtype = group["momentum_dtype"]
            variance_dtype = group["variance_dtype"]
            compensation_buffer_dtype = group["compensation_buffer_dtype"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                pd = p.device
                p.data = p.cpu()
                p.grad.data = p.grad.cpu()

                dev = p.grad.device
                p.grad = p.grad.cpu()
                if p.grad.is_sparse:
                    raise RuntimeError(
                        "AnyPrecisionAdamW does not support sparse gradients"
                    )

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0)

                    # momentum - EMA of gradient values
                    state["exp_avg"] = torch.zeros_like(
                        p,
                        dtype=momentum_dtype,
                    )

                    # variance uncentered - EMA of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(
                        p,
                        dtype=variance_dtype,
                    )

                    # optional Kahan summation - accumulated error tracker
                    if use_kahan_summation:
                        state["compensation"] = torch.zeros_like(
                            p,
                            dtype=compensation_buffer_dtype,
                        )

                # main processing -------------------------

                # update the steps for each param group update
                state["step"] += 1
                step = state["step"]

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                grad = p.grad

                # import pdb ; pdb.set_trace()
                # weight decay, AdamW style
                if weight_decay:
                    p.data.mul_(1 - lr * weight_decay)

                # update momentum
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # update uncentered variance
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # adjust using bias1
                bias_correction1 = 1 - beta1**step

                step_size = lr / bias_correction1

                # adjust using bias2
                denom_correction = (1 - beta2**step) ** 0.5  # avoids math import

                centered_variance = (exp_avg_sq.sqrt() / denom_correction).add_(eps)

                # lr update to compensation
                if use_kahan_summation:
                    compensation = state["compensation"]

                    compensation.addcdiv_(exp_avg, centered_variance, value=-step_size)

                    # update weights with compensation (Kahan summation)
                    # save error back to compensation for next iteration
                    temp_buffer = p.detach().clone()
                    p.data.add_(compensation)
                    compensation.add_(temp_buffer.sub_(p.data))

                else:
                    # usual AdamW updates
                    p.data.addcdiv_(exp_avg, centered_variance, value=-step_size)

                p.data = p.data.to(pd)
                # p.grad.data = p.grad.to(pd)

def recipe_main() -> None:
    """
    Entry point for the recipe.

    Configurable parameters are read in the following order:
        - Parameters specified in ``MemEfficientFTParams``
        - Overwritten by Parameters specified in ``alpaca_llama2_full_finetune.yaml``
        - Overwritten by arguments from the command-line using ``TuneArgumentParser``
    """
    parser = utils.TuneArgumentParser(
        description=MemEfficientFTParams.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    args, _ = parser.parse_known_args()
    args = vars(args)
    recipe_params = MemEfficientFTParams(**args)

    # Env variables set by torch run; only need to initialize process group
    init_process_group(backend="nccl")
    torch.cuda.set_per_process_memory_fraction(0.2, device=torch.cuda.current_device())
    recipe = MemEfficientFTRecipe(params=recipe_params)
    recipe.setup(params=recipe_params)
    recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())
