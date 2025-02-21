# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
import time
from typing import Any, Dict, Optional, Union
from warnings import warn

import torch

import torchtune.modules.common_utils as common_utils
from omegaconf import DictConfig, ListConfig
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchtune import config, modules, training, utils
from torchtune.datasets import ConcatDataset
from torchtune.modules.peft import (
    get_adapter_params,
    get_adapter_state_dict,
    get_merged_lora_ckpt,
    set_trainable_params,
)
from torchtune.recipe_interfaces import FTRecipeInterface
from torchtune.training import DummyProfiler, PROFILER_KEY
from tqdm import tqdm

log = utils.get_logger("DEBUG")


class LoRALatentDiffusionSingleDevice(FTRecipeInterface):
    """
    LoRA finetuning recipe for latent diffusion/flow-matching text-to-image models.

    Args:
        cfg (DictConfig): OmegaConf object parsed from yaml file

    Raises:
        ValueError: If ``dtype`` is set to fp16.
        RuntimeError: If ``dtype`` is set to bf16 and the hardware does not support bf16.
        RuntimeError: If ``enable_activation_offloading`` is True and device is not CUDA.
        RuntimeError: If ``enable_activation_offloading`` is True and ``enable_activation_checkpointing`` is False.
        RuntimeError: If ``left_pad_sequence`` is set as the data collator

    """

    def __init__(self, cfg: DictConfig) -> None:
        self._device = utils.get_device(device=cfg.device)
        self._dtype = training.get_dtype(cfg.dtype, device=self._device)

        # logging attributes
        self._output_dir = cfg.output_dir
        self._log_every_n_steps = cfg.get("log_every_n_steps", 1)
        self._log_peak_memory_stats = cfg.get("log_peak_memory_stats", False)
        if self._log_peak_memory_stats and self._device.type == "cpu":
            log.info(
                "log_peak_memory_stats was set to True, however, training uses cpu. Setting log_peak_memory_stats=False."
            )
            self._log_peak_memory_stats = False

        # These are public properties which are updated by the checkpoint loader
        # when ``resume_from_checkpoint`` is `True` or validated in tests
        self.seed = training.set_seed(seed=cfg.seed)
        self.total_steps = cfg.total_steps
        self.cur_step = 0
        self._resume_from_checkpoint = cfg.resume_from_checkpoint
        self._save_adapter_weights_only = cfg.get("save_adapter_weights_only", False)
        self._gradient_accumulation_steps = cfg.gradient_accumulation_steps
        self._clip_grad_norm = cfg.get("clip_grad_norm", None)

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

        self._sample_every_n_steps = cfg.sample_every_n_steps
        self._checkpoint_every_n_steps = cfg.checkpoint_every_n_steps

    def load_checkpoint(self, cfg_checkpointer: DictConfig) -> Dict[str, Any]:
        """
        Extract the checkpoint state from file and validate. This includes the
        base model weights. If resume_from_checkpoint is True, this also includes
        the adapter weights and recipe state
        """
        self._checkpointer = config.instantiate(
            cfg_checkpointer,
            should_load_recipe_state=self._resume_from_checkpoint,
        )
        checkpoint_dict = self._checkpointer.load_checkpoint()

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
            self.cur_step = ckpt_dict[training.STEPS_KEY]

            # on mismatch, warn the user and prevent the override
            if self.seed != ckpt_dict[training.SEED_KEY]:
                warn(
                    message=(
                        "Config value for seed does not match the checkpoint value, "
                        f"using the checkpoint value: {ckpt_dict[training.SEED_KEY]}"
                    )
                )
                self.seed = ckpt_dict[training.SEED_KEY]

            # on mismatch, warn the user but allow the override
            if self.total_steps != ckpt_dict[training.TOTAL_STEPS_KEY]:
                warn(
                    message=(
                        "Config value for total_steps does not match the checkpoint value, "
                        f"using the config value: {self.total_steps}"
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

        # transformation from general-purpose data to model-specific data
        transform = config.instantiate(cfg.transform)

        # preprocessor for the training dataset and sample prompts
        preprocessor = config.instantiate(cfg.preprocess, self._device, self._dtype)

        # preprocess the dataset and create a dataloader for preprocessed data
        self._dataloader = self._setup_data(
            cfg_dataset=cfg.dataset,
            transform=transform,
            preprocessor=preprocessor,
            shuffle=cfg.shuffle,
            batch_size=cfg.batch_size,
        )

        # sampler for generating samples during training
        self._sampler = config.instantiate(
            cfg.sampler, preprocessor, transform.tokenize, self._device, self._dtype
        )

        # free up memory
        del transform, preprocessor

        # metric logging
        self._metric_logger = config.instantiate(cfg.metric_logger)
        self._metric_logger.log_config(cfg)

        # model compilation
        self._compile = cfg.compile
        if cfg.device == "npu" and cfg.compile:
            raise ValueError(
                "NPU does not support model compilation. Please set `compile: False` in the config."
            )

        # hack to toggle to the low cpu ram version of the reparametrize_as_dtype
        # hook based on the config.
        common_utils._use_low_cpu_ram = cfg.get("low_cpu_ram", False)

        # set up model
        checkpoint_dict = self.load_checkpoint(cfg_checkpointer=cfg.checkpointer)
        self._model = self._setup_model(
            cfg_model=cfg.model,
            enable_activation_checkpointing=self._enable_activation_checkpointing,
            enable_activation_offloading=self._enable_activation_offloading,
            compile_model=cfg.compile,
            base_model_state_dict=checkpoint_dict[training.MODEL_KEY],
            lora_weights_state_dict=(
                checkpoint_dict[training.ADAPTER_KEY]
                if self._resume_from_checkpoint
                else None
            ),
        )

        # set up optimizer
        self._optimizer = self._setup_optimizer(
            cfg_optimizer=cfg.optimizer,
            opt_state_dict=(
                checkpoint_dict[training.OPT_KEY]
                if self._resume_from_checkpoint
                else None
            ),
        )

        # set up loss step
        self._loss_step = config.instantiate(cfg.loss)

        # Learning rate scheduler can only be set up after number of steps
        # has been computed
        self._lr_scheduler = self._setup_lr_scheduler(
            cfg_lr_scheduler=cfg.get("lr_scheduler", None),
            num_training_steps=self.total_steps,
            last_epoch=self.cur_step - 1,
        )

        # Set up profiler, returns DummyProfiler (nullcontext object with no-op `step` method)
        # if cfg is missing profiler key or if `cfg.profiler.enabled = False
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
        enable_activation_offloading: bool,
        compile_model: bool,
        base_model_state_dict: Dict[str, Any],
        lora_weights_state_dict: Optional[Dict[str, Any]] = None,
    ) -> nn.Module:
        with training.set_default_dtype(self._dtype), self._device:
            model = config.instantiate(cfg_model)

        self._lora_rank = cfg_model.lora_rank
        self._lora_alpha = cfg_model.lora_alpha
        self.adapter_params = get_adapter_params(model)
        self._is_dora = any(["magnitude" in k for k in self.adapter_params.keys()])
        set_trainable_params(model, self.adapter_params)
        print("Total params:", sum(p.numel() for p in model.parameters()))
        print(
            "Trainable params:",
            sum(p.numel() for p in model.parameters() if p.requires_grad),
        )

        if compile_model:
            training.compile_model(model)

        if enable_activation_checkpointing:
            training.set_activation_checkpointing(
                model, auto_wrap_policy={modules.TransformerSelfAttentionLayer}
            )

        model.load_state_dict(base_model_state_dict, strict=False)
        # This is for any adapters that need to be initialized after base weights
        # have been loaded (e.g. DoRA).
        if self._is_dora:
            for m in model.modules():
                if hasattr(m, "initialize_dora_magnitude"):
                    m.initialize_dora_magnitude()
        if lora_weights_state_dict:
            model.load_state_dict(lora_weights_state_dict, strict=False)

        # Validate model adapter params were loaded in with the expected dtype
        training.validate_expected_param_dtype(
            self.adapter_params.items(), dtype=self._dtype
        )

        # activation offloading
        self.activations_handling_ctx = training.get_act_offloading_ctx_manager(
            model, enable_activation_offloading
        )

        log.info(f"Model is initialized with precision {self._dtype}.")

        if self._device.type != "cpu":
            memory_stats = training.get_memory_stats(device=self._device)
            training.log_memory_stats(memory_stats)
        return model

    def _setup_optimizer(
        self, cfg_optimizer: DictConfig, opt_state_dict: Optional[Dict[str, Any]] = None
    ) -> Optimizer:
        optimizer = config.instantiate(cfg_optimizer, self._model.parameters())
        if opt_state_dict:
            optimizer.load_state_dict(opt_state_dict)

        log.info("Optimizer and loss are initialized.")
        return optimizer

    def _setup_lr_scheduler(
        self,
        cfg_lr_scheduler: Optional[DictConfig],
        num_training_steps: int,
        last_epoch: int,
    ) -> Optional[Optimizer]:
        if cfg_lr_scheduler is None:
            return None
        lr_scheduler = config.instantiate(
            cfg_lr_scheduler,
            self._optimizer,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch,
        )
        log.info("Learning rate scheduler is initialized.")
        return lr_scheduler

    def _setup_data(
        self,
        cfg_dataset: DictConfig,
        transform,
        preprocessor,
        shuffle: bool,
        batch_size: int,
    ) -> DataLoader:
        """
        All data related setup happens here. Currently this recipe only supports
        Map-style Datasets which fit into memory and an option for random shuffling.
        Samplers, iterable datasets, and streaming datasets are not supported.
        """
        if isinstance(cfg_dataset, ListConfig):
            datasets = [
                config.instantiate(single_cfg_dataset, transform)
                for single_cfg_dataset in cfg_dataset
            ]
            ds = ConcatDataset(datasets=datasets)
        else:
            ds = config.instantiate(cfg_dataset, transform)

        # preprocess the dataset
        preprocessed_ds = preprocessor.preprocess_dataset(ds)

        # create a dataloader for preprocessed data
        dataloader = DataLoader(
            dataset=preprocessed_ds,
            batch_size=batch_size,
            shuffle=shuffle,
            # dropping last avoids shape issues with compile + flex attention
            drop_last=True,
        )

        log.info("Dataloader is initialized.")
        return dataloader

    def save_checkpoint(self) -> None:
        """
        Checkpoint the state of the recipe. The constructed checkpoint state dict
        contains the following information:
        - Merged weights with key MODEL_KEY
        - Adapter weights with key ADAPTER_KEY
        - Relevant recipe state if training is not complete
        - If the `self._save_adapter_weights_only` option is True, the checkpointer will save only the adapter weights

        To correctly resume from training, the adapter weights and recipe state must be provided along with the base model weights.
        """
        ckpt_dict = {}

        intermediate_checkpoint = self.cur_step < self.total_steps
        # if training is in-progress, checkpoint the optimizer state as well
        if intermediate_checkpoint:
            ckpt_dict.update(
                {
                    training.OPT_KEY: self._optimizer.state_dict(),
                    training.SEED_KEY: self.seed,
                    training.STEPS_KEY: self.cur_step,
                    training.TOTAL_STEPS_KEY: self.total_steps,
                }
            )

        adapter_state_dict = get_adapter_state_dict(self._model.state_dict())
        ckpt_dict.update({training.ADAPTER_KEY: adapter_state_dict})

        if not self._save_adapter_weights_only:
            # Construct the full state dict with LoRA weights merged into base LLM weights

            # Move to CPU to avoid a copy on GPU
            state_dict = {k: v.cpu() for k, v in self._model.state_dict().items()}

            merged_state_dict = get_merged_lora_ckpt(
                state_dict,
                rank=self._lora_rank,
                alpha=self._lora_alpha,
            )

            ckpt_dict.update({training.MODEL_KEY: merged_state_dict})

        adapter_config = {
            "r": self._lora_rank,
            "lora_alpha": self._lora_alpha,
            "peft_type": "LORA",
        }
        ckpt_dict.update({training.ADAPTER_CONFIG: adapter_config})

        self._checkpointer.save_checkpoint(
            ckpt_dict,
            epoch=self.cur_step,
            intermediate_checkpoint=intermediate_checkpoint,
            adapter_only=self._save_adapter_weights_only,
        )

    def train(self) -> None:
        """
        The core training loop.
        """
        if self._compile:
            log.info(
                "NOTE: torch.compile is enabled and model is compiled in first forward. Expect a relatively slow first iteration."
            )

        # Initialize tokens count and running loss (for grad accumulation)
        t0 = time.perf_counter()
        running_loss = 0
        idx = 0

        with self._profiler as prof:
            pbar = tqdm(total=self.total_steps)
            while self.cur_step < self.total_steps:
                for batch in self._dataloader:
                    if self.cur_step >= self.total_steps:
                        break

                    if (
                        self.profiler_profile_memory
                        and idx == self.profiler_wait_steps + self.profiler_warmup_steps
                    ):
                        torch.cuda.memory._record_memory_history()

                    utils.batch_to_device(batch, self._device)

                    current_loss = self._loss_step(
                        self._model, batch, self.activations_handling_ctx
                    )
                    running_loss += current_loss
                    current_loss.backward()

                    # Step with optimizer
                    if (idx + 1) % self._gradient_accumulation_steps == 0:
                        if self._clip_grad_norm is not None:
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                self._model.parameters(),
                                max_norm=float(self._clip_grad_norm),
                            )
                        self._optimizer.step()
                        self._optimizer.zero_grad(set_to_none=True)
                        if self._lr_scheduler is not None:
                            self._lr_scheduler.step()
                        # Update the number of steps when the weights are updated
                        self.cur_step += 1

                        loss_to_log = running_loss.item()
                        pbar.update(1)
                        pbar.set_description(f"{self.cur_step}|Loss: {loss_to_log}")

                        # Log per-step metrics
                        if self.cur_step % self._log_every_n_steps == 0:
                            time_per_step = time.perf_counter() - t0
                            log_dict = {
                                "loss": loss_to_log,
                                "lr": self._optimizer.param_groups[0]["lr"],
                                "time_per_step": time_per_step,
                            }
                            if (
                                self._device.type != "cpu"
                                and self._log_peak_memory_stats
                            ):
                                log_dict.update(
                                    training.get_memory_stats(device=self._device)
                                )
                            if self._clip_grad_norm is not None:
                                log_dict.update({"grad_norm": grad_norm})
                            self._metric_logger.log_dict(
                                log_dict,
                                step=self.cur_step,
                            )

                        if self.cur_step % self._sample_every_n_steps == 0:
                            self._model.eval()
                            with torch.no_grad():
                                self._sampler.save_samples(self._model, self.cur_step)
                            self._model.train()

                        if self.cur_step % self._checkpoint_every_n_steps == 0:
                            start_save_checkpoint = time.perf_counter()
                            log.info("Starting checkpoint save...")
                            self.save_checkpoint()
                            log.info(
                                "Checkpoint saved in {:.2f} seconds.".format(
                                    time.perf_counter() - start_save_checkpoint
                                )
                            )

                        # Reset running stats for the next step
                        running_loss = 0
                        t0 = time.perf_counter()

                    # Stop tracking CUDA memory now that active steps are complete
                    if (
                        self.profiler_profile_memory
                        and idx
                        == self.profiler_wait_steps
                        + self.profiler_warmup_steps
                        + self.profiler_active_steps
                    ):
                        torch.cuda.memory._record_memory_history(enabled=None)

                    # Step the profiler
                    # Note we are stepping each batch, which might not include optimizer step in the trace
                    # if the schedule cycle doesn't align with gradient accumulation.
                    prof.step()

                    idx += 1

            start_save_checkpoint = time.perf_counter()
            log.info("Saving final checkpoint...")
            self.save_checkpoint()
            log.info(
                "Checkpoint saved in {:.2f} seconds.".format(
                    time.perf_counter() - start_save_checkpoint
                )
            )

    def cleanup(self) -> None:
        self._metric_logger.close()


@config.parse
def recipe_main(cfg: DictConfig) -> None:
    """
    Entry point for the recipe.

    Configurable parameters are read in the following order:
        - Parameters specified in config (see available configs through ``tune ls``)
        - Overwritten by arguments from the command-line
    """
    config.log_config(recipe_name="LoRALatentDiffusionSingleDevice", cfg=cfg)
    recipe = LoRALatentDiffusionSingleDevice(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())
