# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
import sys
from functools import partial
from itertools import chain
from typing import Any, Dict, List, Optional, Tuple
from warnings import warn

import torch
from omegaconf import DictConfig, ListConfig
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler
from torchtune import config, generation, modules, rlhf, training, utils
from torchtune.data import padded_collate
from torchtune.datasets import ConcatDataset
from torchtune.recipe_interfaces import FTRecipeInterface
from torchtune.rlhf import PPOStats, Trajectory
from tqdm import tqdm

log = utils.get_logger("DEBUG")


class PPOFullFinetuneRecipeSingleDevice(FTRecipeInterface):
    """
    Full finetuning recipe for RLHF with PPO for dense transformer-based LLMs such as LLama2. This recipe is optimized
    for single GPU training. Training on CPU is not supported.

    This implementation is based on `Learning to summarize from human feedback <https://arxiv.org/abs/2009.01325`_ and
    `Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback <https://arxiv.org/abs/2204.05862`_.

    Features:
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

        - Adjusting batch sizes when memory constrained. This recipe uses three different batch sizes:
            - ``batch_size`` controls the total number of samples which are sampled from the dataset for a single trajectory.
            - ``forward_batch_size`` controls the mini-batch size for trajectory generation. Since gradients are disabled
                during trajectory generation, memory consumption is lower and this can be higher than ``ppo_batch_size``.
            - ``ppo_batch_size`` controls the number of samples used for a single optimization step during PPO optimization.
                Since we're optimizing two models at once, adjusting this parameter can have a big impact during training.

        - Gradient Accumulation. You can simulate larger ``ppo_batch_size`` sizes by accumulating gradients. This is
            controlled using the ``gradient_accumulation_steps`` flag.

            For example: with ``ppo_batch_size``=32 and ``gradient_accumulation_steps``=16, each backward pass during
            PPO optimization uses a 'micro batch size' of 2.

            Gradient accumulation is especially useful when you are memory constrained. In this case,
            accumulating gradients might give you better training speed than enabling activation
            checkpointing.

        - Optimizer in Backward. Fusing the optimizer step into the backward pass helps reduce the memory
            footprint associated with gradients. This can be especially helpful when you are memory
            constrained. Note that users can only use ONE of gradient accumulation or optimizer in backward.
            These features currently do not work together. For more details on optimizer in backward, please
            see this tutorial: https://pytorch.org/tutorials/intermediate/optimizer_step_in_backward_tutorial.html

            This paramater can provide significant performance gains, since there the number of optimization steps
            scales with ``ppo_epochs`` and ``batch_size``. Depending on the maximum sequence length sampled from the dataset,
            we've found that setting ``ppo_batch_size`` to the highest you can fit in memory, and `optimizer_in_bwd=True` to
            provide significant memory savings.

        - Lower precision optimizers. This recipe supports lower-precision optimizers from the bitsandbytes
            library (https://huggingface.co/docs/bitsandbytes/main/en/index). We've tested the recipe with
            8-bit AdamW and Paged AdamW. These optimizers are especially helpful when you are memory constrained
            since they help reduce the memory footprint associated with the optimizer states.

        - Checkpointing. Model weights are checkpointed both at the end of each epoch, and at the end of
            training. Optimizer State and recipe state (seed, total_epochs, number of epochs run etc) are
            only saved at the end of a given epoch and used in case of resuming training.

            Resuming training is controlled by the ``resume_from_checkpoint`` flag. Mid-epoch checkpointing is
            currently not supported.

            For more details on the checkpointer, please take a look at
            our checkpointer deepdive (https://pytorch.org/torchtune/main/deep_dives/checkpointer.html).

        - Logging. Terminal, Disk, WandB and TensorBoard are all supported.

    Args:
        cfg (DictConfig): OmegaConf object parsed from yaml file

    Raises:
        RuntimeError: If ``dtype`` is set to fp16.
    """

    def __init__(self, cfg: DictConfig) -> None:

        self._device = utils.get_device(device=cfg.device)
        self._dtype = training.get_dtype(cfg.dtype, device=self._device)

        # Disable for fp16, as we haven't validated "full" fp16 with this recipe, nor
        # enabled necessary features such as gradient scaling.
        if self._dtype == torch.float16:
            raise RuntimeError(
                "full fp16 training is not supported with this recipe. Please use bf16 or fp32 instead."
            )

        # logging attributes
        self._output_dir = cfg.output_dir
        self._log_every_n_steps = cfg.get("log_every_n_steps", 1)
        self._log_peak_memory_stats = cfg.get("log_peak_memory_stats", False)

        # These are public properties which are updated by the checkpoint loader
        # when ``resume_from_checkpoint`` is `True` or validated in tests
        self.seed = training.set_seed(seed=cfg.seed)
        # manually setting up a generator for the recipe
        self._rng = torch.Generator(self._device).manual_seed(self.seed)
        self._total_steps = 0
        self._steps_run = 0
        self._total_epochs = 0
        self._epochs_run = 0
        self.global_step = 0

        # Training cfg
        self._resume_from_checkpoint = cfg.resume_from_checkpoint
        self._gradient_accumulation_steps = cfg.gradient_accumulation_steps

    def setup(self, cfg: DictConfig) -> None:
        """
        Sets up the recipe state correctly. This includes setting recipe attributes based
        on the ``resume_from_checkpoint`` flag.
        """
        self._metric_logger = config.instantiate(cfg.metric_logger)

        # log config with parameter override
        self._metric_logger.log_config(cfg)

        # setup checkpointers
        (
            self._policy_checkpointer,
            ref_policy_checkpointer,
            self._value_checkpointer,
            reward_checkpointer,
        ) = self._setup_checkpointers(
            cfg.checkpointer,
            cfg.ref_policy_checkpointer,
            cfg.value_checkpointer,
            cfg.reward_checkpointer,
        )

        # load policy checkpoints
        policy_model_checkpoint_dict = self._policy_checkpointer.load_checkpoint()
        ref_policy_state_dict = ref_policy_checkpointer.load_checkpoint()

        # load reward and value model checkpoints
        value_model_checkpoint_dict = self._value_checkpointer.load_checkpoint()
        reward_model_state_dict = reward_checkpointer.load_checkpoint()

        # update recipe state
        # ``_setup_model`` handles initialization and loading the state dict. This method
        # should be called before ``_setup_optimizer`` since transforming the optimizer
        # state dict requires the model
        self._model_compile = cfg.compile
        self._optimizer_in_bwd = cfg.optimizer_in_bwd
        (
            self._policy_model,
            self._value_model,
            self._reward_model,
            self._ref_policy_model,
        ) = self._setup_models(
            cfg_model=cfg.policy_model,
            cfg_reward_value_model=cfg.reward_and_value_model,
            enable_activation_checkpointing=cfg.enable_activation_checkpointing,
            compile_model=self._model_compile,
            policy_state_dict=policy_model_checkpoint_dict[training.MODEL_KEY],
            ref_policy_state_dict=ref_policy_state_dict[training.MODEL_KEY],
            value_model_state_dict=value_model_checkpoint_dict[training.MODEL_KEY],
            reward_model_state_dict=reward_model_state_dict[training.MODEL_KEY],
        )

        # setup tokenizer
        self._tokenizer = config.instantiate(cfg.tokenizer)
        log.info("Tokenizer is initialized from file.")

        # _setup_optimizer should take in ckpt_dict only if training is resumed from
        # checkpoint. Transforming the opt state dict is handled by this method
        self._optimizer = self._setup_optimizer(
            cfg_optimizer=cfg.optimizer,
            optimizer_in_bwd=cfg.optimizer_in_bwd,
            opt_state_dict=(
                policy_model_checkpoint_dict[training.OPT_KEY]
                if self._resume_from_checkpoint
                else None
            ),
        )

        self._loss_fn = config.instantiate(cfg.loss)
        log.info("Loss is initialized.")

        # sampler and dataloader depends on the tokenizer and should be set
        # setup afterit is initialized
        self._sampler, self._dataloader = self._setup_data(
            cfg_dataset=cfg.dataset,
            shuffle=cfg.shuffle,
            batch_size=cfg.batch_size,
        )

        self._setup_training_parameters(cfg)
        self._setup_training_hyperparameters(cfg)

        if self._resume_from_checkpoint:
            self._update_recipe_state(policy_model_checkpoint_dict)

        # one "step" is a single gradient update update over a minibatch of trajectories
        self.global_step = (
            self._steps_run
            * self._ppo_epochs
            * (self.batch_size // self._ppo_batch_size)
        )

    def _setup_training_hyperparameters(self, cfg) -> None:
        """
        Sets up the training hyperparameters for the recipe. This includes the GAE hyperparameters,
        generation hyperparameters, reward masking hyperparameters, and stop token ids.
        """

        self._kl_coeff = cfg.kl_coeff
        # GAE hyperparameters
        self._gamma = cfg.gamma
        self._lmbda = cfg.lmbda
        self._whiten_rewards = cfg.whiten_rewards

        # trajectory generation args
        self._temperature = cfg.temperature
        self._top_k = cfg.top_k
        self._max_generated_tokens = cfg.max_generated_tokens

        # reward masking args
        self._min_response_length = cfg.min_response_length
        self._penalise_no_eos = cfg.penalise_no_eos
        self._reward_penalty = cfg.reward_penalty

        # lots of hand holding for stop tokens
        if cfg.get("stop_token_ids", False):
            stop_token_ids = cfg.stop_token_ids
            if self._tokenizer.eos_id not in stop_token_ids:
                warn(
                    f"tokenizer eos_id ({self._tokenizer.eos_id}) is not in stop_token_ids ({stop_token_ids})."
                    "This may lead to unexpected behaviour."
                )
        else:
            if not hasattr(self._tokenizer.stop_tokens):
                warn(
                    "No stop tokens defined in tokenizer, and no stop_token_ids provided. This may lead to unexpected behaviour."
                )
                stop_token_ids = []
            else:
                stop_token_ids = self._tokenizer.stop_tokens
        self._stop_token_ids = torch.tensor(stop_token_ids, device=self._device)

    def _setup_training_parameters(self, cfg: DictConfig) -> None:
        """
        Validates and sets up parameters for used during training and for tracking training state,
        batch sizes for model forward passes during trajectory generation, PPO minibatches, and
        PPO microbatches for gradient accumulation.

        Raises
            - ValueError if:
                - batch_size is not divisible by forward_batch_size
                - batch_size is not divisible by ppo_batch_size
                - ppo_batch_size is not divisible by gradient_accumulation_steps
                - num_steps is less than batch_size
                - gradient_accumulation_steps > 1 and optimizer_in_bwd is True
        """
        self.batch_size = cfg.batch_size
        self._forward_batch_size = cfg.forward_batch_size
        self._ppo_epochs = cfg.ppo_epochs
        self._ppo_batch_size = cfg.ppo_batch_size
        self._gradient_accumulation_steps = cfg.gradient_accumulation_steps
        self._ppo_backward_batch_size = (
            cfg.ppo_batch_size // self._gradient_accumulation_steps
        )

        if self.batch_size % self._forward_batch_size != 0:
            raise ValueError(
                f"batch_size ({self.batch_size}) must be exactly divisible by "
                f"forward_batch_size ({self._forward_batch_size})."
            )
        if self.batch_size % self._ppo_batch_size != 0:
            raise ValueError(
                f"batch_size ({self.batch_size}) must be exactly divisible by "
                f"ppo_batch_size ({self._ppo_batch_size})."
            )
        if self._ppo_batch_size % self._gradient_accumulation_steps != 0:
            raise ValueError(
                f"ppo_batch_size ({self._ppo_batch_size}) must be exactly divisible "
                f"by gradient_accumulation_steps ({self._gradient_accumulation_steps})."
            )

        if self._gradient_accumulation_steps > 1 and self._optimizer_in_bwd:
            raise RuntimeError(
                "Gradient accumulation is not supported with optimizer in bwd."
                "Please set gradient_accumulation_steps=1, or optimizer_in_bwd=False."
            )

        self._total_steps = cfg.num_steps // self.batch_size
        batches_per_epoch = max(
            1, len(self._dataloader)
        )  # when we only have a single batch in the dataset

        self._total_epochs = math.ceil(self._total_steps / batches_per_epoch)
        if self._total_steps == 0:
            raise ValueError(
                f"num_steps {cfg.num_steps} must be greater than the batch size {self.batch_size}."
            )
        if self._total_steps < len(self._dataloader):
            warn(
                f"There are fewer total steps ({self._total_steps}, (num_steps//batch_size) "
                f"than there are batches ({len(self._dataloader)}) in the dataset. "
                f"Training will stop after ({self._total_steps}) steps without saving intermediate checkpoints"
            )
        if (self._total_steps > batches_per_epoch) and (
            self._total_steps % batches_per_epoch != 0
        ):
            warn(
                f"num_steps ({cfg.num_steps}) is not exactly divisible by "
                f"the number of batches in the dataset ({batches_per_epoch}). "
                f"Intermediate checkpoints will only be saved every {batches_per_epoch} steps."
            )
        log.info(
            f"Total steps to run: {self._total_steps}, Total epochs to run: {self._total_epochs}"
        )

    def _setup_checkpointers(
        self,
        policy_cfg: DictConfig,
        ref_policy_cfg: DictConfig,
        value_cfg: DictConfig,
        reward_cfg: DictConfig,
    ) -> Tuple[
        training.Checkpointer,
        training.Checkpointer,
        training.Checkpointer,
        training.Checkpointer,
    ]:
        """
        Sets up checkpointers for policy, reference policy, value, and reward models.
        Only the policy checkpoint handles recipe state for resuming from checkpoints.
        """

        if not self._resume_from_checkpoint:
            assert policy_cfg.checkpoint_dir == ref_policy_cfg.checkpoint_dir, (
                "Policy and reference policy should be loaded from the same checkpoint directories"
                f"at the start of training. Found: {policy_cfg.checkpoint_dir} and"
                f"{ref_policy_cfg.checkpoint_dir}"
            )
            assert policy_cfg.checkpoint_files == ref_policy_cfg.checkpoint_files, (
                "Policy and reference policy should be loaded from the same checkpoint files"
                f"at the start of training. Found: {policy_cfg.checkpoint_files} and"
                f"{ref_policy_cfg.checkpoint_files}"
            )

        policy_checkpointer = config.instantiate(
            policy_cfg,
            resume_from_checkpoint=self._resume_from_checkpoint,
        )

        ref_policy_checkpointer = config.instantiate(
            ref_policy_cfg,
            resume_from_checkpoint=False,
        )

        value_checkpointer = config.instantiate(
            value_cfg,
            resume_from_checkpoint=False,
        )

        reward_checkpointer = config.instantiate(
            reward_cfg,
            resume_from_checkpoint=False,
        )

        return (
            policy_checkpointer,
            ref_policy_checkpointer,
            value_checkpointer,
            reward_checkpointer,
        )

    def _setup_models(
        self,
        cfg_model: DictConfig,
        cfg_reward_value_model: DictConfig,
        enable_activation_checkpointing: bool,
        compile_model: bool,
        policy_state_dict: Dict[str, Any],
        ref_policy_state_dict: Dict[str, Any],
        value_model_state_dict: Dict[str, Any],
        reward_model_state_dict: Dict[str, Any],
    ) -> Tuple[nn.Module, nn.Module, nn.Module]:
        """
        Sets up the policy model, reference policy model, reward model, and value model.
        """

        with training.set_default_dtype(self._dtype), self._device:
            policy_model = config.instantiate(cfg_model)
            ref_policy_model = config.instantiate(cfg_model)
            reward_model = config.instantiate(cfg_reward_value_model)
            value_model = config.instantiate(cfg_reward_value_model)

        if enable_activation_checkpointing:
            training.set_activation_checkpointing(
                policy_model, auto_wrap_policy={modules.TransformerSelfAttentionLayer}
            )
            training.set_activation_checkpointing(
                value_model, auto_wrap_policy={modules.TransformerSelfAttentionLayer}
            )

        policy_model.load_state_dict(policy_state_dict)
        ref_policy_model.load_state_dict(ref_policy_state_dict)

        # since we should be loading a classifier checkpoint into
        # a classifier model, this function should just ensure
        # output.weight appears in the state_dict and the model's parameters,
        # and removes output.bias from the state dict if found
        training.update_state_dict_for_classifier(
            reward_model_state_dict, reward_model.named_parameters()
        )
        reward_model.load_state_dict(reward_model_state_dict)

        # same as above
        training.update_state_dict_for_classifier(
            value_model_state_dict, value_model.named_parameters()
        )
        value_model.load_state_dict(value_model_state_dict)

        # Validate models were loaded in with the expected dtype.
        training.validate_expected_param_dtype(
            value_model.named_parameters(), dtype=self._dtype
        )
        training.validate_expected_param_dtype(
            reward_model.named_parameters(), dtype=self._dtype
        )
        training.validate_expected_param_dtype(
            value_model.named_parameters(), dtype=self._dtype
        )
        training.validate_expected_param_dtype(
            ref_policy_model.named_parameters(), dtype=self._dtype
        )

        log.info(f"Models are initialized with precision {self._dtype}.")

        # disabling dropout if found - non-determinism leads to issues in e.g. comparing logprobs
        # between ref policy and current policy
        for module in policy_model.modules():
            if isinstance(module, torch.nn.Dropout):
                warn(
                    f"Dropout found in {module}. This is likely to cause issues during training. Disabling."
                )
                module.p = 0
        for module in value_model.modules():
            if isinstance(module, torch.nn.Dropout):
                warn(
                    f"Dropout found in {module}. This is likely to cause issues during training. Disabling."
                )
                module.p = 0

        # disabling grad and dropout in reward and reference policy models
        reward_model.eval()
        ref_policy_model.eval()

        for p in reward_model.parameters():
            p.requires_grad = False

        for p in ref_policy_model.parameters():
            p.requires_grad = False

        # Compile model, if enabled.
        if compile_model:
            backend = os.environ.get("TORCH_COMPILE_BACKEND", "inductor")
            log.info("Compiling models with torch.compile...")

            policy_model.compile(backend=backend)
            reward_model.compile(backend=backend)
            ref_policy_model.compile(backend=backend)
            value_model.compile(backend=backend)

        if self._device.type == "cuda":
            memory_stats = training.get_memory_stats(device=self._device)
            training.log_memory_stats(memory_stats)

        return policy_model, value_model, reward_model, ref_policy_model

    def _setup_optimizer(
        self,
        cfg_optimizer: DictConfig,
        optimizer_in_bwd: bool = False,
        opt_state_dict: Optional[Dict[str, Any]] = None,
    ) -> Optimizer:

        if optimizer_in_bwd:
            # Maintain a dict of optims for every parameter.
            optim_dict = {
                p: config.instantiate(cfg_optimizer, [p])
                for p in chain(
                    self._policy_model.parameters(), self._value_model.parameters()
                )
            }
            # Register optimizer step hooks on the models to run optimizer in backward.
            training.register_optim_in_bwd_hooks(
                model=self._policy_model, optim_dict=optim_dict
            )
            training.register_optim_in_bwd_hooks(
                model=self._value_model, optim_dict=optim_dict
            )
            # Create a wrapper for checkpoint save/load of optimizer states when running in backward.
            self._optim_ckpt_wrapper = training.create_optim_in_bwd_wrapper(
                model=self._policy_model, optim_dict=optim_dict
            )
            self._optim_ckpt_wrapper = training.create_optim_in_bwd_wrapper(
                model=self._value_model, optim_dict=optim_dict
            )
            # Load optimizer states. If optimizer states are being restored in an optimizer in backward
            # run, these need to have been saved with the same setting. Cannot restore from runs that did not
            # use optimizer in backward.
            if opt_state_dict is not None:
                try:
                    self._optim_ckpt_wrapper.load_state_dict(opt_state_dict)
                except BaseException as e:
                    raise RuntimeError(
                        "Failed loading in-backward optimizer checkpoints."
                        "Please make sure run being restored from was using in-backward optimizer."
                    ) from e
            log.info("In-backward optimizers are set up.")
            return None
        else:
            optimizer = config.instantiate(
                cfg_optimizer,
                chain(self._policy_model.parameters(), self._value_model.parameters()),
            )
            if opt_state_dict:
                optimizer.load_state_dict(opt_state_dict)

            log.info("Optimizer is initialized.")
            return optimizer

    def _setup_data(
        self, cfg_dataset: DictConfig, shuffle: bool, batch_size: int
    ) -> Tuple[DistributedSampler, DataLoader]:
        """
        All data related setup happens here.
        """
        if isinstance(cfg_dataset, ListConfig):
            datasets = [
                config.instantiate(single_cfg_dataset, tokenizer=self._tokenizer)
                for single_cfg_dataset in cfg_dataset
            ]
            ds = ConcatDataset(datasets=datasets)
        else:
            ds = config.instantiate(cfg_dataset, tokenizer=self._tokenizer)

        sampler = DistributedSampler(
            ds,
            num_replicas=1,
            rank=0,
            shuffle=shuffle,
            seed=0,
        )
        dataloader = DataLoader(
            dataset=ds,
            sampler=sampler,
            batch_size=batch_size,
            # dropping last avoids shape issues with compile + flex attention
            drop_last=True,
            collate_fn=partial(
                padded_collate,
                pad_direction="left",
                keys_to_pad=["tokens", "labels"],
                padding_idx=self._tokenizer.pad_id,
            ),
        )

        return sampler, dataloader

    def save_checkpoint(
        self, epoch: int, is_intermediate_checkpoint: bool = False
    ) -> None:
        """
        Save state dict to file. The recipe save_checkpoint method is responsible for
        correctly creating the checkpoint dict and passing to the checkpointer.
        """
        policy_ckpt_dict = {training.MODEL_KEY: self._policy_model.state_dict()}
        value_ckpt_dict = {training.MODEL_KEY: self._value_model.state_dict()}

        # if training is in-progress, checkpoint the optimizer state and rng state as well
        if is_intermediate_checkpoint:
            policy_ckpt_dict.update(
                {
                    training.SEED_KEY: self.seed,
                    training.EPOCHS_KEY: self._epochs_run,
                    training.TOTAL_EPOCHS_KEY: self._total_epochs,
                    training.MAX_STEPS_KEY: self._total_steps,
                    training.STEPS_KEY: self._steps_run,
                    training.RNG_KEY: self._rng.get_state(),
                }
            )
            if not self._optimizer_in_bwd:
                policy_ckpt_dict[training.OPT_KEY] = self._optimizer.state_dict()
            else:
                policy_ckpt_dict[
                    training.OPT_KEY
                ] = self._optim_ckpt_wrapper.state_dict()

        self._policy_checkpointer.save_checkpoint(
            policy_ckpt_dict,
            epoch=epoch,
            intermediate_checkpoint=is_intermediate_checkpoint,
        )

        self._value_checkpointer.save_checkpoint(
            value_ckpt_dict,
            epoch=epoch,
            intermediate_checkpoint=False,
        )

    def _update_recipe_state(self, ckpt_dict: Dict[str, Any]) -> None:
        """
        Updates the recipe state from checkpoint.
        """
        # If seed or total_steps, or total_epochs don't match,
        # warn the user and overwrite.
        try:
            if (
                self.seed != ckpt_dict[training.SEED_KEY]
                or self._total_steps != ckpt_dict[training.MAX_STEPS_KEY]
                or self._total_epochs != ckpt_dict[training.TOTAL_EPOCHS_KEY]
            ):
                warn(
                    message="""Configured value for seed, total_steps, or total_epochs
                    does not match the value stored in checkpoint."""
                )
            self.seed = training.set_seed(seed=ckpt_dict[training.SEED_KEY])
            self._rng.set_state(ckpt_dict[training.RNG_KEY])
            self._steps_run = ckpt_dict[training.STEPS_KEY]
            self._total_steps = ckpt_dict[training.MAX_STEPS_KEY]
            self._total_epochs = ckpt_dict[training.TOTAL_EPOCHS_KEY]
            self._epochs_run = ckpt_dict[training.EPOCHS_KEY]

        except KeyError as e:
            raise KeyError from e(
                "Checkpoint does not contain the required keys needed for updating recipe state."
                "Are you sure you passed in the right recipe checkpoint?"
            )

    def generate_trajectory(self, input_ids: torch.Tensor) -> Trajectory:
        """
        Generates a trajectory given the current policy and value models, the reference policy model, the reward model,
        and batch of inputs. This is done over the following steps:

        1: Generate responses, and logits corresponding to the responses using the current policy,
            generating (query, response) pairs.
        2. Estimate logprobs of the generated responses using the current policy.
        3. Estimate values from the generated responses using the current value function.
        4. Replace any tokens in the response after the first stop token (usually EOS token) with padding,
            producting truncated responses.
        5. Run the reward model on the (query, truncated-response) pairs.
        6. Mask out all the invalid values in the trajectory due to padding tokens.

        Args:
            input_ids (torch.Tensor): tensor of input token IDs with shape [b, seq_length]

        Returns:
            Trajectory: An instance of :class:`~torchtune.rlhf.Trajectory` comprising
                the current trajectory.
        """
        batch_size, context_length = input_ids.shape

        # step 1: generate responses, and logits corresponding to the responses using the current policy
        query_responses, logits = generation.generate(
            model=self._policy_model,
            prompt=input_ids,
            max_generated_tokens=self._max_generated_tokens,
            temperature=self._temperature,
            top_k=self._top_k,
            pad_id=self._tokenizer.pad_id,
            rng=self._rng,
        )

        responses = query_responses[:, context_length:].clone()
        query_response_padding_masks = query_responses != self._tokenizer.pad_id

        # step 1.1 create attention masks and position IDs for any padding tokens in inputs, used for future forward passes
        masks = generation.get_causal_mask_from_padding_mask(
            query_response_padding_masks
        )
        position_ids = generation.get_position_ids_from_padding_mask(
            query_response_padding_masks
        )

        del query_response_padding_masks

        # step 2. estimate logprobs of the responses using the current policy
        logits = logits[:, context_length - 1 :]
        logprobs = rlhf.logits_to_logprobs(logits, responses, self._temperature)

        del logits

        # step 2.1 estimate logprobs of the responses using the reference policy
        ref_logits = self._ref_policy_model(
            query_responses, input_pos=position_ids, mask=masks
        )
        ref_logits = rlhf.truncate_sequence_for_logprobs(ref_logits, context_length)
        ref_logprobs = rlhf.logits_to_logprobs(ref_logits, responses, self._temperature)

        del ref_logits

        # step 3. estimate values from the responses using the value function
        values = self._value_model(query_responses, input_pos=position_ids, mask=masks)
        values = rlhf.truncate_sequence_for_logprobs(values, context_length).squeeze(-1)

        # step 4. replace any tokens in the responses after the first stop token (usually EOS token) with padding
        # resulting in truncated responses
        response_padding_masks, responses = rlhf.truncate_sequence_at_first_stop_token(
            responses, self._stop_token_ids, self._tokenizer.pad_id
        )

        # step 5. run the reward model on the (query, truncated-response) pairs
        scores = self._reward_model(
            torch.cat([input_ids, responses], dim=1),
            input_pos=position_ids,
            mask=masks,
        )

        del responses

        # step 5.1 the scores from the reward model are the logits for the last non-padding token in
        # each (query, truncated-response) pair
        seq_lens = training.get_unmasked_sequence_lengths(response_padding_masks)
        scores = scores[torch.arange(batch_size), seq_lens + context_length].squeeze(-1)

        # step 5.2 if configured, apply any penalties for sequences without EOS tokens
        # or shorter than a certain length
        if self._penalise_no_eos or self._min_response_length:
            reward_penalty_mask = rlhf.get_reward_penalty_mask(
                response_padding_masks,
                seq_lens,
                self._penalise_no_eos,
                self._min_response_length,
            )
            scores[reward_penalty_mask] = self._reward_penalty

        # step 6. mask out all the invalid values in the trajectory due to padding tokens
        logprobs[response_padding_masks] = 1.0
        ref_logprobs[response_padding_masks] = 1.0

        # step 6.1 values are masked out *after* the last valid token in the response
        value_seq_idxs = torch.where(
            (seq_lens > 0) & (seq_lens < self._max_generated_tokens - 1),
            seq_lens + 1,
            seq_lens,
        )
        value_padding_masks = response_padding_masks.clone()
        value_padding_masks[
            torch.arange(batch_size, device=value_padding_masks.device),
            value_seq_idxs,
        ] = False

        values[value_padding_masks] = 0.0

        return Trajectory(
            query_responses=query_responses,
            logprobs=logprobs,
            ref_logprobs=ref_logprobs,
            values=values,
            masks=masks,
            position_ids=position_ids,
            response_padding_masks=response_padding_masks,
            value_padding_masks=value_padding_masks,
            value_seq_idxs=value_seq_idxs,
            scores=scores,
            seq_lens=seq_lens,
        )

    def generate_trajectory_batched(self, input_ids: torch.Tensor) -> Trajectory:
        """
        Generates a ``self.batch_size`` batch of trajectories using `self._forward_batch_size` batch sizes.
        See ``generate_trajectory`` for more details.

        Args:
            input_ids (torch.Tensor): tensor of input token IDs with shape [b, seq_length]

        Returns:
            Trajectory: An instance of :class:`~torchtune.rlhf.Trajectory`, comprising
                the current trajectory.
        """
        trajectories: List[Trajectory] = []
        with torch.no_grad():
            for batch_start in range(0, self.batch_size, self._forward_batch_size):
                batch_input_ids = input_ids[
                    batch_start : batch_start + self._forward_batch_size
                ]
                trajectories.append(self.generate_trajectory(batch_input_ids))
        return Trajectory(*map(torch.cat, zip(*trajectories)))

    def train(self) -> None:
        """
        The core training loop."""

        if self._model_compile:
            log.info(
                "NOTE: torch.compile is enabled and model is compiled in first forward."
                "Expect a relatively slow first iteration."
            )
        # zero out the gradients before starting training
        if not self._optimizer_in_bwd:
            self._optimizer.zero_grad()

        training_completed = False
        pbar = tqdm(total=self._total_steps, initial=self._steps_run)
        for curr_epoch in range(self._epochs_run, self._total_epochs):
            # Update the sampler to ensure data is correctly shuffled across epochs
            # in case shuffle is True
            self._sampler.set_epoch(curr_epoch)

            for _, batch in enumerate(self._dataloader):
                batch = batch["tokens"].to(self._device)
                _, context_length = batch.shape

                # step 1. generate the trajectory using:
                # - the current policy (pi_theta)
                # - the current value function (V_phi)
                # - the reference frozen policy model (pi_theta_0)
                trajectory = self.generate_trajectory_batched(batch)

                # step 2. get the rewards for the current trajectory. these are based on:
                #   - the divergence between the current policy and the reference policy
                #   - the scores from the reward model
                rewards, kl, kl_rewards = rlhf.get_rewards_ppo(
                    trajectory.scores,
                    trajectory.logprobs,
                    trajectory.ref_logprobs,
                    self._kl_coeff,
                    trajectory.value_seq_idxs,
                )

                # step 3. estimate the advantages using Generalized Advantage Estimation (GAE)
                advantages, returns = rlhf.estimate_advantages(
                    trajectory.values,
                    rewards,
                    self._gamma,
                    self._lmbda,
                    masks=~trajectory.response_padding_masks,
                )

                # step 4. optimise using the PPO objective over multiple epochs
                ppo_stats: List[PPOStats] = []
                for _ in range(self._ppo_epochs):
                    batch_idxs = torch.randperm(self.batch_size, device=self._device)
                    for i in range(0, self.batch_size, self._ppo_batch_size):
                        mini_batch_idxs = batch_idxs[i : i + self._ppo_batch_size]

                        batch_ppo_stats: List[PPOStats] = []
                        for j in range(
                            0, self._ppo_batch_size, self._ppo_backward_batch_size
                        ):
                            backward_batch_idxs = mini_batch_idxs[
                                j : j + self._ppo_backward_batch_size
                            ]

                            batch_trajectory = Trajectory(
                                *map(
                                    partial(
                                        torch.index_select,
                                        dim=0,
                                        index=backward_batch_idxs,
                                    ),
                                    trajectory,
                                )
                            )
                            batch_ppo_stats.append(
                                self._ppo_step(
                                    batch_trajectory,
                                    advantages[backward_batch_idxs],
                                    returns[backward_batch_idxs],
                                    context_length,
                                )
                            )
                            del batch_trajectory

                        ppo_stats.append(PPOStats(*map(sum, zip(*batch_ppo_stats))))

                        if not self._optimizer_in_bwd:
                            self._optimizer.step()
                            self._optimizer.zero_grad(set_to_none=True)

                        self.global_step += 1

                # step 5. profit
                self._steps_run += 1
                if self._steps_run % self._log_every_n_steps == 0:
                    self.log_metrics(
                        trajectory,
                        PPOStats(*map(torch.stack, zip(*ppo_stats))),
                        kl,
                        kl_rewards,
                    )
                self.cleanup_after_step(
                    trajectory, ppo_stats, advantages, returns, kl, kl_rewards
                )
                pbar.update(1)
                if self._steps_run == self._total_steps:
                    training_completed = True
                    break

            # save checkpoint at current epoch
            self._epochs_run += 1

            self.save_checkpoint(
                curr_epoch, is_intermediate_checkpoint=not training_completed
            )
            if training_completed:
                return

    def _ppo_step(
        self,
        trajectory: Trajectory,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        context_length: int,
    ) -> PPOStats:
        """
        Perform a single PPO optimisation step over a batch of trajectories and corresponding advantages and returns.

        Args:
            trajectory (Trajectory): a batch of trajectories
            advantages (torch.Tensor): advantages corresponding to the trajectories
            returns (torch.Tensor): returns corresponding the trajectories
            context_length (int): input ids sequence length

        Returns:
            PPOStats: An instance of :class:`~torchtune.rlhf.PPOStats`, a NamedTuple containing:
               - loss (torch.Tensor): The total PPO loss.
               - policy_loss (torch.Tensor): The policy function loss.
               - value_loss (torch.Tensor): The value function loss.
               - ratios (torch.Tensor): The ratio between the current and old policy probabilities.
               - clipfrac (torch.Tensor): The fraction of ratios that were clipped.
               - approx_policy_kls: Average estimated KL divergence between the policy before and after the optimisation step.

        """
        # estimate logprobs from the policy at the current optimisation step
        pi_logits = self._policy_model(
            trajectory.query_responses,
            input_pos=trajectory.position_ids,
            mask=trajectory.masks,
        )
        pi_logits = rlhf.truncate_sequence_for_logprobs(pi_logits, context_length)
        pi_logprobs = rlhf.logits_to_logprobs(
            pi_logits, trajectory.query_responses[:, context_length:], self._temperature
        )
        pi_logprobs[trajectory.response_padding_masks] = 1.0

        del pi_logits

        # estimate the values from the value function at the current optimisation step
        phi_values = self._value_model(
            trajectory.query_responses,
            input_pos=trajectory.position_ids,
            mask=trajectory.masks,
        )

        phi_values = rlhf.truncate_sequence_for_logprobs(
            phi_values, context_length
        ).squeeze(-1)
        phi_values[trajectory.value_padding_masks] = 0.0

        # calculate ppo loss
        loss, policy_loss, value_loss, ratios, clipfrac = self._loss_fn(
            trajectory.logprobs,
            pi_logprobs,
            advantages,
            trajectory.values,
            phi_values,
            returns,
            padding_masks=~trajectory.response_padding_masks,
            value_padding_masks=~trajectory.value_padding_masks,
        )

        loss /= self._gradient_accumulation_steps
        loss.backward()

        with torch.no_grad():
            approx_policy_kls = (
                0.5 * (pi_logprobs - trajectory.logprobs).pow(2)
            ).mean()

        return PPOStats(
            loss,
            policy_loss / self._gradient_accumulation_steps,
            value_loss / self._gradient_accumulation_steps,
            ratios / self._gradient_accumulation_steps,
            clipfrac / self._gradient_accumulation_steps,
            approx_policy_kls / self._gradient_accumulation_steps,
        )

    def log_metrics(
        self,
        trajectory: Trajectory,
        ppo_stats: PPOStats,
        kl: torch.Tensor,
        kl_rewards: torch.Tensor,
    ) -> None:
        """
        Log metrics and statistics for the current step to the metric logger.
        """
        log_dict = {
            "scores": trajectory.scores.mean(),
            "num_stop_tokens": trajectory.response_padding_masks.any(-1).sum(),
            "rlhf_reward": trajectory.scores.mean() + kl_rewards.sum(1).mean(),
            "kl": kl.sum(1).mean(),
            "kl_reward": kl_rewards.sum(1).mean(),
            "loss": ppo_stats.loss.mean(),
            "policy_loss": ppo_stats.policy_loss.mean(),
            "value_loss": ppo_stats.value_loss.mean(),
            "clipfrac": ppo_stats.clipfrac.mean(),
            "ratios": ppo_stats.ratios.mean(),
            "approx_policy_kl": ppo_stats.approx_policy_kls.mean(),
            "response_lengths": trajectory.seq_lens.float().mean(),
        }
        if self._device.type == "cuda" and self._log_peak_memory_stats:
            log_dict.update(training.get_memory_stats(device=self._device))

        self._metric_logger.log_dict(log_dict, step=self.global_step)

    def cleanup_after_step(
        self,
        trajectory: Trajectory,
        ppo_stats: PPOStats,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        kl: torch.Tensor,
        kl_rewards: torch.Tensor,
    ) -> None:
        """
        Cleanup tensors after each PPO step to free up memory.
        """
        # there shouldn't be any floating references to the individual tensors at the this point, so gc can do its thing
        for v in trajectory:
            del v
        del trajectory
        for v in ppo_stats:
            del v
        del ppo_stats
        del advantages
        del returns
        del kl
        del kl_rewards

    def cleanup(self, **kwargs) -> None:
        self._metric_logger.close()


@config.parse
def recipe_main(cfg: DictConfig) -> None:
    """
    Entry point for the recipe.

    Configurable parameters are read in the following order:
        - Parameters specified in config (see available configs through ``tune ls``)
        - Overwritten by arguments from the command-line
    """
    config.log_config(recipe_name="PPOFullFinetuneRecipeSingleDevice", cfg=cfg)
    recipe = PPOFullFinetuneRecipeSingleDevice(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())
