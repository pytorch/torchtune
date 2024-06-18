# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys

from functools import partial
from typing import Any, Dict, Optional, Tuple
from warnings import warn

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig
from pkg_resources import packaging
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler
from torchtune import config, utils

from torchtune.datasets import ConcatDataset
from torchtune.models.mistral.modules import TransformerLMWithValueHead

from torchtune.modules.peft.peft_utils import (
    disable_adapter,
    get_adapter_params,
    get_merged_lora_ckpt,
    set_trainable_params,
    validate_state_dict_for_lora,
)
from torchtune.recipe_interfaces import FTRecipeInterface
from torchtune.utils import ppo_utils

from tqdm import tqdm

log = utils.get_logger("DEBUG")


class LoRAPPORecipeSingleDevice(FTRecipeInterface):
    def __init__(self, cfg: DictConfig) -> None:

        self._device = utils.get_device(device=cfg.device)
        # Reduced precision logic
        self._dtype = utils.get_dtype(cfg.dtype, device=self._device)
        # fp16 precision is explicitly disabled as it is not supported in this
        # recipe (for example, no gradient scaling).
        if self._dtype == torch.float16:
            raise ValueError(
                "fp16 precision is not supported in this recipe. Please use fp32 or bf16."
            )
        # For CUDA devices, check if the HW supports bf16 if bf16 is specified.
        if self._dtype == torch.bfloat16 and self._device != torch.device("cpu"):
            if torch.cuda.is_available():
                if not torch.cuda.is_bf16_supported():
                    raise RuntimeError(
                        "Full bf16 training is not supported on this hardware."
                    )
            elif torch.backends.mps.is_available():
                if packaging.version.parse(torch.__version__).release < (2, 3):
                    raise RuntimeError(
                        "Full bf16 training is not supported on this hardware."
                    )
        # logging attributes
        self._output_dir = cfg.output_dir
        self._log_every_n_steps = cfg.get("log_every_n_steps", 1)
        self._log_peak_memory_stats = cfg.get("log_peak_memory_stats", False)

        # These are public properties which are updated by the checkpoint loader
        # when ``resume_from_checkpoint`` is `True` or validated in tests
        self.seed = utils.set_seed(seed=cfg.seed)
        self.epochs_run = 0
        self.total_epochs = 0
        self.global_step = 0

        self._resume_from_checkpoint = cfg.resume_from_checkpoint
        self._gradient_accumulation_steps = cfg.gradient_accumulation_steps

    def save_checkpoint(self, epoch: int) -> None:
        """
        Checkpoint the state of the recipe. The constructed checkpoint state dict
        contains the following information:
        - Merged weights with key MODEL_KEY
        - Adapter weights with key ADAPTER_KEY
        - Value head weights with key VALUE_HEAD_KEY
        - Relevant recipe state if training is not complete

        Checkpointer will save the merged weights, adapter weights and recipe state in
        different checkpoint files. To correctly resume from training, the adapter weights
        and recipe state must be provided along with the base model weights.
        """
        ckpt_dict = {}
        # if training is in-progress, checkpoint the optimizer state as well
        if epoch + 1 < self.total_epochs:
            ckpt_dict.update(
                {
                    utils.OPT_KEY: self._optimizer.state_dict(),
                    utils.SEED_KEY: self.seed,
                    utils.EPOCHS_KEY: self.epochs_run,
                    utils.TOTAL_EPOCHS_KEY: self.total_epochs,
                }
            )

        # Move to CPU to avoid a copy on GPU
        state_dict = {k: v.cpu() for k, v in self._model.state_dict().items()}

        # split off the value head weights
        value_head_state_dict = {
            "value_head.weight": state_dict.pop("value_head.weight"),
            "value_head.bias": state_dict.pop("value_head.bias"),
        }
        ckpt_dict.update({utils.VALUE_HEAD_KEY: value_head_state_dict})

        # save base model as usual
        # Construct the full state dict with LoRA weights merged into base LLM weights
        merged_state_dict = get_merged_lora_ckpt(
            state_dict,
            rank=self._lora_rank,
            alpha=self._lora_alpha,
        )
        ckpt_dict.update({utils.MODEL_KEY: merged_state_dict})

        # Construct the adapter weights
        adapter_key_filter = lambda x: x in self.adapter_params
        adapter_state_dict = {
            k: v for k, v in self._model.state_dict().items() if adapter_key_filter(k)
        }
        ckpt_dict.update({utils.ADAPTER_KEY: adapter_state_dict})

        self._checkpointer.save_checkpoint(
            ckpt_dict,
            epoch=epoch,
            intermediate_checkpoint=(epoch + 1 < self.total_epochs),
        )

    def _update_recipe_state(self, ckpt_dict: Dict[str, Any]) -> None:
        """
        Updates the recipe state from checkpoint.
        """
        # If seed or total_epoch,
        # warn the user and overwrite
        if (
            self.seed != ckpt_dict[utils.SEED_KEY]
            or self.total_epochs != ckpt_dict[utils.TOTAL_EPOCHS_KEY]
        ):
            warn(
                message="""Configured value for seed or epochs
                does not match the value stored in checkpoint."""
            )
        self.seed = utils.set_seed(seed=ckpt_dict[utils.SEED_KEY])
        self.epochs_run = ckpt_dict[utils.EPOCHS_KEY]
        self.total_epochs = ckpt_dict[utils.TOTAL_EPOCHS_KEY]

    def setup(self, cfg: DictConfig) -> None:
        """
        Setup the recipe state. This includes recipe state (if resume_from_checkpoint is True),
        model, tokenizer, loss, optimizer, learning rate scheduler, sampler, and dataloader.
        """
        self._metric_logger = config.instantiate(cfg.metric_logger)

        # log config with parameter override
        self._metric_logger.log_config(cfg)

        # setup checkpointers
        self._checkpointer = config.instantiate(
            cfg.checkpointer,
            resume_from_checkpoint=self._resume_from_checkpoint,
        )
        self._reward_checkpointer = config.instantiate(
            cfg.reward_checkpointer,
            # reward model is never being trained
            # base model checkpointer handles recipe state
            resume_from_checkpoint=False,
        )
        # load base model checkpoint
        model_checkpoint_dict = self._checkpointer.load_checkpoint()

        # load reward model checkpoint
        reward_model_checkpoint_dict = self._reward_checkpointer.load_checkpoint()

        # update recipe state
        if self._resume_from_checkpoint:
            # _update_recipe_state will throw an exception if the recipe state is not correctly loaded
            if utils.ADAPTER_KEY not in model_checkpoint_dict:
                raise ValueError(
                    "Adapter weights not found. Please ensure a valid adapter checkpoint is provided."
                )
            if utils.VALUE_HEAD_KEY not in model_checkpoint_dict:
                raise ValueError(
                    "Value head weights not found. Please ensure a valid value head checkpoint is provided."
                )
            self._update_recipe_state(model_checkpoint_dict)

        # ``_setup_model`` handles initialization and loading the state dict. This method
        # should be called before ``_setup_optimizer`` since transforming the optimizer
        # state dict requires the model
        self._model, self._reward_model = self._setup_model(
            cfg_model=cfg.model,
            cfg_reward_model=cfg.reward_model,
            enable_activation_checkpointing=cfg.enable_activation_checkpointing,
            model_state_dict=model_checkpoint_dict[utils.MODEL_KEY],
            reward_model_state_dict=reward_model_checkpoint_dict[utils.MODEL_KEY],
            model_lora_weights_state_dict=(
                model_checkpoint_dict[utils.ADAPTER_KEY]
                if self._resume_from_checkpoint
                else None
            ),
            value_head_state_dict=(
                model_checkpoint_dict[utils.VALUE_HEAD_KEY]
                if self._resume_from_checkpoint
                else None
            ),
            initialise_value_head_from_reward_model=cfg.initialise_value_head_from_reward_model,
        )

        # setup tokenizer
        self._tokenizer = config.instantiate(cfg.tokenizer)
        log.info("Tokenizer is initialized from file.")

        # setup opt
        # _setup_optimizer should take in ckpt_dict only if training is resumed from
        # checkpoint. Transforming the opt state dict is handled by this method
        self._optimizer = self._setup_optimizer(
            cfg_optimizer=cfg.optimizer,
            opt_state_dict=(
                model_checkpoint_dict[utils.OPT_KEY]
                if self._resume_from_checkpoint
                else None
            ),
        )

        # GAE hyperparameters
        self.gamma = cfg.loss.gamma
        self.lmbda = cfg.loss.lmbda
        self.whiten_rewards = cfg.whiten_rewards
        self._loss_fn = config.instantiate(cfg.loss)
        log.info("Loss is initialized.")

        # sampler and dataloader depends on the tokenizer and should be set
        # setup afterit is initialized
        self._sampler, self._dataloader = self._setup_data(
            cfg_dataset=cfg.dataset,
            shuffle=cfg.shuffle,
            batch_size=cfg.batch_size,
        )

        # setup training params:
        # num_steps - global number of optimisation steps (batches)
        # batch_size - number of samples in a batch
        # ppo_epochs - number of epochs to optimise the policy over a batch of episodes
        # ppo_batch_size - number of minibatches (sampled from a single batch) to optimise the policy over
        self.num_steps = cfg.num_steps
        self.batch_size = cfg.batch_size
        self.ppo_epochs = cfg.ppo_epochs
        self.ppo_batch_size = cfg.ppo_batch_size
        self._gradient_accumulation_steps = cfg.gradient_accumulation_steps
        if self.ppo_batch_size % self._gradient_accumulation_steps != 0:
            raise ValueError(
                f"ppo_batch_size ({self.ppo_batch_size})  must be "
                f"exactly divisible by gradient_accumulation_steps ({cfg._gradient_accumulation_steps})."
            )
        self.ppo_backward_batch_size = (
            cfg.ppo_batch_size // self._gradient_accumulation_steps
        )

        # trajectory generation args
        self.temperature = cfg.temperature + 1e-7  # avoid underflow for 0 temperature
        self.top_k = cfg.top_k
        self.forward_batch_size = cfg.forward_batch_size
        self.max_generated_tokens = cfg.max_generated_tokens

        # reward masking args
        self.truncate_after_tokens = cfg.truncate_after_tokens
        self.penalise_no_eos = cfg.penalise_no_eos
        self.reward_penalty = cfg.reward_penalty

        self.total_epochs = self.num_steps // self.batch_size
        self._gradient_accumulation_steps = cfg.gradient_accumulation_steps
        # one "step" is an update over a batch of trajectories
        self._steps_per_epoch = len(self._dataloader) // self.batch_size
        self.global_step = self.epochs_run * self._steps_per_epoch

        # setup adaptive KL controller
        self.kl_controller = config.instantiate(cfg.kl_controller)

        self._profiler_enabled = cfg.profiler.enabled
        self._profiler = config.instantiate(cfg.profiler)

    def _setup_model(
        self,
        cfg_model: DictConfig,
        cfg_reward_model: DictConfig,
        enable_activation_checkpointing: bool,
        model_state_dict: Dict[str, Any],
        reward_model_state_dict: Dict[str, Any],
        model_lora_weights_state_dict: Optional[Dict[str, Any]] = None,
        value_head_state_dict: Optional[Dict[str, Any]] = None,
        initialise_value_head_from_reward_model: bool = False,
    ) -> nn.Module:

        with utils.set_default_dtype(self._dtype), self._device:
            model = config.instantiate(cfg_model)
            reward_model = config.instantiate(cfg_reward_model)

        self._lora_rank = cfg_model.lora_rank
        self._lora_alpha = cfg_model.lora_alpha
        self.adapter_params = get_adapter_params(model)
        set_trainable_params(model, self.adapter_params)

        if enable_activation_checkpointing:
            utils.set_activation_checkpointing(
                model, auto_wrap_policy={TransformerLMWithValueHead}
            )

        # add placeholder value head key to base model state dict keys
        # to ensure wrapped model is validated correctly
        model_state_dict_keys = list(model_state_dict.keys())
        model_state_dict_keys += ["value_head.weight", "value_head.bias"]

        state_dict = model.state_dict()

        validate_state_dict_for_lora(
            lora_attn_modules=cfg_model.lora_attn_modules,
            apply_lora_to_mlp=cfg_model.apply_lora_to_mlp,
            apply_lora_to_output=cfg_model.apply_lora_to_output,
            # base model state dict keys
            full_model_state_dict_keys=state_dict.keys(),
            lora_state_dict_keys=(
                model_lora_weights_state_dict.keys()
                if model_lora_weights_state_dict is not None
                else None
            ),
        )

        for k in state_dict:
            if k in model_state_dict:
                state_dict[k] = model_state_dict[k]

        if value_head_state_dict is not None:
            state_dict["value_head.weight"] = value_head_state_dict[k]
        elif initialise_value_head_from_reward_model:
            state_dict["value_head.weight"] = reward_model_state_dict[
                "output.weight"
            ].clone()
            log.info("Value head is initialized using the reward head.")

        # load checkpoints
        model.load_state_dict(state_dict)
        reward_model.load_state_dict(reward_model_state_dict)

        # Validate model was loaded in with the expected dtype.
        utils.validate_expected_param_dtype(model.named_parameters(), dtype=self._dtype)
        log.info(f"Base model is initialized with precision {self._dtype}.")

        utils.validate_expected_param_dtype(
            reward_model.named_parameters(), dtype=self._dtype
        )
        log.info(f"Reward model is initialized with precision {self._dtype}.")

        return model, reward_model

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

        log.info("Learning rate scheduler is initialized.")
        return lr_scheduler

    def batched_generate(
        self, input_ids: torch.Tensor, model: nn.Module
    ) -> torch.Tensor:
        """
        Generates sequences using a language model.
        Args:
            input_ids (torch.Tensor): The input tensor of shape [b, seq_len].
            model (nn.Module): The model to generate sequences from.
        Returns:
            torch.Tensor: The concatenated queries and generated sequences of shape
                [b, seq_len + max_generated_tokens].
        """
        outputs = []
        forward_batch_size = min(len(input_ids), self.forward_batch_size)
        for batch_start in range(0, input_ids.shape[0], forward_batch_size):
            batch_end = min(batch_start + forward_batch_size, input_ids.shape[0])
            batch_input_ids = input_ids[batch_start:batch_end]

            outputs.extend(
                ppo_utils.generate(
                    model=model,
                    prompt=batch_input_ids,
                    max_generated_tokens=self.max_generated_tokens,
                    temperature=self.temperature,
                    top_k=self.top_k,
                    # disabling truncation since we require some additional logic to handle truncation
                    stop_tokens=None,
                    pad_id=self._tokenizer.pad_id,
                    custom_generate_next_token=ppo_utils.generate_next_token_with_value_head_model,
                    dtype=self._dtype,
                )
            )
        return outputs

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
            collate_fn=partial(
                ppo_utils.left_padded_collate,
                max_seq_len=cfg_dataset.max_seq_len,
                padding_idx=self._tokenizer.pad_id,
            ),
        )

        log.info("Dataset and Sampler are initialized.")

        return sampler, dataloader

    def train(self) -> None:
        """
        The core training loop."""
        for curr_epoch in range(self.epochs_run, self.total_epochs):
            self._sampler.set_epoch(curr_epoch)

            with self._profiler:

                pbar = tqdm(total=self.num_steps)
                batch = next(iter(self._dataloader))
                if self._profiler_enabled:
                    self._profiler.step()

                # generating the current trajectory in inference mode
                with torch.no_grad():
                    # this should support any dataset format since the collator just returns input ids
                    input_ids = batch
                    input_ids = input_ids.to(self._device)
                    context_length = input_ids.shape[1]

                    # input ids are left-padded by default - we need to shift the position ids
                    # and generate causal masks if any sequence in the batch has been padded
                    # both during generation and logit calculation
                    # see https://github.com/huggingface/trl/blob/
                    #       f5168fdbaf9cbf6a3f1bdc64dc44b9db3a9ae333/trl/trainer/utils.py#L1099
                    query_responses = self.batched_generate(input_ids, self._model)
                    query_responses = torch.stack(query_responses).to(self._device)

                    # create position IDs and causal masks for the current trajectory
                    padding_masks = query_responses == self._tokenizer.pad_id
                    # we only need custom causal masks for sequences with left-padding
                    if padding_masks.any():
                        masks = ppo_utils.get_causal_mask(
                            query_responses,
                            padding_mask=padding_masks,
                            dtype=self._dtype,
                        )
                        position_ids = (~padding_masks).cumsum(-1) - (
                            ~padding_masks
                        ).long()
                        position_ids = position_ids.to(
                            device=self._device, dtype=torch.int
                        )
                    else:
                        # defer SDPA to handle causal masks
                        masks, position_ids = None, None

                    # we only need padding masks for responses from here on
                    padding_masks = padding_masks[:, context_length:]

                    # policy and value estimates for the current trajectory
                    # TODO (SalmanMohammadi) implement minibatch model forward pass
                    logits, values = self._model(
                        query_responses, input_pos=position_ids, mask=masks
                    )

                    logits = logits[
                        :, context_length - 1 : -1
                    ]  # [b, max_generated_tokens, vocab_size]
                    logits /= self.temperature
                    # we only need the logprobs of the generated tokens since these are just used for KL rewards
                    logprobs = torch.gather(
                        F.log_softmax(logits, dim=-1),
                        2,
                        query_responses[:, context_length:].unsqueeze(
                            -1
                        ),  # [b, max_generated_tokens, 1]
                    ).squeeze(-1)

                    del logits

                    # generating with lora adapters disabled gives us the pre-finetuning ref model
                    with disable_adapter(self._model):
                        ref_logits, _ = self._model(
                            query_responses, input_pos=position_ids, mask=masks
                        )

                    ref_logits = ref_logits[:, context_length - 1 : -1]
                    ref_logits /= self.temperature
                    # shape [b, max_generated_tokens]
                    ref_logprobs = torch.gather(
                        F.log_softmax(ref_logits, dim=-1),
                        2,
                        query_responses[:, context_length:].unsqueeze(-1),
                    ).squeeze(-1)

                    del ref_logits

                    # dear reviewer - would you rather see this as a separate function and tested?
                    # i'm balancing between comprehensability of algo. design decisions and testability here

                    # masking scores s.t.:
                    # truncate sequences at the first occurence of eos_id and pads
                    if self._tokenizer.eos_id is not None:
                        eos_mask = (
                            query_responses[:, context_length:]
                            == self._tokenizer.eos_id
                        )
                        query_responses[:, context_length:].masked_fill_(
                            torch.logical_xor(eos_mask.cumsum(-1), eos_mask),
                            self._tokenizer.pad_id,
                        )
                    else:
                        eos_mask = torch.ones_like(
                            query_responses[:, context_length:]
                        ).bool()

                    # run reward model on truncated query-response sequences: shape [b, context_length + max_generated_tokens]
                    # TODO (SalmanMohammadi): Add support for _reward_model and _model using different tokenizers
                    scores = self._reward_model(
                        query_responses, input_pos=position_ids, mask=masks
                    )

                    idxs = utils.get_last_non_masked_token(padding_masks)
                    # shape [b, ]
                    scores = scores[
                        torch.arange(0, scores.shape[0]), idxs + context_length
                    ].squeeze(-1)

                    reward_penality_mask = torch.zeros_like(scores).to(bool)

                    # to reviewer - see commment above
                    # - sequences without a EOS ID recieve a score of -1
                    if self.penalise_no_eos:
                        reward_penality_mask = ~eos_mask.any(-1)
                    # - sequences with length < truncate_after_tokens recieve a score of -1
                    if self.truncate_after_tokens is not None:
                        reward_penality_mask |= ~(
                            eos_mask.cumsum(-1).sum(-1) - 1
                            <= self.truncate_after_tokens
                        )

                    # see https://arxiv.org/pdf/1909.08593 section 3.1.2
                    scores.masked_fill_(reward_penality_mask, self.reward_penalty)

                    # now mask out logprobs and values w.r.t. padding masks
                    # https://github.com/huggingface/trl/blob/
                    #   f5168fdbaf9cbf6a3f1bdc64dc44b9db3a9ae333/trl/trainer/ppov2_trainer.py#L359
                    logprobs = logprobs.masked_fill(padding_masks, 1.0)
                    ref_logprobs = ref_logprobs.masked_fill(padding_masks, 1.0)

                    # values are masked up to and including the first padding token
                    # see https://github.com/huggingface/trl/blob/f5168fdbaf9cbf6a3f1bdc64dc44b9db3a9ae333/
                    #   trl/trainer/ppov2_trainer.py#L354
                    # and the link to the excalidaw therein for a visual explanation
                    value_padding_masks = padding_masks.clone()
                    valid_response_lengths = (
                        self.max_generated_tokens
                        - value_padding_masks.cumsum(dim=-1)[:, -1]
                    )
                    valid_response_lengths = valid_response_lengths.clip(
                        0, self.max_generated_tokens - 1
                    )
                    value_padding_masks[
                        torch.arange(input_ids.shape[0]), valid_response_lengths
                    ] = False
                    values = values[:, context_length - 1 : -1].squeeze(
                        -1
                    )  # [b, max_generated_tokens]

                    values = values.masked_fill(value_padding_masks, 0.0)
                    # [b, max_generated_tokens]
                    rewards, kl, kl_rewards = ppo_utils.get_rewards(
                        scores, logprobs, ref_logprobs, self.kl_controller.value
                    )

                    if self.whiten_rewards:
                        # shifting mean is disabled for rewards
                        # https://github.com/huggingface/trl/blob/
                        #   f5168fdbaf9cbf6a3f1bdc64dc44b9db3a9ae333/trl/trainer/ppov2_trainer.py#L373
                        rewards = ppo_utils.masked_whiten(
                            rewards, value_padding_masks, shift_mean=False
                        )
                        rewards = rewards.masked_fill(value_padding_masks, 0.0)

                    advantages, returns = ppo_utils.estimate_advantages(
                        values, rewards, self.gamma, self.lmbda, masks=~padding_masks
                    )
                    advantages = advantages.masked_fill(padding_masks, 0.0)

                    del eos_mask, scores, valid_response_lengths

                # trajectory generated! time to optimise
                approx_policy_kls = []
                losses = []
                policy_losses = []
                value_losses = []
                entropies = []
                for _ in range(self.ppo_epochs):
                    # TODO (SalmanMohammadi): Add support for early stopping
                    # shuffle batch indices every epoch
                    batch_idxs = torch.randperm(self.batch_size)
                    for i in range(0, self.batch_size, self.ppo_batch_size):
                        mini_batch_idxs = batch_idxs[i : i + self.ppo_batch_size]
                        for j in range(
                            0, self.ppo_batch_size, self.ppo_backward_batch_size
                        ):
                            backward_batch_idxs = mini_batch_idxs[
                                j : j + self.ppo_backward_batch_size
                            ]

                            backward_returns = returns[backward_batch_idxs]
                            backward_advantages = advantages[backward_batch_idxs]
                            backward_logprobs = logprobs[backward_batch_idxs]
                            backward_query_responses = query_responses[
                                backward_batch_idxs
                            ]
                            backward_masks = (
                                masks[backward_batch_idxs]
                                if masks is not None
                                else None
                            )
                            backward_position_ids = (
                                position_ids[backward_batch_idxs]
                                if position_ids is not None
                                else None
                            )
                            backwards_padding_masks = padding_masks[backward_batch_idxs]
                            backwards_value_padding_masks = value_padding_masks[
                                backward_batch_idxs
                            ]
                            backward_values = values[backward_batch_idxs]

                            # policy and value estimates for the current optimisation step
                            # pi_{theta] and V_{phi}
                            # TODO (SalmanMohammadi) implement minibatch model forward pass
                            pi_logits, phi_output = self._model(
                                backward_query_responses,
                                input_pos=backward_position_ids,
                                mask=backward_masks,
                            )

                            phi_output = phi_output[:, context_length - 1 : -1].squeeze(
                                -1
                            )
                            phi_output = phi_output.masked_fill(
                                backwards_value_padding_masks, 0.0
                            )

                            pi_logits = pi_logits[:, context_length - 1 : -1]
                            pi_logits /= self.temperature
                            pi_logprobs = torch.gather(
                                F.log_softmax(pi_logits, dim=-1),
                                2,
                                backward_query_responses[:, context_length:].unsqueeze(
                                    -1
                                ),
                            ).squeeze(-1)

                            pi_logprobs = pi_logprobs.masked_fill(
                                backwards_padding_masks, 1.0
                            )

                            loss, policy_loss, value_loss = self._loss_fn(
                                backward_logprobs,
                                pi_logprobs,
                                backward_advantages,
                                backward_values,
                                backward_returns,
                                padding_masks=backwards_padding_masks,
                                value_padding_masks=backwards_value_padding_masks,
                            )

                            prob_dist = torch.nn.functional.softmax(pi_logits, dim=-1)
                            entropy = torch.logsumexp(pi_logits, dim=-1) - torch.sum(
                                prob_dist * pi_logits, dim=-1
                            )
                            entropies.append(entropy.mean())
                            approx_policy_kls.append(
                                (0.5 * (pi_logprobs - backward_logprobs).pow(2)).mean()
                            )
                            losses.append(loss.detach().item())
                            policy_losses.append(policy_loss.item())
                            value_losses.append(value_loss.item())

                            loss.backward()
                            self._optimizer.step()
                            self._optimizer.zero_grad(set_to_none=True)

            self.epochs_run += 1
            # self.save_checkpoint(epoch=curr_epoch)
            pbar.update(1)
            pbar.set_description(
                f"Step: {curr_epoch+1}| reward: {rewards.sum(1).mean()}"
            )

            kl = logprobs - ref_logprobs
            log_dict = {
                "loss": torch.tensor(loss).mean().item(),
                "reward": rewards.sum(1).mean().item(),
                "kl": kl.sum(1).mean().item(),
                "policy_loss": torch.tensor(policy_losses).mean().item(),
                "value_loss": torch.tensor(value_losses).mean().item(),
                "approx_policy_kl": torch.tensor(approx_policy_kls).mean().item(),
                "entropy": torch.tensor(entropies).mean().item(),
            }
            self._metric_logger.log_dict(
                log_dict,
                step=self.global_step,
            )
            self.global_step += 1
            self.kl_controller.update(kl.sum(1).mean().item(), curr_epoch)

            # delete values and clear cache

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
    config.log_config(recipe_name="LoRAPPORecipeSingleDevice", cfg=cfg)
    recipe = LoRAPPORecipeSingleDevice(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())
