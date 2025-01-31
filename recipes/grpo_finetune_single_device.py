import contextlib
import math
import sys
import time
from functools import partial
from itertools import chain
from typing import Any, Dict, List, Optional, Tuple, Union
from warnings import warn
import random

import torch
from omegaconf import DictConfig
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler
from torchtune import config, generation, training, utils
from torchtune.data import CROSS_ENTROPY_IGNORE_IDX
from torchtune.modules import local_kv_cache
from torchtune.recipe_interfaces import FTRecipeInterface
from torchtune.training import DummyProfiler, PROFILER_KEY

# Only needed if we do LoRA.
from torchtune.modules.peft import (
    disable_adapter, 
    get_adapter_state_dict, 
    get_adapter_params, 
    set_trainable_params,
    validate_missing_and_unexpected_for_lora,
)

import importlib

log = utils.get_logger("DEBUG")
torch._dynamo.config.cache_size_limit = 16

import torch

from typing import List, Dict, Any

########################
# Custom collate to keep additional fields. 
########################

def padded_collate_verifiable(
    batch: List[Dict[str, Any]],
    padding_idx: int = 0,
    ignore_idx: int = CROSS_ENTROPY_IGNORE_IDX,
) -> Dict[str, torch.Tensor]:
    """
    Pad a batch of sequences to the longest sequence length in the batch, and
    convert integer lists to tensors. Also preserves all other columns as lists.
    """
    input_ids = pad_sequence(
        [torch.tensor(x["tokens"]) for x in batch],
        batch_first=True,
        padding_value=padding_idx,
    )

    labels = pad_sequence(
        [torch.tensor(x["labels"]) for x in batch],
        batch_first=True,
        padding_value=ignore_idx,
    )

    input_ids_seq_len = input_ids.shape[-1]
    labels_seq_len = labels.shape[-1]

    if input_ids_seq_len > labels_seq_len:
        labels = F.pad(
            labels, (0, input_ids_seq_len - labels_seq_len), value=ignore_idx
        )
    elif labels_seq_len > input_ids_seq_len:
        input_ids = F.pad(
            input_ids,
            (0, labels_seq_len - input_ids_seq_len),
            value=padding_idx,
        )

    collated = {
        "tokens": input_ids.long(),
        "labels": labels.long(),
    }

    for key in batch[0].keys():
        if key not in ("tokens", "labels"):
            collated[key] = [sample[key] for sample in batch]

    return collated

########################
# Verifiable reward function examples
########################

def thinking_reward(tokenizer, prompts, completions, all_text, batch_data):
    """
    Reward completions based on the number of lines within <think>...</think> tags.
    The more lines inside the tags, the higher the reward, normalized by total lines.

    Args:
        tokenizer: Tokenizer (not used directly in this function, since we already have all_text)
        prompts: List[torch.Tensor] of B (each prompt) repeated G times
        completions: List[torch.Tensor] of length B*G
        all_text: List[str] (decoded completions), length B*G
        batch_data: Additional batch info if needed.

    Returns:
        Torch float32 tensor of shape [B*G] with the computed reward.
    """
    rewards = torch.zeros(len(completions), dtype=torch.float32)

    for i, text in enumerate(all_text):
        # Extract content inside <think>...</think> tags
        start_tag = "<think>"
        end_tag = "</think>"

        total_lines = text.count("\n") + 1
        start_idx = text.find(start_tag)
        end_idx = text.find(end_tag, start_idx)

        if start_idx != -1 and end_idx != -1:
            think_content = text[start_idx + len(start_tag):end_idx].strip()
            num_lines = think_content.count("\n") + 1 if think_content else 0
            # Reward is fraction of lines inside <think> relative to total lines
            rewards[i] = num_lines / total_lines
        else:
            # Default = 0 if no <think> tags
            rewards[i] = 0.0

    return rewards

def correctness_reward(tokenizer, prompts, completions, all_text, batch_data):
    """
    Reward completions based on correctness of the model's answer.
    We look for an <answer>...</answer> section, compare with ground truth.
s
    Args:
        tokenizer: Tokenizer (not used directly since we have all_text)
        prompts: List[torch.Tensor]
        completions: List[torch.Tensor]
        all_text: List[str], the decoded completions
        batch_data: Dictionary that should include 'results' for each completion.

    Returns:
        A float32 tensor of shape [len(completions)], with 1.0 if correct, 0.1 if partial, etc.
    """
    # Try to read ground_truths from batch_data
    ground_truths = batch_data['result']
    rewards = torch.zeros(len(completions), dtype=torch.float32)

    for i, text in enumerate(all_text):
        start_tag = "<answer>"
        end_tag = "</answer>"
        start_idx = text.find(start_tag)
        end_idx = text.find(end_tag, start_idx)

        extracted_answer = None
        if start_idx != -1 and end_idx != -1:
            extracted_answer = text[start_idx + len(start_tag):end_idx].strip()

        # If we have ground_truths and this index is valid, compare.
        if i < len(ground_truths) and extracted_answer:
            if extracted_answer == ground_truths[i]:
                rewards[i] = 1.0
            else:
                rewards[i] = 0.1  # partial reward
        else:
            # No valid <answer> or no matching ground truth
            rewards[i] = 0.0

    return rewards


########################
# Utilities for function loading, token-level logprobs/loss/kl with optional EOS masking
########################

def _resolve_function(name: str):
    """Dynamically resolve a function by its full module path.
    If it references recipes.grpo_full_finetune_single_device, fallback to a local reference."""
    # custom fallback to avoid import error from recipes __init__
    if name.startswith("recipes.grpo_full_finetune_single_device."):
        func_name = name.split(".")[-1]
        # Lookup in module-level globals
        if func_name in globals():
            return globals()[func_name]
        else:
            raise ValueError(f"Function '{func_name}' not found in local scope.")
    else:
        module_name, function_name = name.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, function_name)

def get_per_token_logprobs(
    model: nn.Module,
    input_ids: torch.Tensor,
    tokenizer,
    prompt_len: int,
    mask_after_eos: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Given a batch of sequences (input_ids) of shape [B, seq_len], run the model to get logits
    for each token, shift by one so we align logits with the actual tokens, then return
    per-token logprobs for the entire sequence. Also return a mask that is 1 for valid tokens
    in the completion portion and 0 otherwise. This includes ignoring everything after the
    first EOS (if mask_after_eos=True).
    """
    outputs = model(input_ids)
    logits = outputs[:, :-1, :]  # drop last logits because there's no next token
    shifted_input_ids = input_ids[:, 1:]
    log_probs = logits.log_softmax(dim=-1)
    chosen_token_logprob = torch.gather(
        log_probs, dim=2, index=shifted_input_ids.unsqueeze(-1)
    ).squeeze(-1)

    pad_zeros = torch.zeros(
        (chosen_token_logprob.size(0), 1),
        dtype=chosen_token_logprob.dtype,
        device=chosen_token_logprob.device,
    )
    per_token_logprobs = torch.cat([pad_zeros, chosen_token_logprob], dim=1)

    B, seq_len = input_ids.shape
    valid_mask = torch.zeros_like(per_token_logprobs, dtype=torch.bool)

    for i in range(B):
        completion_start = prompt_len
        valid_mask[i, completion_start:] = True
        if mask_after_eos:
            eos_positions = (input_ids[i, prompt_len:] == tokenizer.eos_id).nonzero()
            if eos_positions.numel() > 0:
                first_eos = eos_positions[0].item() + prompt_len
                valid_mask[i, first_eos:] = False

    return per_token_logprobs, valid_mask.float()

def compute_token_level_kl(
    policy_logprobs: torch.Tensor, baseline_logprobs: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute an approximate KL by summing (p_lprobs - b_lprobs) * exp(p_lprobs), restricted by mask.
    Alternatively, a simpler approach is (policy_lprobs - baseline_lprobs) * mask, then average.
    """
    kl = (policy_logprobs - baseline_logprobs) * mask
    denom = mask.sum() + 1e-6
    kl_value = kl.sum() / denom
    return kl_value

def compute_token_level_loss(
    policy_logprobs: torch.Tensor,
    baseline_logprobs: torch.Tensor,
    mask: torch.Tensor,
    advantages: torch.Tensor,
    kl_coeff: float,
) -> torch.Tensor:
    """
    Token level loss for the group, with KL divergence:
      - We apply the advantage to each token's log probability.
      - We compute a token-level KL as well.
    The final objective is something like:
      L = - mean( advantage * policy_logprobs * mask ) + kl_coeff * KL
    where KL is a mean over tokens of (policy_lprobs - baseline_lprobs).
    """
    if advantages.dim() == 1:
        advantages = advantages.unsqueeze(1)

    pg_term = -advantages * policy_logprobs * mask
    denom = mask.sum() + 1e-6
    pg_loss = pg_term.sum() / denom

    kl_value = compute_token_level_kl(policy_logprobs, baseline_logprobs, mask)

    return pg_loss + kl_coeff * kl_value

class GRPOFinetuneRecipeSingleDevice(FTRecipeInterface):
    """
    Finetuning recipe for GRPO on a single GPU, integrating optional LoRA and token-level KL.

    Features:
      - Group-based reward computation, no critic model.
      - Fine-grained token-level KL and advantage weighting.
      - Optional LoRA-based training (frozen reference model).
      - Gradient accumulation and activation checkpointing for memory.
      - Supports bf16 and fp32 precision.
      - Custom reward functions for flexible reward shaping.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self._device = utils.get_device(device=cfg.device)
        self._dtype = training.get_dtype(cfg.dtype, device=self._device)

        if self._dtype == torch.float16:
            raise RuntimeError("fp16 training is not supported with this recipe. Use bf16 or fp32.")

        self._output_dir = cfg.output_dir
        self._log_every_n_steps = cfg.get("log_every_n_steps", 1)
        self.seed = training.set_seed(seed=cfg.seed)
        self._rng = torch.Generator(self._device).manual_seed(self.seed)
        self._total_steps = 0

        self._gradient_accumulation_steps = cfg.gradient_accumulation_steps

        # Check if LoRA is used
        self._use_lora = False
        lora_rank = None
        if "model" in cfg and ("lora_rank" in cfg.model or "use_lora" in cfg.model):
            candidate = cfg.model.get("lora_rank", 0)
            if candidate and candidate > 0:
                self._use_lora = True
                lora_rank = candidate
            if cfg.model.get("use_lora", False) is True:
                self._use_lora = True

        # Number of completions to generate per prompt
        self._num_generations = cfg.get("num_generations", 1)
        self._max_generated_tokens = cfg.get("max_generated_tokens", 128)

        self._freeze_ref_model = cfg.get("freeze_ref_model", True)

    def setup(self, cfg: DictConfig) -> None:
        self._metric_logger = config.instantiate(cfg.metric_logger)
        self._metric_logger.log_config(cfg)

        self._tokenizer = config.instantiate(cfg.tokenizer)
        self._max_length = cfg.get("max_length", 8192)
        self._enable_kv_cache = cfg.get("   ", True)

        # Reward model or custom reward functions
        self._reward_model = (
            self._initialize_model(cfg.reward_model) if "reward_model" in cfg else None
        )
        self._reward_functions = []
        for func_name in cfg.get("reward_functions", []):
            resolved_func = _resolve_function(func_name)
            if callable(resolved_func):
                self._reward_functions.append(resolved_func)
            else:
                raise ValueError(f"Reward function '{func_name}' is not callable.")

        # Initialize checkpointing
        self._checkpointer = config.instantiate(
            cfg.get('checkpointer'),
            should_load_recipe_state=False, # TODO: add resuming!
        )
        # Load initial checkpoint (TODO: resuming)
        checkpoint_dict = self._checkpointer.load_checkpoint()
        self._save_adapter_weights_only = cfg.get('checkpointer').get('save_adapter_weights_only', False)

        # Initialize policy (model!)
        self._policy_model = self._initialize_model(cfg.model, checkpoint_dict[training.MODEL_KEY])

        # Initialize baseline (reference) model
        if self._use_lora:
            self._freeze_ref_model = True
        else:
            self._baseline_model = self._initialize_model(cfg.model)
            self._baseline_model.eval()
            for param in self._baseline_model.parameters():
                param.requires_grad = False

        # Initialize optimizer
        self._optimizer = self._setup_optimizer(cfg.optimizer)

        # Initialize datasets
        self._sampler, self._dataloader = self._setup_data(
            cfg.dataset, cfg.shuffle, cfg.batch_size
        )

        self._setup_training_parameters(cfg)

    def _setup_training_parameters(self, cfg: DictConfig) -> None:
        self._kl_coeff = cfg.kl_coeff
        self._batch_size = cfg.batch_size
        self._num_epochs = cfg.epochs
        self._temperature = cfg.get("temperature", 1.0)
        self._top_k = cfg.get("top_k", None)

        # Set up local KV-cache context if enabled
        self.cache_ctx_manager = lambda enable_kv_cache: (
            local_kv_cache(
                self._policy_model,
                batch_size=self._batch_size,
                dtype=self._dtype,
                decoder_max_seq_len=self._tokenizer.max_seq_len + self._max_generated_tokens,
                device=self._device,
            )
            if enable_kv_cache  
            else contextlib.nullcontext()
        )

    def _initialize_model(self, cfg_model: DictConfig, base_model_state_dict: Dict[str, Any], lora_adapter_state_dict: Optional[Dict[str, Any]] = None ) -> nn.Module:
        with training.set_default_dtype(self._dtype), self._device:
            model = config.instantiate(cfg_model)
        if self._use_lora:
            self._lora_rank = cfg_model.lora_rank
            self._lora_alpha = cfg_model.lora_alpha
            self._lora_attn_modules = list(cfg_model.lora_attn_modules)
            self._apply_lora_to_mlp = cfg_model.apply_lora_to_mlp
            self._apply_lora_to_output = cfg_model.get("apply_lora_to_output", False)
            self.adapter_params = get_adapter_params(model)
            set_trainable_params(model, self.adapter_params)
            base_missing, base_unexpected = model.load_state_dict(
                base_model_state_dict, strict=False
            )
            if lora_adapter_state_dict:
                lora_missing, lora_unexpected = model.load_state_dict(lora_adapter_state_dict, strict=False)
            else: 
                lora_missing, lora_unexpected = None, None
            validate_missing_and_unexpected_for_lora(
                lora_attn_modules=self._lora_attn_modules,
                apply_lora_to_mlp=self._apply_lora_to_mlp,
                apply_lora_to_output=False,
                base_missing=base_missing,
                base_unexpected=base_unexpected,
                lora_missing=lora_missing,
                lora_unexpected=lora_unexpected,
            )
        else:
            model.load_state_dict(base_model_state_dict)

        if self._device.type != "cpu":
            memory_stats = training.get_memory_stats(device=self._device)
            training.log_memory_stats(memory_stats)
        return model

    def _setup_optimizer(self, cfg_optimizer: DictConfig) -> Optimizer:
        optimizer = config.instantiate(cfg_optimizer, params=self._policy_model.parameters())
        return optimizer

    def _setup_data(self, dataset_cfg: DictConfig, shuffle: bool, batch_size: int):
        dataset = config.instantiate(dataset_cfg, self._tokenizer)
        sampler = DistributedSampler(dataset) if shuffle else None
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=padded_collate_verifiable,
        )
        return sampler, dataloader

    #TODO: Add checkpoint saving.

    def _compute_rewards(
        self,
        tokenizer,
        prompts: List[torch.Tensor],
        completions: List[torch.Tensor],
        batch_data: Dict[str, Any],
    ) -> torch.Tensor:
        device = self._device
        B = len(prompts)
        total_completions = len(completions)
        if total_completions != B * self._num_generations:
            raise ValueError("Mismatch in B*g vs completions length.")

        # Convert all completions to text
        all_text = [tokenizer.decode(c.cpu().tolist()) for c in completions]
        total_rewards = torch.zeros(total_completions)

        # Repeat each prompt num_generations times (so B -> B*g)
        repeated_prompts = []
        for i in range(B):
            repeated_prompts.extend([prompts[i]] * self._num_generations)

        # Build a new dict called extended_data that replicates each value
        # for additional ground truth keys.
        extended_data = {}
        for key, values in batch_data.items():
            if key not in ("tokens", "labels"):
                replicated = []
                for i in range(B):
                    replicated.extend([values[i]] * self._num_generations)
                extended_data[key] = replicated

        if self._reward_functions:
            for func in self._reward_functions:
                rw = func(
                    tokenizer,
                    repeated_prompts,
                    completions,
                    all_text,
                    extended_data,
                )
                if not isinstance(rw, torch.Tensor):
                    rw = torch.tensor(rw, dtype=torch.float32)
                total_rewards += rw
        elif self._reward_model:
            raise NotImplementedError("Reward model usage not integrated here.")
        else:
            raise ValueError("No custom reward functions or reward model specified.")

        # Reshape rewards into [B, num_generations]
        total_rewards = total_rewards.view(B, self._num_generations)
        return total_rewards

    def compute_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Given a batch of data, generate completions, compute rewards, and return the final RL loss.
        This function does NOT perform backward() or step() yet.
        """
        input_ids = batch["tokens"].to(self._device)
        B = input_ids.size(0)
       
        # Generate completions
        all_completions = []
        with torch.no_grad():
            for i in range(B):
                prompt_tensor = input_ids[i].unsqueeze(0)
                for _ in range(self._num_generations):
                    with self.cache_ctx_manager(self._enable_kv_cache):
                        completion, _ = generation.generate(
                            model=self._policy_model,
                            prompt=prompt_tensor,
                            max_generated_tokens=self._max_length,
                            temperature=self._temperature,
                            top_k=self._top_k,
                            pad_id=self._tokenizer.pad_id,
                            stop_tokens=self._tokenizer.stop_tokens,
                            rng=self._rng,
                        )
                        log.info(self._tokenizer.decode(completion[0].tolist()))
                    c = completion[0]
                    all_completions.append(c)

        rewards_2d = self._compute_rewards(
            self._tokenizer,
            input_ids,
            all_completions,
            batch
        ).to(self._device)

        cat_tensors = []
        prompt_lengths = []
        for i in range(B):
            prompt_len = (input_ids[i] != self._tokenizer.pad_id).sum().item()
            for g in range(self._num_generations):
                idx = i * self._num_generations + g
                c = all_completions[idx]
                cat_tensors.append(c)
                prompt_lengths.append(prompt_len)

        cat_lengths = [t.size(0) for t in cat_tensors]
        max_len = max(cat_lengths)
        batched_input_ids = torch.full(
            (len(cat_tensors), max_len),
            fill_value=self._tokenizer.pad_id,
            dtype=cat_tensors[0].dtype,
            device=self._device,
        )
        for i, seq in enumerate(cat_tensors):
            batched_input_ids[i, : seq.size(0)] = seq

        policy_logprobs, _ = get_per_token_logprobs(
            self._policy_model,
            batched_input_ids,
            self._tokenizer,
            prompt_len=0,
            mask_after_eos=False,
        )

        # If LoRA, just disable the adapter for baseline
        if (self._use_lora):
            with disable_adapter(self._policy_model), torch.no_grad():
                baseline_logprobs, _ = get_per_token_logprobs(
                    self._policy_model,
                    batched_input_ids,
                    self._tokenizer,
                    prompt_len=0,
                    mask_after_eos=False,
                )
        else:
            with torch.no_grad():
                baseline_logprobs, _ = get_per_token_logprobs(
                    self._baseline_model,
                    batched_input_ids,
                    self._tokenizer,
                    prompt_len=0,
                    mask_after_eos=False,
                )

        final_mask = torch.zeros_like(policy_logprobs)
        for i in range(len(cat_tensors)):
            pl = prompt_lengths[i]
            final_mask[i, pl:] = 1
            row = batched_input_ids[i]
            eos_positions = (row == self._tokenizer.eos_id).nonzero()
            if eos_positions.numel() > 0:
                eos_pos = eos_positions[0].item()
                final_mask[i, eos_pos:] = 0

        means = rewards_2d.mean(dim=1, keepdim=True)
        stds = rewards_2d.std(dim=1, keepdim=True) + 1e-5
        advantages_2d = (rewards_2d - means) / stds
        advantages = advantages_2d.view(-1)

        loss = compute_token_level_loss(
            policy_logprobs,
            baseline_logprobs,
            final_mask,
            advantages,
            self._kl_coeff,
        )
        return loss

    def update_baseline_model(self):
        if not self._freeze_ref_model:
            self._baseline_model.load_state_dict(self._policy_model.state_dict())

    def train(self):
        for epoch in range(self._num_epochs):
            accumulated_loss = 0.0
            for step, batch_data in enumerate(self._dataloader):
                # Forward pass to compute RL loss
                loss_val = self.compute_loss(batch_data)

                # Scale loss by gradient_accumulation_steps so net gradient remains correct
                (loss_val / self._gradient_accumulation_steps).backward()
                accumulated_loss += loss_val.item()

                if (step + 1) % self._gradient_accumulation_steps == 0:
                    self._optimizer.step()
                    self._optimizer.zero_grad()

                    # logging
                    cur_step = (step + 1) // self._gradient_accumulation_steps
                    log.info(f"Epoch {epoch}, Step {cur_step}, Loss: {accumulated_loss:.4f}")
                    accumulated_loss = 0.0

            # End of epoch: update baseline model (unless frozen)
            self.update_baseline_model()
            log.info(f"Completed epoch {epoch}")

@config.parse
def recipe_main(cfg: DictConfig) -> None:
    config.log_config(recipe_name="GRPOFinetuneRecipeSingleDevice", cfg=cfg)
    recipe = GRPOFinetuneRecipeSingleDevice(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.train()
    recipe.cleanup()

if __name__ == "__main__":
    sys.exit(recipe_main())