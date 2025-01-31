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
        prompts: List[torch.Tensor] of length B repeated G times
        completions: List[torch.Tensor] of length B*G
        all_text: List[str], length B*G (decoded completions)
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
            rewards[i] = 0.0

    return rewards


def correctness_reward(tokenizer, prompts, completions, all_text, batch_data):
    """
    Reward completions based on correctness of the model's answer.
    We look for an <answer>...</answer> section, compare with ground truth.

    Args:
        tokenizer: Tokenizer (not used directly since we have all_text)
        prompts: List[torch.Tensor] of length B repeated G times
        completions: List[torch.Tensor] of length B*G
        all_text: List[str], the decoded completions
        batch_data: Dictionary that should include 'result' for ground truth.

    Returns:
        A float32 tensor of shape [B*G], with 1.0 if correct, 0.1 if partial, etc.
    """
    ground_truths = batch_data.get('result', [])
    rewards = torch.zeros(len(completions), dtype=torch.float32)

    for i, text in enumerate(all_text):
        start_tag = "<answer>"
        end_tag = "</answer>"
        start_idx = text.find(start_tag)
        end_idx = text.find(end_tag, start_idx)

        extracted_answer = None
        if start_idx != -1 and end_idx != -1:
            extracted_answer = text[start_idx + len(start_tag):end_idx].strip()

        if i < len(ground_truths) and extracted_answer:
            if extracted_answer == ground_truths[i]:
                rewards[i] = 1.0
            else:
                rewards[i] = 0.1  # partial reward
        else:
            rewards[i] = 0.0

    return rewards

########################
# Utilities for function loading, etc.
########################

def _resolve_function(name: str):
    """Dynamically resolve a function by its full module path."""
    if name.startswith("recipes.grpo_finetune_single_device."):
        func_name = name.split(".")[-1]
        if func_name in globals():
            return globals()[func_name]
        else:
            raise ValueError(f"Function '{func_name}' not found in local scope.")
    else:
        module_name, function_name = name.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, function_name)


def compute_token_level_kl(policy_logprobs: torch.Tensor,
                            baseline_logprobs: torch.Tensor,
                            mask: torch.Tensor) -> torch.Tensor:
    """Compute approximate KL. e.g. (p_lprobs - b_lprobs) * mask, averaged."""
    kl = (policy_logprobs - baseline_logprobs) * mask
    denom = mask.sum() + 1e-6
    return kl.sum() / denom


def compute_token_level_loss(
    policy_logprobs: torch.Tensor,
    baseline_logprobs: torch.Tensor,
    mask: torch.Tensor,
    advantages: torch.Tensor,
    kl_coeff: float,
) -> torch.Tensor:
    """Token-level RL loss: PG + KL."""
    if advantages.dim() == 1:
        advantages = advantages.unsqueeze(1)

    # Policy gradient term
    pg_term = -advantages * policy_logprobs * mask
    denom = mask.sum() + 1e-6
    pg_loss = pg_term.sum() / denom

    # KL term
    kl_val = compute_token_level_kl(policy_logprobs, baseline_logprobs, mask)

    return pg_loss + kl_coeff * kl_val

class GRPOFinetuneRecipeSingleDevice(FTRecipeInterface):
    """
    Finetuning recipe for GRPO on a single GPU, with optional LoRA.
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
        self._use_lora = False
        lora_rank = None
        if "model" in cfg and ("lora_rank" in cfg.model or "use_lora" in cfg.model):
            candidate = cfg.model.get("lora_rank", 0)
            if candidate and candidate > 0:
                self._use_lora = True
                lora_rank = candidate
            if cfg.model.get("use_lora", False) is True:
                self._use_lora = True

        self._num_generations = cfg.get("num_generations", 1)
        self._max_generated_tokens = cfg.get("max_generated_tokens", 128)
        self._freeze_ref_model = cfg.get("freeze_ref_model", True)

    def setup(self, cfg: DictConfig) -> None:
        self._metric_logger = config.instantiate(cfg.metric_logger)
        self._metric_logger.log_config(cfg)

        self._tokenizer = config.instantiate(cfg.tokenizer)
        self._max_length = cfg.get("max_length", 8192)
        self._enable_kv_cache = cfg.get("enable_kv_cache", True)

        # Possibly a reward model or custom reward functions.
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
            should_load_recipe_state=False,
        )
        checkpoint_dict = self._checkpointer.load_checkpoint()
        self._save_adapter_weights_only = cfg.get('checkpointer').get('save_adapter_weights_only', False)

        # Initialize policy
        model_state = checkpoint_dict.get(training.MODEL_KEY, None)
        self._policy_model = self._initialize_model(cfg.model, model_state)

        # Initialize baseline
        if self._use_lora:
            # We'll disable adapter for baseline logprobs.
            self._freeze_ref_model = True
            self._baseline_model = self._policy_model
        else:
            self._baseline_model = self._initialize_model(cfg.model, model_state)
            self._baseline_model.eval()
            for param in self._baseline_model.parameters():
                param.requires_grad = False

        self._optimizer = self._setup_optimizer(cfg.optimizer)

        self._sampler, self._dataloader = self._setup_data(
            cfg.dataset, cfg.shuffle, cfg.batch_size
        )

        self._kl_coeff = cfg.kl_coeff
        self._batch_size = cfg.batch_size
        self._num_epochs = cfg.epochs
        self._temperature = cfg.get("temperature", 1.0)
        self._top_k = cfg.get("top_k", None)

        self.cache_ctx_manager = lambda enable_kv_cache: (
            local_kv_cache(
                self._policy_model,
                batch_size=self._batch_size,
                dtype=self._dtype,
                decoder_max_seq_len=self._tokenizer.max_seq_len + self._max_generated_tokens,
                device=self._device,
            ) if enable_kv_cache else contextlib.nullcontext()
        )

    def _initialize_model(self, cfg_model: DictConfig, base_model_state_dict: Optional[Dict[str, Any]] = None) -> nn.Module:
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
            if base_model_state_dict:
                base_missing, base_unexpected = model.load_state_dict(
                    base_model_state_dict, strict=False
                )
                validate_missing_and_unexpected_for_lora(
                    lora_attn_modules=self._lora_attn_modules,
                    apply_lora_to_mlp=self._apply_lora_to_mlp,
                    apply_lora_to_output=False,
                    base_missing=base_missing,
                    base_unexpected=base_unexpected,
                    lora_missing=None,
                    lora_unexpected=None,
                )
        else:
            if base_model_state_dict:
                model.load_state_dict(base_model_state_dict)

        if self._device.type != "cpu":
            memory_stats = training.get_memory_stats(device=self._device)
            training.log_memory_stats(memory_stats)
        return model

    def _setup_optimizer(self, cfg_optimizer: DictConfig) -> Optimizer:
        return config.instantiate(cfg_optimizer, params=self._policy_model.parameters())

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

    def _compute_rewards(
        self,
        tokenizer,
        prompts: List[torch.Tensor],
        completions: List[torch.Tensor],
        batch_data: Dict[str, Any],
    ) -> torch.Tensor:
        """
        For B prompts and self._num_generations completions/prompt, build [B*g].
        We decode all completions, then sum up all custom reward functions.
        """
        B = len(prompts)
        total_completions = len(completions)
        if total_completions != B * self._num_generations:
            raise ValueError("Mismatch in B*g vs completions length.")

        all_text = [tokenizer.decode(c.cpu().tolist()) for c in completions]
        total_rewards = torch.zeros(total_completions)

        # replicate batch_data (except tokens/labels) for the G completions.
        extended_data = {}
        for key, values in batch_data.items():
            if key not in ("tokens", "labels"):
                replicated = []
                for i in range(B):
                    replicated.extend([values[i]] * self._num_generations)
                extended_data[key] = replicated

        if self._reward_functions:
            repeated_prompts = []
            for i in range(B):
                repeated_prompts.extend([prompts[i]] * self._num_generations)

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
            raise NotImplementedError("Reward model usage not integrated.")
        else:
            raise ValueError("No custom reward functions or reward model specified.")

        # shape [B, G]
        total_rewards = total_rewards.view(B, self._num_generations)
        return total_rewards

    def compute_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        device = self._device
        input_ids = batch["tokens"].to(device)
        B = input_ids.size(0)

        # Generate completions
        all_completions = []
        prompt_lengths = []
        with torch.no_grad():
            for i in range(B):
                # figure out the actual prompt length
                prompt_len = (input_ids[i] != self._tokenizer.pad_id).sum().item()
                prompt_tensor = input_ids[i][:prompt_len].unsqueeze(0)

                prompt_lengths.append(prompt_len)
                for _ in range(self._num_generations):
                    with self.cache_ctx_manager(self._enable_kv_cache):
                        completion, _ = generation.generate(
                            model=self._policy_model,
                            prompt=prompt_tensor,
                            max_generated_tokens=self._max_generated_tokens,
                            temperature=self._temperature,
                            top_k=self._top_k,
                            pad_id=self._tokenizer.pad_id,
                            stop_tokens=self._tokenizer.stop_tokens,
                            rng=self._rng,
                        )
                    log.info(self._tokenizer.decode(completion[0].tolist()))
                    all_completions.append(completion[0])

        # compute rewards
        #  shape: [B, G]
        rewards_2d = self._compute_rewards(
            self._tokenizer,
            [row for row in input_ids],
            all_completions,
            batch,
        ).to(device)

        # build the full prompt+completion sequences
        full_sequences = []  # list of torch.Tensor
        full_prompts_len = []

        idx = 0
        for i in range(B):
            pl = prompt_lengths[i]
            prompt_seq = input_ids[i][:pl]
            for g in range(self._num_generations):
                comp = all_completions[idx]
                idx += 1
                # cat prompt + completion
                combined = torch.cat([prompt_seq, comp], dim=0)
                full_sequences.append(combined)
                full_prompts_len.append(pl)

        lengths = [seq.size(0) for seq in full_sequences]
        max_len = max(lengths)
        policy_input = torch.full(
            (len(full_sequences), max_len),
            fill_value=self._tokenizer.pad_id,
            dtype=full_sequences[0].dtype,
            device=device,
        )
        for i, seq in enumerate(full_sequences):
            policy_input[i, : seq.size(0)] = seq

        # Forward
        outputs = self._policy_model(policy_input)
        logits = outputs[:, :-1, :]  # drop the last position
        shifted_input_ids = policy_input[:, 1:]
        log_probs = logits.log_softmax(dim=-1)
        chosen_token_logprob = torch.gather(
            log_probs, dim=2, index=shifted_input_ids.unsqueeze(-1)
        ).squeeze(-1)

        # Insert a dummy zero for alignment
        B2, seq_len_minus1 = chosen_token_logprob.size()
        dummy_zeros = torch.zeros((B2, 1), device=device, dtype=chosen_token_logprob.dtype)
        policy_logprobs = torch.cat([dummy_zeros, chosen_token_logprob], dim=1)

        # Build final mask to ignore prompt and anything after first EOS
        final_mask = torch.zeros_like(policy_logprobs)
        for i, seq in enumerate(full_sequences):
            pl = full_prompts_len[i]
            length_i = seq.size(0)
            # set [pl:length_i] = 1
            final_mask[i, pl:length_i] = 1
            # find first eos
            eos_positions = (seq == self._tokenizer.eos_id).nonzero()
            if eos_positions.numel() > 0:
                eos_pos = eos_positions[0].item()
                final_mask[i, eos_pos:] = 0

        # get baseline logprobs
        if self._use_lora:
            with disable_adapter(self._policy_model), torch.no_grad():
                ref_outputs = self._policy_model(policy_input)
                ref_logits = ref_outputs[:, :-1, :]
                ref_log_probs = ref_logits.log_softmax(dim=-1)
                ref_chosen = torch.gather(
                    ref_log_probs, dim=2, index=shifted_input_ids.unsqueeze(-1)
                ).squeeze(-1)
                dummy_zeros2 = torch.zeros((B2, 1), device=device, dtype=ref_chosen.dtype)
                baseline_logprobs = torch.cat([dummy_zeros2, ref_chosen], dim=1)
        else:
            with torch.no_grad():
                ref_outputs = self._baseline_model(policy_input)
                ref_logits = ref_outputs[:, :-1, :]
                ref_log_probs = ref_logits.log_softmax(dim=-1)
                ref_chosen = torch.gather(
                    ref_log_probs, dim=2, index=shifted_input_ids.unsqueeze(-1)
                ).squeeze(-1)
                dummy_zeros2 = torch.zeros((B2, 1), device=device, dtype=ref_chosen.dtype)
                baseline_logprobs = torch.cat([dummy_zeros2, ref_chosen], dim=1)

        # advantages shape [B, G]
        means = rewards_2d.mean(dim=1, keepdim=True)
        stds = rewards_2d.std(dim=1, keepdim=True) + 1e-5
        norm_adv_2d = (rewards_2d - means) / stds
        advantages = norm_adv_2d.view(-1)

        # RL loss
        loss_val = compute_token_level_loss(
            policy_logprobs,
            baseline_logprobs,
            final_mask,
            advantages,
            self._kl_coeff,
        )

        return loss_val

    def update_baseline_model(self):
        if (not self._freeze_ref_model) and (not self._use_lora):
            self._baseline_model.load_state_dict(self._policy_model.state_dict())

    def train(self):
        for epoch in range(self._num_epochs):
            accumulated_loss = 0.0
            for step, batch_data in enumerate(self._dataloader):
                loss_val = self.compute_loss(batch_data)
                (loss_val / self._gradient_accumulation_steps).backward()
                accumulated_loss += loss_val.item()

                if (step + 1) % self._gradient_accumulation_steps == 0:
                    self._optimizer.step()
                    self._optimizer.zero_grad()

                    cur_step = (step + 1) // self._gradient_accumulation_steps
                    log.info(f"Epoch {epoch}, Step {cur_step}, Loss: {accumulated_loss:.4f}")
                    accumulated_loss = 0.0

            # End of epoch
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
