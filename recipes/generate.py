# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import sys
import time
from typing import Any, Dict

import torch
from omegaconf import DictConfig

from torch import nn

from torchtune import config, utils

logger = utils.get_logger("DEBUG")


class InferenceRecipe:
    def __init__(self, cfg: DictConfig) -> None:
        self._device = utils.get_device(device=cfg.device)
        self._dtype = utils.get_dtype(dtype=cfg.dtype)

        utils.set_seed(seed=cfg.seed)

    def load_checkpoint(self, checkpointer_cfg: DictConfig) -> Dict[str, Any]:
        checkpointer = config.instantiate(checkpointer_cfg)
        checkpoint_dict = checkpointer.load_checkpoint()
        return checkpoint_dict

    def setup(self, cfg: DictConfig) -> None:
        ckpt_dict = self.load_checkpoint(cfg.checkpointer)
        self._model = self._setup_model(
            model_cfg=cfg.model,
            model_state_dict=ckpt_dict[utils.MODEL_KEY],
        )
        self._tokenizer = config.instantiate(cfg.tokenizer)

    def _setup_model(
        self,
        model_cfg: DictConfig,
        model_state_dict: Dict[str, Any],
    ) -> nn.Module:
        with utils.set_default_dtype(self._dtype), self._device:
            model = config.instantiate(model_cfg)

        model.load_state_dict(model_state_dict)

        # Validate model was loaded in with the expected dtype.
        utils.validate_expected_param_dtype(model.named_parameters(), dtype=self._dtype)
        logger.info(f"Model is initialized with precision {self._dtype}.")

        # Ensure the cache is setup on the right device
        with self._device:
            model.setup_caches(max_batch_size=1, dtype=self._dtype)
        return model

    def _multinomial_sample_one_no_sync(self, probs_sort):
        q = torch.empty_like(probs_sort).exponential_(1)
        return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)

    def _logits_to_probs(self, logits: torch.Tensor, temperature: float, top_k: int):
        logits = logits / max(temperature, 1e-5)

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            pivot = v.select(-1, -1).unsqueeze(-1)
            logits = torch.where(logits < pivot, -float("Inf"), logits)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return probs

    def _sample(self, logits: torch.Tensor, temperature: float, top_k: int):
        probs = self._logits_to_probs(logits[0, -1], temperature, top_k)
        idx_next = self._multinomial_sample_one_no_sync(probs)
        return idx_next, probs

    def _prefill(
        self,
        model: nn.Module,
        x: torch.Tensor,
        input_pos: torch.Tensor,
        temperature: float,
        top_k: int,
    ) -> torch.Tensor:
        # input_pos: [B, S]
        logits = model(x, input_pos)
        return self._sample(logits, temperature, top_k)[0]

    def _decode_one_token(
        self,
        model: nn.Module,
        x: torch.Tensor,
        input_pos: torch.Tensor,
        temperature: float,
        top_k: int,
    ):
        # input_pos: [B, 1]
        assert input_pos.shape[-1] == 1
        logits = model(x, input_pos)
        return self._sample(logits, temperature, top_k)

    def _decode_n_tokens(
        self,
        model: nn.Module,
        cur_token: torch.Tensor,
        input_pos: torch.Tensor,
        num_new_tokens: int,
        temperature: float,
        top_k: int,
    ):
        new_tokens, new_probs = [], []
        for i in range(num_new_tokens):
            with torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_mem_efficient=False, enable_math=True
            ):
                next_token, next_prob = self._decode_one_token(
                    model, cur_token, input_pos, temperature, top_k
                )
                input_pos += 1
                new_tokens.append(next_token.clone())
                new_probs.append(next_prob.clone())
                cur_token = next_token.view(1, -1)
        return new_tokens, new_probs

    @torch.no_grad()
    def generate(self, cfg: DictConfig):
        tokens = self._tokenizer.encode(cfg.prompt, add_bos=True, add_eos=False)
        prompt = torch.tensor(tokens, dtype=torch.int, device=self._device)

        prompt_length = prompt.size(0)
        total_length = prompt_length + cfg.max_new_tokens

        empty = torch.empty(total_length, dtype=torch.int, device=self._device)
        empty[:prompt_length] = prompt
        seq = empty
        input_pos = torch.arange(0, prompt_length, device=self._device)

        next_token = self._prefill(
            self._model,
            prompt.view(1, -1),
            input_pos,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
        )
        seq[prompt_length] = next_token
        input_pos = torch.tensor([prompt_length], device=self._device, dtype=torch.int)

        t0 = time.perf_counter()
        generated_tokens, _ = self._decode_n_tokens(
            self._model,
            next_token.view(1, -1),
            input_pos,
            cfg.max_new_tokens - 1,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
        )
        seq[prompt_length + 1 :] = torch.cat(generated_tokens)
        t = time.perf_counter() - t0

        logger.info(self._tokenizer.decode(seq.tolist()))

        tokens_generated = seq.size(0) - prompt_length
        tokens_sec = tokens_generated / t
        print(f"Time for inference: {t:.02f} sec total, {tokens_sec:.02f} tokens/sec")


@config.parse
def main(cfg: DictConfig) -> None:
    recipe = InferenceRecipe(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.generate(cfg=cfg)


if __name__ == "__main__":
    sys.exit(main())
