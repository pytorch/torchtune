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
    """
    Recipe for generating tokens from a dense Transformer-based LLM.

    Currently this recipe support single-GPU generation only. Speculative
    decoding is not supported.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self._device = utils.get_device(device=cfg.device)
        self._dtype = utils.get_dtype(dtype=cfg.dtype)
        self._quantization_mode = cfg.quantization_mode

        utils.set_seed(seed=cfg.seed)

    def load_checkpoint(self, checkpointer_cfg: DictConfig, weights_only: bool = True) -> Dict[str, Any]:
        checkpointer = config.instantiate(checkpointer_cfg)
        checkpoint_dict = checkpointer.load_checkpoint(weights_only=weights_only)
        return checkpoint_dict

    def setup(self, cfg: DictConfig) -> None:
        # weights_only needs to be False when loading a quantized model
	weights_only = (self._quantization_mode is None)
        ckpt_dict = self.load_checkpoint(cfg.checkpointer, weights_only=weights_only)
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

        if self._quantization_mode is not None:
            quantizer = utils.get_quantizer(self._quantization_mode)
            model = quantizer.quantize(model)

        model.load_state_dict(model_state_dict, assign=True)

        # Validate model was loaded in with the expected dtype.
        # TODO: enable this for quantization as well
        if self._quantization_mode is None:
            utils.validate_expected_param_dtype(model.named_parameters(), dtype=self._dtype)
        logger.info(f"Model is initialized with precision {self._dtype}.")

        # Ensure the cache is setup on the right device
        with self._device:
            model.setup_caches(max_batch_size=1, dtype=self._dtype)

        return model

    @torch.no_grad()
    def generate(self, cfg: DictConfig):
        tokens = self._tokenizer.encode(cfg.prompt, add_bos=True, add_eos=False)
        prompt = torch.tensor(tokens, dtype=torch.int, device=self._device)

        t0 = time.perf_counter()
        generated_tokens = utils.generate(
            model=self._model,
            prompt=prompt,
            max_generated_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
            eos_id=self._tokenizer.eos_id,
        )
        t = time.perf_counter() - t0

        logger.info(self._tokenizer.decode(generated_tokens))

        tokens_generated = len(generated_tokens) - prompt.size(0)
        tokens_sec = tokens_generated / t
        logger.info(
            f"Time for inference: {t:.02f} sec total, {tokens_sec:.02f} tokens/sec"
        )
        logger.info(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


@config.parse
def main(cfg: DictConfig) -> None:
    recipe = InferenceRecipe(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.generate(cfg=cfg)


if __name__ == "__main__":
    sys.exit(main())
