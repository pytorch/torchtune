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


class QuantizationRecipe:
    """
    Recipe for quantizing a Transformer-based LLM.

    Supported quantization modes are:
      8w: int8  weight only per axis group quantization
      4w: int4  weight only per axis group quantization
    after torch 2.3.0:
      8da4w: int8 dynamic activation quantization and int4 weight per axis group quantization
      8da4w-gptq: int8 dynamic activation quantization and int4 weight per axis group quantization with GPTQ
      4w-gptq: int4 weight only per axis group quantization with GPTQ
    """

    def __init__(self, cfg: DictConfig) -> None:
        self._device = utils.get_device(device=cfg.device)
        self._dtype = utils.get_dtype(dtype=cfg.dtype)
        self._quantization_mode = cfg.quantization_mode
        utils.set_seed(seed=cfg.seed)

    def load_checkpoint(self, checkpointer_cfg: DictConfig) -> Dict[str, Any]:
        self._checkpointer = config.instantiate(checkpointer_cfg)
        checkpoint_dict = self._checkpointer.load_checkpoint()
        return checkpoint_dict

    def setup(self, cfg: DictConfig) -> None:
        ckpt_dict = self.load_checkpoint(cfg.checkpointer)
        self._model = self._setup_model(
            model_cfg=cfg.model,
            model_state_dict=ckpt_dict[utils.MODEL_KEY],
        )

    def _setup_model(
        self,
        model_cfg: DictConfig,
        model_state_dict: Dict[str, Any],
    ) -> nn.Module:
        with utils.set_default_dtype(self._dtype), self._device:
            model = config.instantiate(model_cfg)

        model.load_state_dict(model_state_dict, assign=True)

        # Validate model was loaded in with the expected dtype.
        utils.validate_expected_param_dtype(model.named_parameters(), dtype=self._dtype)
        logger.info(f"Model is initialized with precision {self._dtype}.")

        # Ensure the cache is setup on the right device
        # with self._device:
        #     model.setup_caches(max_batch_size=1, dtype=self._dtype)

        return model

    @torch.no_grad()
    def quantize(self, cfg: DictConfig):
        from torchao.quantization.GPTQ import InputRecorder
        tokenizer = config.instantiate(cfg.tokenizer)
        calibration_seq_length = 100
        calibration_tasks = ['wikitext']
        inputs = InputRecorder(
            tokenizer,
            calibration_seq_length,
            vocab_size=self._model.tok_embeddings.weight.shape[0],
            device="cpu",
        ).record_inputs(
            calibration_tasks,
            5,
        ).get_inputs()
        # calib = 1, limit=1 [eleuther_eval.py:202] wikitext: {'word_perplexity,none': 6.282903500084389, 'word_perplexity_stderr,none': 'N/A', 'byte_perplexity,none': 1.4254172116452406, 'byte_perplexity_stderr,none': 'N/A', 'bits_per_byte,none': 0.5113842498565163, 'bits_per_byte_stderr,none': 'N/A', 'alias': 'wikitext'}
        # calib = 1, limit=None [eleuther_eval.py:202] wikitext: {'word_perplexity,none': 9.4572063787555, 'word_perplexity_stderr,none': 'N/A', 'byte_perplexity,none': 1.5222028600938176, 'byte_perplexity_stderr,none': 'N/A', 'bits_per_byte,none': 0.6061606359433654, 'bits_per_byte_stderr,none': 'N/A', 'alias': 'wikitext'}
        # calib = 5, limit=None [eleuther_eval.py:202] wikitext: {'word_perplexity,none': 9.181452425193726, 'word_perplexity_stderr,none': 'N/A', 'byte_perplexity,none': 1.513802586466077, 'byte_perplexity_stderr,none': 'N/A', 'bits_per_byte,none': 0.5981770771226372, 'bits_per_byte_stderr,none': 'N/A', 'alias': 'wikitext'}
        quantizer = utils.get_quantizer(self._quantization_mode, blocksize=128, percdamp=.01, groupsize=128)
        t0 = time.perf_counter()
        self._model = quantizer.quantize(self._model, inputs)
        t = time.perf_counter() - t0
        logger.info(
            f"Time for quantization: {t:.02f} sec"
        )
        logger.info(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")

    def save_checkpoint(self, cfg: DictConfig):
        ckpt_dict = self._model.state_dict()
        file_name = cfg.checkpointer.checkpoint_files[0].split(".")[0]
        torch.save(ckpt_dict, cfg.checkpointer.output_dir + file_name + "." + self._quantization_mode + ".pt")


@config.parse
def main(cfg: DictConfig) -> None:
    recipe = QuantizationRecipe(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.quantize(cfg=cfg)
    recipe.save_checkpoint(cfg=cfg)


if __name__ == "__main__":
    sys.exit(main())
