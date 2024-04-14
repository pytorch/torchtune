# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import torch
from omegaconf import DictConfig

from torch import nn

from torchtune import config, utils

logger = utils.get_logger("DEBUG")


class QuantizationRecipe:
    """
    Recipe for quantizing a Transformer-based LLM.
    Uses quantizer classes from torchao to quantize a model.
    Please refer to `recipes/configs/quantize.yaml` for supported quantizers and how to use them.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self._device = utils.get_device(device=cfg.device)
        self._dtype = utils.get_dtype(dtype=cfg.dtype)
        self._quantizer = config.instantiate(cfg.quantizer)
        self._quantization_mode = utils.get_quantizer_mode(self._quantizer)
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

        model.load_state_dict(model_state_dict)

        # Validate model was loaded in with the expected dtype.
        utils.validate_expected_param_dtype(model.named_parameters(), dtype=self._dtype)
        logger.info(f"Model is initialized with precision {self._dtype}.")
        return model

    @torch.no_grad()
    def quantize(self, cfg: DictConfig):
        t0 = time.perf_counter()
        self._model = self._quantizer.quantize(self._model)
        t = time.perf_counter() - t0
        logger.info(f"Time for quantization: {t:.02f} sec")
        logger.info(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")

    def save_checkpoint(self, cfg: DictConfig):
        ckpt_dict = self._model.state_dict()
        file_name = cfg.checkpointer.checkpoint_files[0].split(".")[0]

        output_dir = Path(cfg.checkpointer.output_dir)
        output_dir.mkdir(exist_ok=True)
        checkpoint_file = Path.joinpath(
            output_dir, f"{file_name}-{self._quantization_mode}"
        ).with_suffix(".pt")

        torch.save(ckpt_dict, checkpoint_file)
        logger.info(
            "Model checkpoint of size "
            f"{os.path.getsize(checkpoint_file) / 1000**3:.2f} GB "
            f"saved to {checkpoint_file}"
        )


@config.parse
def main(cfg: DictConfig) -> None:
    recipe = QuantizationRecipe(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.quantize(cfg=cfg)
    recipe.save_checkpoint(cfg=cfg)


if __name__ == "__main__":
    sys.exit(main())
