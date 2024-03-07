# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import logging

from functools import partial
from typing import Dict

import pytest

from omegaconf import OmegaConf
from recipes.lora_finetune_multi_gpu import LoRAFinetuneDistributedRecipe
from recipes.lora_finetune_single_gpu import LoRAFinetuneSingleDeviceRecipe

from recipes.tests.utils import (
    default_recipe_kwargs,
    fetch_loss_values,
    lora_llama2_small_test_ckpt,
    validate_loss_values,
)
from tests.test_utils import single_box_init

from torchtune import models

test_lora_attn_modules = ["q_proj", "k_proj", "v_proj", "output_proj"]
models.lora_small_test_ckpt = partial(
    lora_llama2_small_test_ckpt,
    lora_attn_modules=test_lora_attn_modules,
    apply_lora_to_mlp=False,
)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestLoRAFinetuneRecipe:
    def _fetch_expected_loss_values(self, ckpt) -> Dict[str, float]:
        small_test_ckpt_loss_values = {
            "1|1|": 10.5074,
            "1|2|": 10.5614,
            "2|1|": 10.5205,
            "2|2|": 10.4918,
        }
        if "small_test_ckpt" in ckpt:
            return small_test_ckpt_loss_values
        # TODO: no support for large scale test yet for LoRA
        raise ValueError(f"Unknown ckpt {ckpt}")

    @pytest.mark.parametrize("multi_gpu", [False, True])
    def test_loss(self, multi_gpu, capsys, pytestconfig):
        context_manager = single_box_init if enable_fsdp else contextlib.nullcontext
        with context_manager():
            # No support for large scale test yet for LoRA
            ckpt = "lora_small_test_ckpt"
            expected_loss_values = self._fetch_expected_loss_values(ckpt)
            kwargs_values = default_recipe_kwargs(ckpt)
            kwargs_values["model"].update(
                {
                    "lora_attn_modules": test_lora_attn_modules,
                    "apply_lora_to_mlp": False,
                    "lora_rank": 8,
                    "lora_alpha": 16,
                    # Note: multi-gpu just signifies to run the
                    # recipe that supports multi-gpu training w/
                    # distributed + FSDP. In CI, this test
                    # initializes distributed but runs on a single
                    # CPU: distributed CI still needs to be enabled:
                    # https://github.com/pytorch-labs/torchtune/issues/219
                    "enable_fsdp": multi_gpu,
                }
            )
            recipe_cfg = OmegaConf.create(kwargs_values)
            if multi_gpu:
                recipe = LoRAFinetuneDistributedRecipe(recipe_cfg)
            else:
                recipe = LoRAFinetuneSingleDeviceRecipe(recipe_cfg)

            recipe.setup(cfg=recipe_cfg)
            recipe.train()

            loss_values = fetch_loss_values(capsys.readouterr().err)
            validate_loss_values(loss_values, expected_loss_values)
