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
from recipes.lora_finetune import LoRAFinetuneRecipe
from recipes.params.lora_finetune import LoRAFinetuneParams
from recipes.tests.utils import (
    default_recipe_kwargs,
    fetch_loss_values,
    lora_llama2_small_test_ckpt,
    validate_loss_values,
)
from tests.test_utils import single_box_init

from torchtune import models

test_lora_attn_modules = ["q_proj", "k_proj", "v_proj", "output_proj"]
models.ALL_MODELS["lora_small_test_ckpt"] = partial(
    lora_llama2_small_test_ckpt, lora_attn_modules=test_lora_attn_modules
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

    @pytest.mark.parametrize("enable_fsdp", [False, True])
    def test_loss(self, enable_fsdp, capsys, pytestconfig):
        context_manager = single_box_init if enable_fsdp else contextlib.nullcontext
        with context_manager():
            # No support for large scale test yet for LoRA
            ckpt = "lora_small_test_ckpt"
            expected_loss_values = self._fetch_expected_loss_values(ckpt)
            kwargs_values = default_recipe_kwargs(ckpt)
            kwargs_values.update(enable_fsdp=enable_fsdp)
            kwargs_values["lora_attn_modules"] = test_lora_attn_modules
            recipe_params = LoRAFinetuneParams(**kwargs_values)

            recipe = LoRAFinetuneRecipe(recipe_params)
            recipe.setup(params=recipe_params)
            recipe.train()

            loss_values = fetch_loss_values(capsys.readouterr().err)
            validate_loss_values(loss_values, expected_loss_values)
