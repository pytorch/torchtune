# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import runpy
import torch
import sys
from functools import partial
from typing import Dict

import pytest

from tests.common import TUNE_PATH
from tests.recipes.common import RECIPE_TESTS_DIR

from tests.recipes.utils import (
    fetch_ckpt_model_path,
    fetch_loss_values,
    llama2_small_test_ckpt,
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
    def test_loss(self, capsys, tmpdir, enable_fsdp, monkeypatch):
        # No support for large scale test yet for LoRA
        ckpt = "lora_small_test_ckpt"
        expected_loss_values = self._fetch_expected_loss_values(ckpt)
        import pdb; pdb.set_trace()
        config_path = RECIPE_TESTS_DIR / "lora_finetune_test_config.yaml"
        cmd = f"""
        tune lora_finetune
            --config {config_path} \
            --override \
            output_dir={tmpdir} \
            model._component_=torchtune.models.{ckpt} \
            model_checkpoint={fetch_ckpt_model_path(ckpt)} \
            model.lora_rank=8 \
            model.lora_alpha=16 \
            model.apply_lora_to_mlp=False \
        """.split()

        # Have to attach this after so it parses correctly
        cmd += ['model.lora_attn_modules=["q_proj", "k_proj", "v_proj", "output_proj"]']

        if enable_fsdp:
            cmd.append("enable_fsdp=True")
            context_manager = single_box_init
        else:
            context_manager = contextlib.nullcontext

        with context_manager():
            monkeypatch.setattr(sys, "argv", cmd)
            with pytest.raises(SystemExit):
                runpy.run_path(TUNE_PATH, run_name="__main__")

        loss_values = fetch_loss_values(capsys.readouterr().err)
        validate_loss_values(loss_values, expected_loss_values)

class TestLoRAFinalCheckpoints:
    # @pytest.mark.parametrize("enable_fsdp", [False, True])
    def test_save_merged_weights(self, tmpdir, enable_fsdp, monkeypatch):
        # No support for large scale test yet for LoRA
        ckpt = "lora_small_test_ckpt"

        config_path = RECIPE_TESTS_DIR / "lora_finetune_test_config.yaml"
        cmd = f"""
        tune lora_finetune
            --config {config_path} \
            --override \
            output_dir={tmpdir} \
            model._component_=torchtune.models.{ckpt} \
            model_checkpoint={fetch_ckpt_model_path(ckpt)} \
            model.lora_rank=8 \
            model.lora_alpha=16 \
            model.apply_lora_to_mlp=True \
            save_full_final_checkpoint=True
            merge_lora_weights=True
        """.split()

        # Have to attach this after so it parses correctly
        cmd += ['model.lora_attn_modules=["q_proj", "k_proj", "v_proj", "output_proj"]']

        # if enable_fsdp:
        #     cmd.append("--enable-fsdp")
        #     context_manager = contextlib.nullcontext
        # else:
        #     context_manager = single_box_init
        # with context_manager():
        monkeypatch.setattr(sys, "argv", cmd)
        with pytest.raises(SystemExit):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        base_model = llama2_small_test_ckpt()
        with open(f'{tmpdir}/model_0.ckpt', 'rb') as f:
            sd = torch.load(f)
            base_model.load_state_dict(f)
