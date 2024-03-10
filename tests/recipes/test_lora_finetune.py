# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import runpy
import sys
from functools import partial
from typing import Dict

import pytest
import torch
import torchtune

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

test_lora_attn_modules = ["q_proj", "k_proj", "v_proj", "output_proj"]
torchtune.models.lora_small_test_ckpt = partial(
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

    @pytest.mark.parametrize("multi_gpu", [False, True])
    def test_loss(self, capsys, tmpdir, multi_gpu, monkeypatch):
        # No support for large scale test yet for LoRA
        ckpt = "lora_small_test_ckpt"
        expected_loss_values = self._fetch_expected_loss_values(ckpt)
        config_path = RECIPE_TESTS_DIR / "lora_finetune_test_config.yaml"
        recipe_name = (
            "lora_finetune_single_device"
            if not multi_gpu
            else "lora_finetune_distributed"
        )
        # TODO (rohan-varma): setting CUDA_VISIBLE_DEVICES to ignore all GPUs
        # on machine to simulate current CI environment that does not have GPUs.
        # Will consolidate as part of addressing https://github.com/pytorch-labs/torchtune/issues/473
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
        cmd = f"""
        tune {recipe_name}
            --config {config_path} \
            --override \
            output_dir={tmpdir} \
            enable_fsdp={multi_gpu} \
            model._component_=torchtune.models.{ckpt} \
            model_checkpoint={fetch_ckpt_model_path(ckpt)} \
            model.lora_rank=8 \
            model.lora_alpha=16 \
            model.apply_lora_to_mlp=False \
        """.split()

        # Have to attach this after so it parses correctly
        cmd += ['model.lora_attn_modules=["q_proj", "k_proj", "v_proj", "output_proj"]']
        monkeypatch.setattr(sys, "argv", cmd)
        with pytest.raises(SystemExit):
            with (
                single_box_init(init_pg=False)
                if multi_gpu
                else contextlib.nullcontext()
            ):
                runpy.run_path(TUNE_PATH, run_name="__main__")

        loss_values = fetch_loss_values(capsys.readouterr().err)
        validate_loss_values(loss_values, expected_loss_values)


class TestLoRAFinalCheckpoints:
    @pytest.mark.parametrize("enable_fsdp", [False, True])
    def test_save_and_load_merged_weights(self, tmpdir, enable_fsdp, monkeypatch):
        # No support for large scale test yet for LoRA
        ckpt = "lora_small_test_ckpt"

        config_path = RECIPE_TESTS_DIR / "lora_finetune_test_config.yaml"
        recipe_name = (
            "lora_finetune_single_device"
            if not enable_fsdp
            else "lora_finetune_distributed"
        )

        # Have to attach this after so it parses correctly
        lora_cfg = [
            'model.lora_attn_modules=["q_proj", "k_proj", "v_proj", "output_proj"]'
        ]

        # TODO: see analogous comment in TestLoRAFinetuneRecipe
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
        cmd = f"""
        tune {recipe_name}
            --config {config_path} \
            --override \
            enable_fsdp={enable_fsdp} \
            model._component_=torchtune.models.{ckpt} \
            model_checkpoint={fetch_ckpt_model_path(ckpt)} \
            model.lora_rank=8 \
            model.lora_alpha=16 \
            model.apply_lora_to_mlp=True \
            epochs=1
        """.split()
        cmd += lora_cfg
        cmd += ["full_bf16=False"]  # TODO: figure out why we need this
        baseline_cmd = cmd + [f"output_dir={tmpdir}{enable_fsdp}baseline"]
        merged_cmd = cmd + [
            f"output_dir={tmpdir}{enable_fsdp}merged_ckpt",
            "save_merged_final_checkpoint=True",
        ]
        cmds = [baseline_cmd, merged_cmd]
        for current_cmd in cmds:
            monkeypatch.setattr(sys, "argv", current_cmd)
            with pytest.raises(SystemExit):
                with (
                    single_box_init(init_pg=False)
                    if enable_fsdp
                    else contextlib.nullcontext()
                ):
                    runpy.run_path(TUNE_PATH, run_name="__main__")

        inputs = torch.randint(low=0, high=32_000, size=(2, 100))

        lora_model = lora_llama2_small_test_ckpt(
            lora_attn_modules=["q_proj", "k_proj", "v_proj", "output_proj"],
            apply_lora_to_mlp=True,
            lora_rank=8,
            lora_alpha=16,
        )

        # Load partial final ckpt + original weights and call fwd
        with open(f"{tmpdir}{enable_fsdp}baseline/model_0.ckpt", "rb") as f:
            lora_sd = torch.load(f, weights_only=True)
        with open(fetch_ckpt_model_path(ckpt), "rb") as f:
            base_model_sd = torch.load(f, weights_only=True)
        lora_model.load_state_dict(lora_sd["model"], strict=False)
        lora_model.load_state_dict(base_model_sd["model"], strict=False)
        baseline_out = lora_model(inputs)

        # Load merged final ckpt directly into llama2 and call fwd
        llama2_model = llama2_small_test_ckpt()
        with open(f"{tmpdir}{enable_fsdp}merged_ckpt/model_0.ckpt", "rb") as f:
            sd = torch.load(f, weights_only=True)
        llama2_model.load_state_dict(sd["model"])
        merged_ckpt_out = llama2_model(inputs)
        torch.testing.assert_close(baseline_out, merged_ckpt_out, rtol=1e-5, atol=1e-5)
