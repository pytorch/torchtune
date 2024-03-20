# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import json
import os
import runpy
import sys
from functools import partial
from pathlib import Path
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
# No support for large scale test yet for LoRA
torchtune.models.lora_small_test_model = partial(
    lora_llama2_small_test_ckpt,
    lora_attn_modules=test_lora_attn_modules,
    apply_lora_to_mlp=False,
    apply_lora_to_output=False,
)


class TestLoRAFinetuneRecipe:
    def _fetch_expected_loss_values(self) -> Dict[str, float]:
        return {
            "1|1|": 10.5074,
            "1|2|": 10.5614,
            "2|1|": 10.5205,
            "2|2|": 10.4918,
        }

    def fetch_checkpointer(self, ckpt):
        if ckpt == "small_test_ckpt_tune":
            return "FullModelTorchTuneCheckpointer"
        if ckpt == "small_test_ckpt_hf":
            return "FullModelHFCheckpointer"
        if ckpt == "small_test_ckpt_meta":
            return "FullModelMetaCheckpointer"

    @pytest.mark.parametrize("enable_fsdp", [False])
    def test_loss(self, capsys, tmpdir, enable_fsdp, monkeypatch):
        expected_loss_values = self._fetch_expected_loss_values()
        config_path = RECIPE_TESTS_DIR / "lora_finetune_test_config.yaml"
        recipe_name = (
            "lora_finetune_single_device"
            if not enable_fsdp
            else "lora_finetune_distributed"
        )

        for ckpt in [
            "small_test_ckpt_hf",
            "small_test_ckpt_meta",
            "small_test_ckpt_tune",
        ]:
            ckpt_path = Path(fetch_ckpt_model_path(ckpt))
            ckpt_dir = ckpt_path.parent
            # TODO (rohan-varma): setting CUDA_VISIBLE_DEVICES to ignore all GPUs
            # on machine to simulate current CI environment that does not have GPUs.
            # Will consolidate as part of addressing https://github.com/pytorch/torchtune/issues/473
            monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
            cmd = f"""
            tune {recipe_name}
                --config {config_path} \
                output_dir={tmpdir} \
                enable_fsdp={enable_fsdp} \
                model=torchtune.models.lora_small_test_model \
                checkpointer._component_=torchtune.utils.{self.fetch_checkpointer(ckpt)}
                checkpointer.checkpoint_dir='{ckpt_dir}' \
                checkpointer.checkpoint_files=[{ckpt_path}]\
                checkpointer.output_dir={tmpdir} \
                checkpointer.model_type=LLAMA2 \
                model.lora_rank=8 \
                model.lora_alpha=16 \
                model.apply_lora_to_mlp=False \
                model.apply_lora_to_output=False \
            """.split()

            # Have to attach this after so it parses correctly
            cmd += [
                'model.lora_attn_modules=["q_proj", "k_proj", "v_proj", "output_proj"]'
            ]
            monkeypatch.setattr(sys, "argv", cmd)
            with pytest.raises(SystemExit):
                with (
                    single_box_init(init_pg=False)
                    if enable_fsdp
                    else contextlib.nullcontext()
                ):
                    runpy.run_path(TUNE_PATH, run_name="__main__")

            loss_values = fetch_loss_values(capsys.readouterr().err)
            validate_loss_values(loss_values, expected_loss_values)

    @pytest.mark.parametrize("enable_fsdp", [False])
    def test_training_state_on_resume(self, enable_fsdp, capsys, tmpdir, monkeypatch):
        """Test whether the recipe state is correctly updated on resume. Since this
        is model agnostic, we should run this on the small model only. The test
        consists of three stages:
            - Train a model for 4 epochs
            - Resume training after epoch 3
            - Make sure final loss matches the expected value of a model successfully resumed from a ckpt
        """

        config_path = RECIPE_TESTS_DIR / "lora_finetune_test_config.yaml"
        recipe_name = (
            "lora_finetune_single_device"
            if not enable_fsdp
            else "lora_finetune_distributed"
        )

        model_ckpt = "small_test_ckpt_hf"
        expected_loss_values = self._fetch_expected_loss_values()

        ckpt_path = Path(fetch_ckpt_model_path(model_ckpt))
        ckpt_dir = ckpt_path.parent

        # config file needed for model conversion. Since this is a really small json
        # this can be written within the test instead of downloading from s3.
        # We need two copies one for initial read and one for resume
        config = {
            "hidden_size": 256,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
        }
        config_file = Path.joinpath(Path(ckpt_dir), "config.json")
        with config_file.open("w") as f:
            json.dump(config, f)

        # Have to attach this after so it parses correctly
        lora_cmd = (
            'model.lora_attn_modules=["q_proj", "k_proj", "v_proj", "output_proj"]'
        )

        # Train
        cmd_1 = f"""
        tune {recipe_name}
            --config {config_path} \
            output_dir={tmpdir} \
            enable_fsdp={enable_fsdp}
            model=torchtune.models.lora_small_test_model \
            checkpointer._component_=torchtune.utils.FullModelHFCheckpointer \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[{ckpt_path}]\
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type=LLAMA2 \
            model.lora_rank=8 \
            model.lora_alpha=16 \
            model.apply_lora_to_mlp=True \
            model.apply_lora_to_output=False \
            epochs=2 \
        """.split()
        cmd_1 = cmd_1 + [lora_cmd]
        monkeypatch.setattr(sys, "argv", cmd_1)
        with pytest.raises(SystemExit):
            with (
                single_box_init(init_pg=False)
                if enable_fsdp
                else contextlib.nullcontext()
            ):
                runpy.run_path(TUNE_PATH, run_name="__main__")

        # Clear stdout
        capsys.readouterr()

        config_file = Path.joinpath(Path(tmpdir), "config.json")
        with config_file.open("w") as f:
            json.dump(config, f)

        # Resume training
        cmd_2 = f"""
        tune {recipe_name}
            --config {config_path} \
            output_dir={tmpdir} \
            enable_fsdp={enable_fsdp} \
            model=torchtune.models.lora_small_test_model \
            checkpointer._component_=torchtune.utils.FullModelHFCheckpointer \
            checkpointer.checkpoint_dir={tmpdir} \
            checkpointer.checkpoint_files=[{ckpt_path}]\
            checkpointer.adapter_checkpoint={os.path.join(tmpdir, "adapter_0.pt")}
            checkpointer.recipe_checkpoint={os.path.join(tmpdir, "recipe_state.pt")}
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type=LLAMA2 \
            model.lora_rank=8 \
            model.lora_alpha=16 \
            model.apply_lora_to_mlp=True \
            model.apply_lora_to_output=False \
            epochs=2 \
            resume_from_checkpoint=True \
        """.split()
        cmd_2 = cmd_2 + [lora_cmd]
        monkeypatch.setattr(sys, "argv", cmd_2)
        with pytest.raises(SystemExit):
            with (
                single_box_init(init_pg=False)
                if enable_fsdp
                else contextlib.nullcontext()
            ):
                runpy.run_path(TUNE_PATH, run_name="__main__")

        expected_loss_values = {
            "2|1|": 10.5205,
            "2|2|": 10.4918,
        }

        loss_values = fetch_loss_values(capsys.readouterr().err)
        validate_loss_values(loss_values, expected_loss_values)

    @pytest.mark.parametrize("enable_fsdp", [False])
    def test_save_and_load_merged_weights(self, tmpdir, enable_fsdp, monkeypatch):
        ckpt = "small_test_ckpt_tune"
        config_path = RECIPE_TESTS_DIR / "lora_finetune_test_config.yaml"
        recipe_name = (
            "lora_finetune_single_device"
            if not enable_fsdp
            else "lora_finetune_distributed"
        )

        ckpt_path = Path(fetch_ckpt_model_path(ckpt))
        ckpt_dir = ckpt_path.parent
        # TODO (rohan-varma): setting CUDA_VISIBLE_DEVICES to ignore all GPUs
        # on machine to simulate current CI environment that does not have GPUs.
        # Will consolidate as part of addressing https://github.com/pytorch/torchtune/issues/473
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
        cmd = f"""
        tune {recipe_name}
            --config {config_path} \
            output_dir={tmpdir} \
            enable_fsdp={enable_fsdp} \
            model=torchtune.models.lora_small_test_model \
            checkpointer._component_=torchtune.utils.FullModelTorchTuneCheckpointer
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[{ckpt_path}]\
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type=LLAMA2 \
            model.lora_rank=8 \
            model.lora_alpha=16 \
            model.apply_lora_to_mlp=True \
            model.apply_lora_to_output=False \
        """.split()

        # Have to attach this after so it parses correctly
        cmd += ['model.lora_attn_modules=["q_proj", "k_proj", "v_proj", "output_proj"]']
        monkeypatch.setattr(sys, "argv", cmd)
        with pytest.raises(SystemExit):
            with (
                single_box_init(init_pg=False)
                if enable_fsdp
                else contextlib.nullcontext()
            ):
                runpy.run_path(TUNE_PATH, run_name="__main__")

        # Next load both the merged weights in a Llama2 base model
        # and the base model weights + trained adapter weights in the LoRA Llama 2 model
        # The results of calling forward on dummy inputs should be the same.
        inputs = torch.randint(low=0, high=32_000, size=(2, 100))

        # Build LoRA model for loading base + adapter weights separately
        lora_model = lora_llama2_small_test_ckpt(
            lora_attn_modules=["q_proj", "k_proj", "v_proj", "output_proj"],
            apply_lora_to_mlp=True,
            apply_lora_to_output=False,
            lora_rank=8,
            lora_alpha=16,
        )
        # Build base llama2 model for loading merged weights
        llama2_model = llama2_small_test_ckpt()

        # Load base model and trained adapter weights into LoRA model and call fwd
        with open(f"{tmpdir}/adapter_1.pt", "rb") as f:
            lora_sd = torch.load(f, weights_only=True)
        with open(ckpt_path, "rb") as f:
            base_model_sd = torch.load(f, weights_only=True)
        lora_model.load_state_dict(lora_sd, strict=False)
        lora_model.load_state_dict(base_model_sd, strict=False)
        baseline_out = lora_model(inputs)

        # Load merged final ckpt directly into llama2 and call fwd
        with open(f"{tmpdir}/torchtune_model_1.pt", "rb") as f:
            sd = torch.load(f, weights_only=True)
        llama2_model.load_state_dict(sd)
        merged_ckpt_out = llama2_model(inputs)

        torch.testing.assert_close(baseline_out, merged_ckpt_out, rtol=1e-5, atol=1e-5)
