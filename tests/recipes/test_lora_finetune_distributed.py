# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import runpy
import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from tests.common import TUNE_PATH
from tests.recipes.utils import (
    CKPT_COMPONENT_MAP,
    dummy_alpaca_dataset_config,
    MODEL_TEST_CONFIGS,
    write_hf_ckpt_config,
)
from tests.test_utils import (
    CKPT_MODEL_PATHS,
    gen_log_file_name,
    get_loss_values_from_metric_logger,
    gpu_test,
    TOKENIZER_PATHS,
)
from torchtune import config


class TestLoRAFinetuneDistributedRecipe:
    def _get_test_config_overrides(self):
        return [
            "batch_size=4",
            "dataset.train_on_input=False",
            "seed=9",
            "epochs=2",
            "dtype=fp32",
            "max_steps_per_epoch=2",
            "optimizer.lr=2e-5",
            "log_every_n_steps=1",
            "gradient_accumulation_steps=1",
            "compile=False",
        ] + dummy_alpaca_dataset_config()

    def _fetch_expected_loss_values(self, model_type):
        # These values have been validated against single device recipe test via
        # https://gist.github.com/ebsmothers/f1c3db7c66655a23a91e0290360960c4
        loss_values_map = {
            "llama2": [10.5136, 10.4856, 10.5292, 10.5345],
            "llama3": [11.9325, 11.9325, 11.9325, 11.9369],
        }
        return loss_values_map[model_type]

    @pytest.mark.integration_test
    @gpu_test(gpu_count=2)
    @pytest.mark.parametrize(
        "reshard_after_forward",
        [
            True,
            False,
        ],
    )
    def test_loss(self, reshard_after_forward, tmpdir, monkeypatch):
        ckpt = "llama2_tune"
        ckpt_path = Path(CKPT_MODEL_PATHS[ckpt])
        ckpt_dir = ckpt_path.parent
        log_file = gen_log_file_name(tmpdir)
        cmd = f"""
        tune run --nnodes 1 --nproc_per_node 2 lora_finetune_distributed
            --config llama2/7B_lora \
            output_dir={tmpdir} \
            checkpointer=torchtune.training.FullModelTorchTuneCheckpointer \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[{ckpt_path}]\
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type=LLAMA2 \
            metric_logger.filename={log_file} \
            tokenizer.path=/tmp/test-artifacts/tokenizer.model \
            tokenizer.prompt_template=null \
            reshard_after_forward={reshard_after_forward} \
            enable_activation_checkpointing=False \
        """.split()

        model_config = MODEL_TEST_CONFIGS["llama2_lora"]

        cmd = cmd + self._get_test_config_overrides() + model_config
        monkeypatch.setattr(sys, "argv", cmd)
        runpy.run_path(TUNE_PATH, run_name="__main__")
        loss_values = get_loss_values_from_metric_logger(log_file)
        expected_loss_values = self._fetch_expected_loss_values("llama2")
        torch.testing.assert_close(
            loss_values, expected_loss_values, rtol=1e-5, atol=1e-5
        )

    @pytest.mark.integration_test
    @gpu_test(gpu_count=2)
    @pytest.mark.parametrize(
        "config, model_type, ckpt_type, save_adapter_weights_only",
        [
            ("llama2/7B_lora", "llama2", "hf", False),
            ("llama3/8B_lora", "llama3", "tune", False),
            ("llama2/7B_lora", "llama2", "hf", True),
        ],
    )
    def test_training_state_on_resume(
        self,
        config,
        model_type,
        ckpt_type,
        tmpdir,
        monkeypatch,
        save_adapter_weights_only,
    ):
        """Test whether the recipe state is correctly updated on resume. Since this
        is model agnostic, we should run this on the small model only. The test
        consists of three stages:
            - Train a model for 2 epochs
            - Resume training after epoch 1
            - Make sure final loss matches the expected value of a model successfully resumed from a ckpt
        """
        ckpt_component = CKPT_COMPONENT_MAP[ckpt_type]
        ckpt = model_type + "_" + ckpt_type
        expected_loss_values = self._fetch_expected_loss_values(model_type)

        ckpt_path = Path(CKPT_MODEL_PATHS[ckpt])
        tokenizer_path = Path(TOKENIZER_PATHS[model_type])
        ckpt_dir = ckpt_path.parent
        log_file = gen_log_file_name(tmpdir)

        # Config file needed for model conversion.
        # Create a second copy for training resume
        write_hf_ckpt_config(ckpt_dir)
        write_hf_ckpt_config(tmpdir)

        # Train for two epochs
        cmd_1 = f"""
        tune run --nnodes 1 --nproc_per_node 2 lora_finetune_distributed \
            --config {config} \
            output_dir={tmpdir} \
            checkpointer._component_={ckpt_component} \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[{ckpt_path}]\
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type={model_type.upper()} \
            tokenizer.path='{tokenizer_path}' \
            tokenizer.prompt_template=null \
            save_adapter_weights_only={save_adapter_weights_only} \
            enable_activation_checkpointing=True \
        """.split()

        model_config = MODEL_TEST_CONFIGS[model_type + "_lora"]

        cmd_1 = cmd_1 + self._get_test_config_overrides() + model_config
        monkeypatch.setattr(sys, "argv", cmd_1)
        runpy.run_path(TUNE_PATH, run_name="__main__")

        # Resume training
        cmd_2 = f"""
        tune run --nnodes 1 --nproc_per_node 2 lora_finetune_distributed \
            --config {config} \
            output_dir={tmpdir} \
            checkpointer._component_={ckpt_component} \
            checkpointer.checkpoint_dir={tmpdir} \
            checkpointer.checkpoint_files=[{ckpt_path}]\
            checkpointer.adapter_checkpoint={os.path.join(tmpdir, "adapter_0.pt")}
            checkpointer.recipe_checkpoint={os.path.join(tmpdir, "recipe_state.pt")}
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type={model_type.upper()} \
            tokenizer.path='{tokenizer_path}' \
            tokenizer.prompt_template=null \
            resume_from_checkpoint=True \
            metric_logger.filename={log_file} \
            enable_activation_checkpointing=True \
        """.split()

        cmd_2 = cmd_2 + self._get_test_config_overrides() + model_config
        monkeypatch.setattr(sys, "argv", cmd_2)
        runpy.run_path(TUNE_PATH, run_name="__main__")

        expected_loss_values = self._fetch_expected_loss_values(model_type)[2:]

        loss_values = get_loss_values_from_metric_logger(log_file)
        torch.testing.assert_close(
            loss_values, expected_loss_values, rtol=1e-5, atol=1e-5
        )

    @pytest.mark.integration_test
    @pytest.mark.parametrize(
        "recipe_config, model_type, ckpt_type",
        [
            ("llama2/7B_lora", "llama2", "tune"),
            ("llama3/8B_lora", "llama3", "tune"),
        ],
    )
    @gpu_test(gpu_count=2)
    def test_save_and_load_merged_weights(
        self, recipe_config, model_type, ckpt_type, tmpdir, monkeypatch
    ):
        ckpt_component = CKPT_COMPONENT_MAP[ckpt_type]
        ckpt = model_type + "_" + ckpt_type
        ckpt_path = Path(CKPT_MODEL_PATHS[ckpt])
        tokenizer_path = Path(TOKENIZER_PATHS[model_type])
        ckpt_dir = ckpt_path.parent
        cmd = f"""
        tune run --nnodes 1 --nproc_per_node 2 lora_finetune_distributed \
            --config {recipe_config} \
            output_dir={tmpdir} \
            model=torchtune.models.lora_small_test_model \
            checkpointer._component_={ckpt_component} \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[{ckpt_path}]\
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type={model_type.upper()} \
            tokenizer.path='{tokenizer_path}' \
            tokenizer.prompt_template=null \
            enable_activation_checkpointing=True \
        """.split()

        model_config = MODEL_TEST_CONFIGS[model_type + "_lora"]

        cmd = cmd + self._get_test_config_overrides() + model_config
        monkeypatch.setattr(sys, "argv", cmd)
        runpy.run_path(TUNE_PATH, run_name="__main__")

        # Next load both the merged weights in a base model
        # and the base model weights + trained adapter weights in the LoRA model
        # The results of calling forward on dummy inputs should be the same.
        inputs = torch.randint(low=0, high=32_000, size=(2, 100))

        # Build LoRA model for loading base + adapter weights separately
        lora_model = config.instantiate(OmegaConf.from_dotlist(model_config).model)

        # Build base model for loading merged weights
        base_config = MODEL_TEST_CONFIGS[model_type]
        model = config.instantiate(OmegaConf.from_dotlist(base_config).model)

        # Load base model and trained adapter weights into LoRA model and call fwd
        with open(f"{tmpdir}/adapter_1.pt", "rb") as f:
            lora_sd = torch.load(f, weights_only=True)
        with open(ckpt_path, "rb") as f:
            base_model_sd = torch.load(f, weights_only=True)
        lora_model.load_state_dict(lora_sd, strict=False)
        lora_model.load_state_dict(base_model_sd, strict=False)
        baseline_out = lora_model(inputs)

        # Load merged final ckpt directly into model and call fwd
        with open(f"{tmpdir}/torchtune_model_1.pt", "rb") as f:
            sd = torch.load(f, weights_only=True)
        model.load_state_dict(sd)
        merged_ckpt_out = model(inputs)

        torch.testing.assert_close(baseline_out, merged_ckpt_out, rtol=1e-5, atol=1e-5)
