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
from tests.recipes.utils import dummy_alpaca_dataset_config, MODEL_TEST_CONFIGS
from tests.test_utils import (
    CKPT_MODEL_PATHS,
    gen_log_file_name,
    get_loss_values_from_metric_logger,
    gpu_test,
    TOKENIZER_PATHS,
)
from torchtune import config

from torchtune.training.checkpointing._utils import (
    ADAPTER_MODEL_FNAME,
    get_largest_iter_folder,
    RECIPE_STATE_DIRNAME,
    safe_torch_load,
    SHARD_FNAME,
)


class TestQATLoRAFinetuneDistributedRecipe:
    def _get_test_config_overrides(self):
        return [
            "dataset.train_on_input=False",
            "seed=9",
            "epochs=2",
            "dtype=fp32",
            "max_steps_per_epoch=2",
            "optimizer.lr=2e-5",
            "log_every_n_steps=1",
            "compile=False",
        ] + dummy_alpaca_dataset_config()

    def _fetch_expected_loss_values(self, model_ckpt):
        loss_values_map = {
            "llama3_hf_138m": [
                # TODO
                11.977421760559082,
                11.979637145996094,
                11.948746681213379,
                11.912514686584473,
            ],
        }
        return loss_values_map[model_ckpt]

    @pytest.mark.integration_test
    @gpu_test(gpu_count=4)
    @pytest.mark.parametrize(
        "model_ckpt, micro_batch_size, gradient_accumulation_steps, should_compile",
        [("llama3_hf_138m", 4, 1, True), ("llama3_hf_138m", 1, 4, False)],
    )
    def test_loss(
        self,
        micro_batch_size,
        gradient_accumulation_steps,
        should_compile,
        tmpdir,
        monkeypatch,
        model_ckpt,
    ):
        ckpt_dir = Path(CKPT_MODEL_PATHS[model_ckpt])
        tokenizer_path = Path(TOKENIZER_PATHS[model_ckpt])
        log_file = gen_log_file_name(tmpdir)
        cmd = f"""
        tune run --nnodes 1 --nproc_per_node 4 qat_lora_finetune_distributed
            --config llama3/8B_qat_lora \
            batch_size={micro_batch_size} \
            gradient_accumulation_steps={gradient_accumulation_steps} \
            output_dir={tmpdir} \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[model.safetensors]\
            checkpointer.output_dir={tmpdir} \
            metric_logger.filename={log_file} \
            tokenizer.path='{tokenizer_path}' \
            tokenizer.prompt_template=null \
            compile={should_compile} \
            enable_activation_checkpointing=False \
            enable_activation_offloading=False \
            quantizer.groupsize=32 \
        """.split()

        model_config = MODEL_TEST_CONFIGS[model_ckpt + "_lora"]

        cmd = cmd + self._get_test_config_overrides() + model_config
        monkeypatch.setattr(sys, "argv", cmd)
        runpy.run_path(TUNE_PATH, run_name="__main__")
        loss_values = get_loss_values_from_metric_logger(log_file)

        expected_loss_values = self._fetch_expected_loss_values(model_ckpt)
        torch.testing.assert_close(
            loss_values, expected_loss_values, rtol=1e-5, atol=1e-5
        )

    @pytest.mark.integration_test
    @gpu_test(gpu_count=4)
    @pytest.mark.parametrize(
        "config, model_ckpt, save_adapter_weights_only",
        [
            ("llama3/8B_qat_lora", "llama3_hf_138m", False),
        ],
    )
    def test_training_state_on_resume(
        self,
        config,
        model_ckpt,
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
        expected_loss_values = self._fetch_expected_loss_values(model_ckpt)

        ckpt_dir = Path(CKPT_MODEL_PATHS[model_ckpt])
        tokenizer_path = Path(TOKENIZER_PATHS[model_ckpt])
        log_file = gen_log_file_name(tmpdir)

        # Train for two epochs
        cmd_1 = f"""
        tune run --nnodes 1 --nproc_per_node 4 qat_lora_finetune_distributed \
            --config {config} \
            batch_size=4 \
            gradient_accumulation_steps=1 \
            output_dir={tmpdir} \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[model.safetensors]\
            checkpointer.output_dir={tmpdir} \
            tokenizer.path='{tokenizer_path}' \
            tokenizer.prompt_template=null \
            save_adapter_weights_only={save_adapter_weights_only} \
            enable_activation_checkpointing=True \
            enable_activation_offloading=True \
            quantizer.groupsize=32 \
        """.split()

        model_config = MODEL_TEST_CONFIGS[model_ckpt + "_lora"]

        cmd_1 = cmd_1 + self._get_test_config_overrides() + model_config
        monkeypatch.setattr(sys, "argv", cmd_1)
        runpy.run_path(TUNE_PATH, run_name="__main__")

        # Resume training
        epoch_folder = get_largest_iter_folder(tmpdir)
        epoch_folder_minus_one = f"epoch_{int(epoch_folder.split('_')[-1]) - 1}"
        cmd_2 = f"""
        tune run --nnodes 1 --nproc_per_node 4 qat_lora_finetune_distributed \
            --config {config} \
            batch_size=4 \
            gradient_accumulation_steps=1 \
            output_dir={tmpdir} \
            checkpointer.checkpoint_dir={ckpt_dir} \
            checkpointer.checkpoint_files=[model.safetensors]\
            checkpointer.adapter_checkpoint={os.path.join(epoch_folder_minus_one, f"{ADAPTER_MODEL_FNAME}.pt")} \
            checkpointer.recipe_checkpoint={os.path.join(RECIPE_STATE_DIRNAME, "recipe_state.pt")} \
            checkpointer.output_dir={tmpdir} \
            tokenizer.path='{tokenizer_path}' \
            tokenizer.prompt_template=null \
            resume_from_checkpoint=True \
            metric_logger.filename={log_file} \
            enable_activation_checkpointing=True \
            enable_activation_offloading=True \
            quantizer.groupsize=32 \
        """.split()

        cmd_2 = cmd_2 + self._get_test_config_overrides() + model_config
        monkeypatch.setattr(sys, "argv", cmd_2)
        runpy.run_path(TUNE_PATH, run_name="__main__")

        expected_loss_values = self._fetch_expected_loss_values(model_ckpt)[2:]

        loss_values = get_loss_values_from_metric_logger(log_file)
        torch.testing.assert_close(
            loss_values, expected_loss_values, rtol=1e-5, atol=1e-5
        )

    @pytest.mark.integration_test
    @gpu_test(gpu_count=2)
    @pytest.mark.parametrize(
        "config, model_ckpt, save_adapter_weights_only",
        [
            ("llama3/8B_qat_lora", "llama3_hf_138m", False),
        ],
    )
    def test_training_state_on_resume_with_async_checkpointing(
        self,
        config,
        model_ckpt,
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
        expected_loss_values = self._fetch_expected_loss_values(model_ckpt)

        ckpt_dir = Path(CKPT_MODEL_PATHS[model_ckpt])
        tokenizer_path = Path(TOKENIZER_PATHS[model_ckpt])
        log_file = gen_log_file_name(tmpdir)

        # Train for two epochs
        cmd_1 = f"""
        tune run --nnodes 1 --nproc_per_node 2 qat_lora_finetune_distributed \
            --config {config} \
            batch_size=8 \
            gradient_accumulation_steps=1 \
            output_dir={tmpdir} \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[model.safetensors]\
            checkpointer.output_dir={tmpdir} \
            tokenizer.path='{tokenizer_path}' \
            tokenizer.prompt_template=null \
            save_adapter_weights_only={save_adapter_weights_only} \
            enable_activation_checkpointing=True \
            enable_activation_offloading=True \
            enable_async_checkpointing=True \
            quantizer.groupsize=32 \
        """.split()

        model_config = MODEL_TEST_CONFIGS[model_ckpt + "_lora"]

        cmd_1 = cmd_1 + self._get_test_config_overrides() + model_config
        monkeypatch.setattr(sys, "argv", cmd_1)
        runpy.run_path(TUNE_PATH, run_name="__main__")

        # Resume training
        epoch_folder = get_largest_iter_folder(tmpdir)
        epoch_folder_minus_one = f"epoch_{int(epoch_folder.split('_')[-1]) - 1}"
        cmd_2 = f"""
        tune run --nnodes 1 --nproc_per_node 2 qat_lora_finetune_distributed \
            --config {config} \
            batch_size=8 \
            gradient_accumulation_steps=1 \
            output_dir={tmpdir} \
            checkpointer.checkpoint_dir={ckpt_dir} \
            checkpointer.checkpoint_files=[model.safetensors]\
            checkpointer.adapter_checkpoint={os.path.join(epoch_folder_minus_one, f"{ADAPTER_MODEL_FNAME}.pt")} \
            checkpointer.recipe_checkpoint={os.path.join(RECIPE_STATE_DIRNAME, "recipe_state.pt")} \
            checkpointer.output_dir={tmpdir} \
            tokenizer.path='{tokenizer_path}' \
            tokenizer.prompt_template=null \
            resume_from_checkpoint=True \
            metric_logger.filename={log_file} \
            enable_activation_checkpointing=True \
            enable_activation_offloading=True \
            enable_async_checkpointing=True \
            quantizer.groupsize=32 \
        """.split()

        cmd_2 = cmd_2 + self._get_test_config_overrides() + model_config
        monkeypatch.setattr(sys, "argv", cmd_2)
        runpy.run_path(TUNE_PATH, run_name="__main__")

        expected_loss_values = self._fetch_expected_loss_values(model_ckpt)[2:]

        loss_values = get_loss_values_from_metric_logger(log_file)
        torch.testing.assert_close(
            loss_values, expected_loss_values, rtol=1e-4, atol=1e-4
        )

    @pytest.mark.integration_test
    @pytest.mark.parametrize(
        "recipe_config, model_ckpt",
        [
            ("llama3/8B_qat_lora", "llama3_hf_138m"),
        ],
    )
    @gpu_test(gpu_count=4)
    def test_save_and_load_merged_weights(
        self, recipe_config, model_ckpt, tmpdir, monkeypatch
    ):
        ckpt_dir = Path(CKPT_MODEL_PATHS[model_ckpt])
        tokenizer_path = Path(TOKENIZER_PATHS[model_ckpt])
        cmd = f"""
        tune run --nnodes 1 --nproc_per_node 4 qat_lora_finetune_distributed \
            --config {recipe_config} \
            batch_size=4 \
            gradient_accumulation_steps=1 \
            output_dir={tmpdir} \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[model.safetensors]\
            checkpointer.output_dir={tmpdir} \
            tokenizer.path='{tokenizer_path}' \
            tokenizer.prompt_template=null \
            enable_activation_checkpointing=True \
            enable_activation_offloading=True \
            quantizer.groupsize=32 \
        """.split()

        model_config = MODEL_TEST_CONFIGS[model_ckpt + "_lora"]

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
        base_config = MODEL_TEST_CONFIGS[model_ckpt]
        model = config.instantiate(OmegaConf.from_dotlist(base_config).model)

        # Load base model and trained adapter weights into LoRA model and call fwd
        epoch_folder = get_largest_iter_folder(tmpdir)
        adpt_path = os.path.join(tmpdir, epoch_folder, f"{ADAPTER_MODEL_FNAME}.pt")
        lora_sd = safe_torch_load(adpt_path, weights_only=True)

        # Load base model from HF checkpoint
        base_model_path = os.path.join(ckpt_dir, "model.safetensors")
        base_model_sd = safe_torch_load(base_model_path, weights_only=True)

        lora_model.load_state_dict(lora_sd, strict=False)
        lora_model.load_state_dict(base_model_sd, strict=False)
        baseline_out = lora_model(inputs)

        # Load merged final ckpt directly into model and call fwd
        suffix = ".safetensors"
        model_ckpt_fname = (
            SHARD_FNAME.format(cpt_idx="1".zfill(5), num_shards="1".zfill(5)) + suffix
        )
        model_path = os.path.join(tmpdir, epoch_folder, model_ckpt_fname)
        sd = safe_torch_load(model_path, weights_only=True)

        model.load_state_dict(sd)
        merged_ckpt_out = model(inputs)

        torch.testing.assert_close(baseline_out, merged_ckpt_out, rtol=1e-5, atol=1e-5)
