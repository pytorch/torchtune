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
    dummy_alpaca_dataset_config,
    llama2_test_config,
    lora_llama2_test_config,
    write_hf_ckpt_config,
)
from tests.test_utils import (
    CKPT_MODEL_PATHS,
    gen_log_file_name,
    get_loss_values_from_metric_logger,
    gpu_test,
)
from torch.distributed.checkpoint.state_dict import StateDictOptions
from torchtune import config


class TestLoRAFinetuneDistributedRecipe:
    def _get_test_config_overrides(self):
        return [
            "batch_size=4",
            "enable_activation_checkpointing=False",
            "tokenizer.path=/tmp/test-artifacts/tokenizer.model",
            "dataset.train_on_input=False",
            "seed=9",
            "epochs=2",
            "dtype=fp32",
            "max_steps_per_epoch=2",
            "optimizer.lr=2e-5",
            "log_every_n_steps=1",
            "gradient_accumulation_steps=1",
        ] + dummy_alpaca_dataset_config()

    def _fetch_expected_loss_values(self):
        # These values have been validated against single device recipe test via
        # https://gist.github.com/ebsmothers/f1c3db7c66655a23a91e0290360960c4
        return [10.5136, 10.4857, 10.5292, 10.5345]

    @pytest.mark.integration_test
    @gpu_test(gpu_count=2)
    @pytest.mark.skipif(
        not hasattr(StateDictOptions, "broadcast_from_rank0"),
        reason="need latest pytorch nightly",
    )
    def test_loss(self, tmpdir, monkeypatch):
        ckpt = "small_test_ckpt_tune"
        ckpt_path = Path(CKPT_MODEL_PATHS[ckpt])
        ckpt_dir = ckpt_path.parent
        log_file = gen_log_file_name(tmpdir)
        cmd = f"""
        tune run --nnodes 1 --nproc_per_node 2 lora_finetune_distributed
            --config llama2/7B_lora \
            output_dir={tmpdir} \
            checkpointer=torchtune.utils.FullModelTorchTuneCheckpointer \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[{ckpt_path}]\
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type=LLAMA2 \
            metric_logger.filename={log_file} \
        """.split()

        model_config = lora_llama2_test_config(
            lora_attn_modules=["q_proj", "k_proj", "v_proj", "output_proj"],
            apply_lora_to_mlp=False,
            apply_lora_to_output=False,
            lora_rank=8,
            lora_alpha=16,
        )

        cmd = cmd + self._get_test_config_overrides() + model_config
        monkeypatch.setattr(sys, "argv", cmd)
        runpy.run_path(TUNE_PATH, run_name="__main__")
        loss_values = get_loss_values_from_metric_logger(log_file)
        expected_loss_values = self._fetch_expected_loss_values()
        torch.testing.assert_close(
            loss_values, expected_loss_values, rtol=1e-5, atol=1e-5
        )

    @pytest.mark.integration_test
    @gpu_test(gpu_count=2)
    @pytest.mark.skipif(True, reason="resolve FSDP2 optimizer state dict and enable")
    def test_training_state_on_resume(self, tmpdir, monkeypatch):
        """Test whether the recipe state is correctly updated on resume. Since this
        is model agnostic, we should run this on the small model only. The test
        consists of three stages:
            - Train a model for 2 epochs
            - Resume training after epoch 1
            - Make sure final loss matches the expected value of a model successfully resumed from a ckpt
        """

        ckpt = "small_test_ckpt_hf"
        expected_loss_values = self._fetch_expected_loss_values()

        ckpt_path = Path(CKPT_MODEL_PATHS[ckpt])
        ckpt_dir = ckpt_path.parent
        log_file = gen_log_file_name(tmpdir)

        # Config file needed for model conversion.
        # Create a second copy for training resume
        write_hf_ckpt_config(ckpt_dir)
        write_hf_ckpt_config(tmpdir)

        # Train for two epochs
        cmd_1 = f"""
        tune run --nnodes 1 --nproc_per_node 2 lora_finetune_distributed \
            --config llama2/7B_lora \
            output_dir={tmpdir} \
            checkpointer=torchtune.utils.FullModelHFCheckpointer \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[{ckpt_path}]\
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type=LLAMA2 \
        """.split()

        model_config = lora_llama2_test_config(
            lora_attn_modules=["q_proj", "k_proj", "v_proj", "output_proj"],
            apply_lora_to_mlp=True,
            apply_lora_to_output=False,
            lora_rank=8,
            lora_alpha=16,
        )

        cmd_1 = cmd_1 + self._get_test_config_overrides() + model_config
        monkeypatch.setattr(sys, "argv", cmd_1)
        runpy.run_path(TUNE_PATH, run_name="__main__")

        # Resume training
        cmd_2 = f"""
        tune run --nnodes 1 --nproc_per_node 2 lora_finetune_distributed \
            --config llama2/7B_lora \
            output_dir={tmpdir} \
            checkpointer=torchtune.utils.FullModelHFCheckpointer \
            checkpointer.checkpoint_dir={tmpdir} \
            checkpointer.checkpoint_files=[{ckpt_path}]\
            checkpointer.adapter_checkpoint={os.path.join(tmpdir, "adapter_0.pt")}
            checkpointer.recipe_checkpoint={os.path.join(tmpdir, "recipe_state.pt")}
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type=LLAMA2 \
            resume_from_checkpoint=True \
            metric_logger.filename={log_file} \
        """.split()

        cmd_2 = cmd_2 + self._get_test_config_overrides() + model_config
        monkeypatch.setattr(sys, "argv", cmd_2)
        runpy.run_path(TUNE_PATH, run_name="__main__")

        expected_loss_values = self._fetch_expected_loss_values()[2:]

        loss_values = get_loss_values_from_metric_logger(log_file)
        torch.testing.assert_close(
            loss_values, expected_loss_values, rtol=1e-5, atol=1e-5
        )

    @pytest.mark.integration_test
    @gpu_test(gpu_count=2)
    @pytest.mark.skipif(
        not hasattr(StateDictOptions, "broadcast_from_rank0"),
        reason="need latest pytorch nightly",
    )
    def test_save_and_load_merged_weights(self, tmpdir, monkeypatch):
        ckpt = "small_test_ckpt_tune"

        ckpt_path = Path(CKPT_MODEL_PATHS[ckpt])
        ckpt_dir = ckpt_path.parent
        cmd = f"""
        tune run --nnodes 1 --nproc_per_node 2 lora_finetune_distributed \
            --config llama2/7B_lora \
            output_dir={tmpdir} \
            model=torchtune.models.lora_small_test_model \
            checkpointer=torchtune.utils.FullModelTorchTuneCheckpointer
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[{ckpt_path}]\
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type=LLAMA2 \
        """.split()

        model_config = lora_llama2_test_config(
            lora_attn_modules=["q_proj", "k_proj", "v_proj", "output_proj"],
            apply_lora_to_mlp=True,
            apply_lora_to_output=False,
            lora_rank=8,
            lora_alpha=16,
        )

        cmd = cmd + self._get_test_config_overrides() + model_config
        monkeypatch.setattr(sys, "argv", cmd)
        runpy.run_path(TUNE_PATH, run_name="__main__")

        # Next load both the merged weights in a Llama2 base model
        # and the base model weights + trained adapter weights in the LoRA Llama 2 model
        # The results of calling forward on dummy inputs should be the same.
        inputs = torch.randint(low=0, high=32_000, size=(2, 100))

        # Build LoRA model for loading base + adapter weights separately
        lora_model = config.instantiate(OmegaConf.from_dotlist(model_config).model)

        # Build base llama2 model for loading merged weights
        base_llama2_config = llama2_test_config()
        llama2_model = config.instantiate(
            OmegaConf.from_dotlist(base_llama2_config).model
        )

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
