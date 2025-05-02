# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import runpy
import sys
from pathlib import Path

import pytest
import torch
from tests.common import TUNE_PATH
from tests.recipes.utils import (
    dummy_stack_exchange_dataset_config,
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


class TestFullDPODistributedRecipe:
    def _get_test_config_overrides(self, dtype_str: str = "fp32", epochs: int = 2):
        return [
            "batch_size=1",
            "device=cuda",
            "enable_activation_checkpointing=True",
            "enable_activation_offloading=True",
            f"dtype={dtype_str}",
            "dataset.train_on_input=False",
            "seed=9",
            f"epochs={epochs}",
            "max_steps_per_epoch=2",
            "optimizer=torch.optim.AdamW",
            "optimizer.lr=2e-6",
            "log_every_n_steps=1",
            "gradient_accumulation_steps=4",
            "clip_grad_norm=100",
            "tokenizer.max_seq_len=256",
        ] + dummy_stack_exchange_dataset_config()

    @pytest.mark.integration_test
    @gpu_test(gpu_count=2)
    def test_training_state_on_resume(self, tmpdir, monkeypatch):
        """Test whether the recipe state is correctly updated on resume. Since this
        is model agnostic, we should run this on the small model only. The test
        consists of three stages:
            - Train a model for 2 epochs
            - Resume training after epoch 1
            - Make sure final loss matches the expected value of a model successfully resumed from a ckpt
        Unlike `tests.recipes.test_lora_finetune_single_device`, this test does not use pre-computed loss
        values to benchmark against. This test just ensures the loss values are identical when resuming.
        """

        ckpt = "llama3_tune"
        ckpt_path = Path(CKPT_MODEL_PATHS[ckpt])
        ckpt_dir = ckpt_path.parent
        log_file = gen_log_file_name(tmpdir)
        tokenizer_path = Path(TOKENIZER_PATHS["llama3"])

        # Config file needed for model conversion.
        # Create a second copy for training resume
        write_hf_ckpt_config(ckpt_dir)
        write_hf_ckpt_config(tmpdir)

        # Train for two epochs
        cmd_1 = f"""
        tune run --nnodes 1 --nproc_per_node 2 full_dpo_distributed \
            --config llama3_1/8B_full_dpo \
            output_dir={tmpdir} \
            checkpointer=torchtune.training.FullModelTorchTuneCheckpointer \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[{ckpt_path}]\
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type=LLAMA3 \
            ref_checkpointer=torchtune.training.FullModelTorchTuneCheckpointer \
            ref_checkpointer.checkpoint_dir='{ckpt_dir}' \
            ref_checkpointer.checkpoint_files=[{ckpt_path}]\
            ref_checkpointer.output_dir={tmpdir} \
            ref_checkpointer.model_type=LLAMA3 \
            tokenizer.path='{tokenizer_path}' \
            tokenizer.prompt_template=null \
            tokenizer.max_seq_len=256 \
            metric_logger.filename={log_file} \
            enable_activation_checkpointing=True \
            enable_activation_offloading=True \
            batch_size=1 \
            gradient_accumulation_steps=4
        """.split()

        model_config = MODEL_TEST_CONFIGS["llama3"]

        cmd_1 = cmd_1 + self._get_test_config_overrides() + model_config
        monkeypatch.setattr(sys, "argv", cmd_1)
        # with pytest.raises(SystemExit, match=""):
        runpy.run_path(TUNE_PATH, run_name="__main__")

        expected_loss_values = get_loss_values_from_metric_logger(log_file)

        resumed_log_dir = (tmpdir / "resumed/").mkdir()
        resumed_log_file = gen_log_file_name(resumed_log_dir)

        # Resume training
        cmd_2 = f"""
        tune run --nnodes 1 --nproc_per_node 2 full_dpo_distributed \
            --config llama3_1/8B_full_dpo \
            output_dir={tmpdir} \
            checkpointer=torchtune.training.FullModelTorchTuneCheckpointer \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[{ckpt_path}]\
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type=LLAMA3 \
            ref_checkpointer=torchtune.training.FullModelTorchTuneCheckpointer \
            ref_checkpointer.checkpoint_dir='{ckpt_dir}' \
            ref_checkpointer.checkpoint_files=[{ckpt_path}]\
            ref_checkpointer.output_dir={tmpdir} \
            ref_checkpointer.model_type=LLAMA3 \
            resume_from_checkpoint=True \
            tokenizer.path='{tokenizer_path}' \
            tokenizer.prompt_template=null \
            tokenizer.max_seq_len=256 \
            metric_logger.filename={resumed_log_file} \
            enable_activation_checkpointing=True \
            enable_activation_offloading=True \
            batch_size=1 \
            gradient_accumulation_steps=4
        """.split()
        cmd_2 = cmd_2 + self._get_test_config_overrides(epochs=3) + model_config
        monkeypatch.setattr(sys, "argv", cmd_2)
        runpy.run_path(TUNE_PATH, run_name="__main__")

        # Second epoch only
        resumed_loss_values = get_loss_values_from_metric_logger(resumed_log_file)

        torch.testing.assert_close(
            resumed_loss_values, expected_loss_values, rtol=1e-5, atol=1e-5
        )

    @pytest.mark.integration_test
    @gpu_test(gpu_count=2)
    def test_training_state_on_resume_with_async_checkpointing(
        self, tmpdir, monkeypatch
    ):
        """Test whether the recipe state is correctly updated on resume. Since this
        is model agnostic, we should run this on the small model only. The test
        consists of three stages:
            - Train a model for 2 epochs
            - Resume training after epoch 1
            - Make sure final loss matches the expected value of a model successfully resumed from a ckpt
        Unlike `tests.recipes.test_lora_finetune_single_device`, this test does not use pre-computed loss
        values to benchmark against. This test just ensures the loss values are identical when resuming.
        """

        ckpt = "llama3_tune"
        ckpt_path = Path(CKPT_MODEL_PATHS[ckpt])
        ckpt_dir = ckpt_path.parent
        log_file = gen_log_file_name(tmpdir)
        tokenizer_path = Path(TOKENIZER_PATHS["llama3"])

        # Config file needed for model conversion.
        # Create a second copy for training resume
        write_hf_ckpt_config(ckpt_dir)
        write_hf_ckpt_config(tmpdir)

        # Train for two epochs
        cmd_1 = f"""
        tune run --nnodes 1 --nproc_per_node 2 full_dpo_distributed \
            --config llama3_1/8B_full_dpo \
            output_dir={tmpdir} \
            checkpointer=torchtune.training.FullModelTorchTuneCheckpointer \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[{ckpt_path}]\
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type=LLAMA3 \
            ref_checkpointer=torchtune.training.FullModelTorchTuneCheckpointer \
            ref_checkpointer.checkpoint_dir='{ckpt_dir}' \
            ref_checkpointer.checkpoint_files=[{ckpt_path}]\
            ref_checkpointer.output_dir={tmpdir} \
            ref_checkpointer.model_type=LLAMA3 \
            tokenizer.path='{tokenizer_path}' \
            tokenizer.prompt_template=null \
            tokenizer.max_seq_len=256 \
            metric_logger.filename={log_file} \
            enable_activation_checkpointing=True \
            enable_activation_offloading=True \
            enable_async_checkpointing=True \
            batch_size=1 \
            gradient_accumulation_steps=4
        """.split()

        model_config = MODEL_TEST_CONFIGS["llama3"]

        cmd_1 = cmd_1 + self._get_test_config_overrides() + model_config
        monkeypatch.setattr(sys, "argv", cmd_1)
        # with pytest.raises(SystemExit, match=""):
        runpy.run_path(TUNE_PATH, run_name="__main__")

        expected_loss_values = get_loss_values_from_metric_logger(log_file)

        resumed_log_dir = (tmpdir / "resumed/").mkdir()
        resumed_log_file = gen_log_file_name(resumed_log_dir)

        # Resume training
        cmd_2 = f"""
        tune run --nnodes 1 --nproc_per_node 2 full_dpo_distributed \
            --config llama3_1/8B_full_dpo \
            output_dir={tmpdir} \
            checkpointer=torchtune.training.FullModelTorchTuneCheckpointer \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[{ckpt_path}]\
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type=LLAMA3 \
            ref_checkpointer=torchtune.training.FullModelTorchTuneCheckpointer \
            ref_checkpointer.checkpoint_dir='{ckpt_dir}' \
            ref_checkpointer.checkpoint_files=[{ckpt_path}]\
            ref_checkpointer.output_dir={tmpdir} \
            ref_checkpointer.model_type=LLAMA3 \
            resume_from_checkpoint=True \
            tokenizer.path='{tokenizer_path}' \
            tokenizer.prompt_template=null \
            tokenizer.max_seq_len=256 \
            metric_logger.filename={resumed_log_file} \
            enable_activation_checkpointing=True \
            enable_activation_offloading=True \
            enable_async_checkpointing=True \
            batch_size=1 \
            gradient_accumulation_steps=4
        """.split()
        cmd_2 = cmd_2 + self._get_test_config_overrides(epochs=3) + model_config
        monkeypatch.setattr(sys, "argv", cmd_2)
        runpy.run_path(TUNE_PATH, run_name="__main__")

        # Second epoch only
        resumed_loss_values = get_loss_values_from_metric_logger(resumed_log_file)

        torch.testing.assert_close(
            resumed_loss_values, expected_loss_values, rtol=1e-5, atol=1e-5
        )
