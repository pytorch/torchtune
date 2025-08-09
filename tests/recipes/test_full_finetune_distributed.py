# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import runpy
import shutil
import sys
from pathlib import Path

import pytest
import torch
from packaging import version
from tests.common import TUNE_PATH

from tests.recipes.utils import dummy_alpaca_dataset_config, MODEL_TEST_CONFIGS
from tests.test_utils import (
    CKPT_MODEL_PATHS,
    gen_log_file_name,
    get_loss_values_from_metric_logger,
    gpu_test,
    TOKENIZER_PATHS,
)

from torchtune.training.checkpointing._utils import get_largest_iter_folder


class TestFullFinetuneDistributedRecipe:
    def _get_test_config_overrides(self, epochs: int = 2):
        return [
            "dtype=fp32",
            "enable_activation_checkpointing=False",
            "enable_activation_offloading=False",
            "dataset.train_on_input=False",
            "seed=9",
            f"epochs={epochs}",
            "max_steps_per_epoch=2",
            "optimizer=torch.optim.AdamW",
            "optimizer.lr=2e-5",
            "log_every_n_steps=1",
        ] + dummy_alpaca_dataset_config()

    def _fetch_expected_loss_values_multi_rank(self, model_type):
        loss_values_map = {
            "llama3_hf_138m": [11.8934, 11.9444, 11.8903, 11.8915],
        }
        return loss_values_map[model_type]

    def _fetch_expected_loss_values_single_rank(self, model_type):
        loss_values_map = {"llama3_hf_138m": [11.8721, 11.9327, 11.8781, 11.9294]}
        return loss_values_map[model_type]

    @pytest.mark.integration_test
    @pytest.mark.parametrize(
        "config, model_ckpt, micro_batch_size, gradient_accumulation_steps, optim_in_bwd",
        [
            ("llama3/8B_full", "llama3_hf_138m", 1, 4, False),
            ("llama3/8B_full", "llama3_hf_138m", 4, 1, True),
        ],
    )
    @gpu_test(gpu_count=2)
    def test_loss(
        self,
        micro_batch_size,
        gradient_accumulation_steps,
        config,
        model_ckpt,
        optim_in_bwd,
        tmpdir,
        monkeypatch,
    ):
        ckpt_dir = Path(CKPT_MODEL_PATHS[model_ckpt])
        tokenizer_path = Path(TOKENIZER_PATHS[model_ckpt])
        model_config = MODEL_TEST_CONFIGS[model_ckpt]
        log_file = gen_log_file_name(tmpdir)

        cmd = f"""
        tune run --nnodes 1 --nproc_per_node 2 full_finetune_distributed \
            --config {config} \
            batch_size={micro_batch_size} \
            gradient_accumulation_steps={gradient_accumulation_steps} \
            output_dir={tmpdir} \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[model.safetensors]\
            checkpointer.output_dir={tmpdir} \
            tokenizer.path='{tokenizer_path}' \
            tokenizer.prompt_template=null \
            metric_logger.filename={log_file} \
        """.split()

        cmd = cmd + self._get_test_config_overrides() + model_config
        # "optimizer_in_bwd=True" would free gradient info before clip_grad, causing
        # wrong grad_norm, so we only test one of them each time. But loss values
        # should be the same.
        if not optim_in_bwd:
            cmd.append("clip_grad_norm=100")
            # Test that gradient clipping works with CPU offload
            cmd.append("fsdp_cpu_offload=True")
        else:
            cmd.append("optimizer_in_bwd=True")

        monkeypatch.setattr(sys, "argv", cmd)
        runpy.run_path(TUNE_PATH, run_name="__main__")
        loss_values = get_loss_values_from_metric_logger(log_file)
        expected_loss_values = self._fetch_expected_loss_values_multi_rank(model_ckpt)
        torch.testing.assert_close(
            loss_values, expected_loss_values, rtol=1e-4, atol=1e-4
        )

    @pytest.mark.skipif(
        version.parse(torch.__version__).base_version < "2.8.0",
        reason="2D parallel test requires PyTorch >= 2.8",
    )
    @pytest.mark.integration_test
    @pytest.mark.parametrize(
        "config, model_ckpt, micro_batch_size, gradient_accumulation_steps, optim_in_bwd, tensor_parallel_dim",
        [
            ("llama3/8B_full", "llama3_hf_138m", 4, 1, True, 2),
            ("llama3/8B_full", "llama3_hf_138m", 4, 1, True, 4),
        ],
    )
    @gpu_test(gpu_count=4)
    def test_loss_2d_parallel(
        self,
        micro_batch_size,
        gradient_accumulation_steps,
        config,
        model_ckpt,
        optim_in_bwd,
        tensor_parallel_dim,
        tmpdir,
        monkeypatch,
    ):
        ckpt_dir = Path(CKPT_MODEL_PATHS[model_ckpt])
        tokenizer_path = Path(TOKENIZER_PATHS[model_ckpt])
        model_config = MODEL_TEST_CONFIGS[model_ckpt]
        log_file = gen_log_file_name(tmpdir)

        cmd = f"""
        tune run --nnodes 1 --nproc_per_node 4 full_finetune_distributed \
            --config {config} \
            batch_size={micro_batch_size} \
            gradient_accumulation_steps={gradient_accumulation_steps} \
            output_dir={tmpdir} \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[model.safetensors]\
            checkpointer.output_dir={tmpdir} \
            tokenizer.path='{tokenizer_path}' \
            tokenizer.prompt_template=null \
            tensor_parallel_dim={tensor_parallel_dim} \
            tensor_parallel_plan._component_=torchtune.models.llama3.base_llama_tp_plan \
            metric_logger.filename={log_file} \
        """.split()
        cmd = cmd + self._get_test_config_overrides() + model_config
        # "optimizer_in_bwd=True" would free gradient info before clip_grad, causing
        # wrong grad_norm, so we only test one of them each time. But loss values
        # should be the same.
        if not optim_in_bwd:
            cmd.append("clip_grad_norm=100")
            # Test that gradient clipping works with CPU offload
            cmd.append("fsdp_cpu_offload=True")
        else:
            cmd.append("optimizer_in_bwd=True")

        monkeypatch.setattr(sys, "argv", cmd)
        runpy.run_path(TUNE_PATH, run_name="__main__")
        loss_values = get_loss_values_from_metric_logger(log_file)

        # For tp_dim = 2, we have dp_dim = 2, so 2x global batch size.
        # For tp_dim = 4 there is no data parallelism (since there are 4 workers).
        # This means we expect the multi-rank loss for tp_dim=2 but single-rank loss for tp_dim=4.
        expected_loss_values = (
            self._fetch_expected_loss_values_multi_rank(model_ckpt)
            if tensor_parallel_dim == 2
            else self._fetch_expected_loss_values_single_rank(model_ckpt)
        )

        torch.testing.assert_close(
            loss_values, expected_loss_values, rtol=1e-4, atol=1e-4
        )

    @pytest.mark.integration_test
    @pytest.mark.parametrize(
        "config, model_ckpt, micro_batch_size, gradient_accumulation_steps, optim_in_bwd",
        [
            ("llama3/8B_full", "llama3_hf_138m", 1, 4, False),
            ("llama3/8B_full", "llama3_hf_138m", 4, 1, True),
        ],
    )
    @gpu_test(gpu_count=1)
    def test_loss_single_rank(
        self,
        micro_batch_size,
        gradient_accumulation_steps,
        config,
        model_ckpt,
        optim_in_bwd,
        tmpdir,
        monkeypatch,
    ):
        ckpt_dir = Path(CKPT_MODEL_PATHS[model_ckpt])
        tokenizer_path = Path(TOKENIZER_PATHS[model_ckpt])
        model_config = MODEL_TEST_CONFIGS[model_ckpt]
        log_file = gen_log_file_name(tmpdir)

        cmd = f"""
        tune run --nnodes 1 --nproc_per_node 1 full_finetune_distributed \
            --config {config} \
            batch_size={micro_batch_size} \
            gradient_accumulation_steps={gradient_accumulation_steps} \
            output_dir={tmpdir} \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[model.safetensors]\
            checkpointer.output_dir={tmpdir} \
            tokenizer.path='{tokenizer_path}' \
            tokenizer.prompt_template=null \
            metric_logger.filename={log_file} \
        """.split()
        model_config = MODEL_TEST_CONFIGS[model_ckpt]
        cmd = cmd + self._get_test_config_overrides() + model_config
        # "optimizer_in_bwd=True" would free gradient info before clip_grad, causing
        # wrong grad_norm, so we only test one of them each time. But loss values
        # should be the same.
        if not optim_in_bwd:
            cmd.append("clip_grad_norm=100")
        else:
            cmd.append("optimizer_in_bwd=True")

        monkeypatch.setattr(sys, "argv", cmd)
        runpy.run_path(TUNE_PATH, run_name="__main__")
        loss_values = get_loss_values_from_metric_logger(log_file)
        expected_loss_values = self._fetch_expected_loss_values_single_rank(model_ckpt)
        torch.testing.assert_close(
            loss_values, expected_loss_values, rtol=1e-4, atol=1e-4
        )

    @pytest.mark.skipif(
        torch.__version__ < "2.7.0", reason="Test requires at least PyTorch 2.7"
    )
    @pytest.mark.integration_test
    @pytest.mark.parametrize(
        "config, model_ckpt, micro_batch_size, gradient_accumulation_steps, optim_in_bwd",
        [
            ("llama3/8B_full", "llama3_hf_138m", 1, 4, False),
            ("llama3/8B_full", "llama3_hf_138m", 4, 1, True),
        ],
    )
    @gpu_test(gpu_count=2)
    def test_training_state_on_resume(
        self,
        micro_batch_size,
        gradient_accumulation_steps,
        config,
        model_ckpt,
        optim_in_bwd,
        tmpdir,
        monkeypatch,
    ):
        ckpt_dir = Path(CKPT_MODEL_PATHS[model_ckpt])
        tokenizer_path = Path(TOKENIZER_PATHS[model_ckpt])
        model_config = MODEL_TEST_CONFIGS[model_ckpt]
        log_file = gen_log_file_name(tmpdir)

        # Train for two epochs
        cmd_1 = f"""
        tune run --nnodes 1 --nproc_per_node 2 full_finetune_distributed \
            --config {config} \
            batch_size={micro_batch_size} \
            gradient_accumulation_steps={gradient_accumulation_steps} \
            output_dir={tmpdir} \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[model.safetensors]\
            checkpointer.output_dir={tmpdir} \
            tokenizer.path='{tokenizer_path}' \
            tokenizer.prompt_template=null \
        """.split()

        # "optimizer_in_bwd=True" would free gradient info before clip_grad, causing
        # wrong grad_norm, so we only test one of them each time. But loss values
        # should be the same.
        if not optim_in_bwd:
            cmd_1.append("clip_grad_norm=100")
            cmd_1.append("optimizer_in_bwd=False")
        else:
            cmd_1.append("optimizer_in_bwd=True")

        cmd_1 = cmd_1 + self._get_test_config_overrides() + model_config

        monkeypatch.setattr(sys, "argv", cmd_1)
        runpy.run_path(TUNE_PATH, run_name="__main__")

        # Resume training
        epoch_folder = get_largest_iter_folder(tmpdir)
        epoch_folder_minus_one = f"epoch_{int(epoch_folder.split('_')[-1]) - 1}"

        cmd_2 = f"""
        tune run --nnodes 1 --nproc_per_node 2 full_finetune_distributed \
            --config {config} \
            batch_size={micro_batch_size} \
            gradient_accumulation_steps={gradient_accumulation_steps} \
            output_dir={tmpdir} \
            checkpointer.checkpoint_dir='{tmpdir}/{epoch_folder_minus_one}' \
            checkpointer.checkpoint_files=["model.safetensors"]\
            checkpointer.output_dir={tmpdir} \
            tokenizer.path='{tokenizer_path}' \
            tokenizer.prompt_template=null \
            resume_from_checkpoint=True \
            metric_logger.filename={log_file}
        """.split()

        cmd_2 = cmd_2 + self._get_test_config_overrides() + model_config

        if not optim_in_bwd:
            cmd_2.append("clip_grad_norm=100")
            cmd_2.append("optimizer_in_bwd=False")
        else:
            cmd_2.append("optimizer_in_bwd=True")

        monkeypatch.setattr(sys, "argv", cmd_2)
        runpy.run_path(TUNE_PATH, run_name="__main__")

        loss_values = get_loss_values_from_metric_logger(log_file)
        expected_loss_values = self._fetch_expected_loss_values_multi_rank(model_ckpt)[
            2:
        ]
        torch.testing.assert_close(
            loss_values, expected_loss_values, rtol=1e-4, atol=1e-4
        )

    @pytest.mark.integration_test
    @pytest.mark.parametrize(
        "config, model_ckpt, micro_batch_size, gradient_accumulation_steps, optim_in_bwd",
        [
            ("llama3/8B_full", "llama3_hf_138m", 1, 4, False),
            ("llama3/8B_full", "llama3_hf_138m", 4, 1, True),
        ],
    )
    @gpu_test(gpu_count=1)
    def test_training_state_on_resume_from_distributed_checkpoint_single_rank(
        self,
        micro_batch_size,
        gradient_accumulation_steps,
        config,
        model_ckpt,
        optim_in_bwd,
        tmpdir,
        monkeypatch,
    ):
        """Test whether the recipe state is correctly updated on resume. Since this
        is model agnostic, we should run this on the small model only. The test
        consists of three stages:
            - Train a model for 2 epochs
            - Resume training after epoch 1
            - Make sure final loss matches the expected value of a model successfully resumed from a ckpt
        """

        ckpt_dir = Path(CKPT_MODEL_PATHS[model_ckpt])
        tokenizer_path = Path(TOKENIZER_PATHS[model_ckpt])
        model_config = MODEL_TEST_CONFIGS[model_ckpt]
        log_file = gen_log_file_name(tmpdir)

        # Train for two epochs
        cmd_1 = f"""
        tune run --nnodes 1 --nproc_per_node 1 full_finetune_distributed \
            --config {config} \
            batch_size={micro_batch_size} \
            gradient_accumulation_steps={gradient_accumulation_steps} \
            output_dir={tmpdir} \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[model.safetensors]\
            checkpointer.output_dir={tmpdir} \
            tokenizer.path='{tokenizer_path}' \
            tokenizer.prompt_template=null \
            metric_logger.filename={log_file} \
            enable_async_checkpointing=True \
        """.split()

        cmd_1 = cmd_1 + self._get_test_config_overrides() + model_config
        # "optimizer_in_bwd=True" would free gradient info before clip_grad, causing
        # wrong grad_norm, so we only test one of them each time. But loss values
        # should be the same.
        if not optim_in_bwd:
            cmd_1.append("clip_grad_norm=100")
            cmd_1.append("optimizer_in_bwd=False")
        else:
            cmd_1.append("optimizer_in_bwd=True")

        monkeypatch.setattr(sys, "argv", cmd_1)
        runpy.run_path(TUNE_PATH, run_name="__main__")

        expected_loss_values_first_run = get_loss_values_from_metric_logger(log_file)
        expected_loss_values = self._fetch_expected_loss_values_single_rank(model_ckpt)
        torch.testing.assert_close(
            expected_loss_values_first_run, expected_loss_values, rtol=1e-4, atol=1e-4
        )

        resumed_log_dir = (tmpdir / "resumed/").mkdir()
        resumed_log_file = gen_log_file_name(resumed_log_dir)
        shutil.rmtree((tmpdir / "epoch_1"))

        # Resume training
        cmd_2 = f"""
        tune run --nnodes 1 --nproc_per_node 1 full_finetune_distributed \
            --config {config} \
            batch_size={micro_batch_size} \
            gradient_accumulation_steps={gradient_accumulation_steps} \
            output_dir={tmpdir} \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[model.safetensors]\
            checkpointer.output_dir={tmpdir} \
            tokenizer.path='{tokenizer_path}' \
            tokenizer.prompt_template=null \
            metric_logger.filename={resumed_log_file} \
            resume_from_checkpoint=True \
            enable_async_checkpointing=True \
        """.split()

        cmd_2 = cmd_2 + self._get_test_config_overrides(epochs=3) + model_config

        if not optim_in_bwd:
            cmd_2.append("clip_grad_norm=100")
            cmd_2.append("optimizer_in_bwd=False")
        else:
            cmd_2.append("optimizer_in_bwd=True")

        monkeypatch.setattr(sys, "argv", cmd_2)
        runpy.run_path(TUNE_PATH, run_name="__main__")

        # Validate that the expected loss values are close to the ones observed after the resume
        resumed_loss_values = get_loss_values_from_metric_logger(resumed_log_file)
        torch.testing.assert_close(
            resumed_loss_values[:2], expected_loss_values[2:], rtol=1e-3, atol=1e-3
        )

    @pytest.mark.integration_test
    @pytest.mark.parametrize(
        "config, model_ckpt, micro_batch_size, gradient_accumulation_steps, optim_in_bwd",
        [
            ("llama3/8B_full", "llama3_hf_138m", 1, 4, False),
            ("llama3/8B_full", "llama3_hf_138m", 4, 1, True),
        ],
    )
    @gpu_test(gpu_count=2)
    def test_training_state_on_resume_from_distributed_checkpoint_multi_rank(
        self,
        micro_batch_size,
        gradient_accumulation_steps,
        config,
        model_ckpt,
        optim_in_bwd,
        tmpdir,
        monkeypatch,
    ):
        """Test whether the recipe state is correctly updated on resume. Since this
        is model agnostic, we should run this on the small model only. The test
        consists of three stages:
            - Train a model for 2 epochs
            - Resume training after epoch 1
            - Make sure final loss matches the expected value of a model successfully resumed from a ckpt
        """

        ckpt_dir = Path(CKPT_MODEL_PATHS[model_ckpt])
        tokenizer_path = Path(TOKENIZER_PATHS[model_ckpt])
        model_config = MODEL_TEST_CONFIGS[model_ckpt]
        log_file = gen_log_file_name(tmpdir)

        # Train for two epochs
        cmd_1 = f"""
        tune run --nnodes 1 --nproc_per_node 2 full_finetune_distributed \
            --config {config} \
            batch_size={micro_batch_size} \
            gradient_accumulation_steps={gradient_accumulation_steps} \
            output_dir={tmpdir} \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[model.safetensors]\
            checkpointer.output_dir={tmpdir} \
            tokenizer.path='{tokenizer_path}' \
            tokenizer.prompt_template=null \
            metric_logger.filename={log_file} \
            enable_async_checkpointing=True \
        """.split()

        cmd_1 = cmd_1 + self._get_test_config_overrides() + model_config
        # "optimizer_in_bwd=True" would free gradient info before clip_grad, causing
        # wrong grad_norm, so we only test one of them each time. But loss values
        # should be the same.
        if not optim_in_bwd:
            cmd_1.append("clip_grad_norm=100")
            cmd_1.append("optimizer_in_bwd=False")
        else:
            cmd_1.append("optimizer_in_bwd=True")

        monkeypatch.setattr(sys, "argv", cmd_1)
        runpy.run_path(TUNE_PATH, run_name="__main__")

        expected_loss_values_first_run = get_loss_values_from_metric_logger(log_file)

        # Validate that the expected loss values are close to the ones observed in the first run
        expected_loss_values = self._fetch_expected_loss_values_multi_rank(model_ckpt)
        torch.testing.assert_close(
            expected_loss_values_first_run, expected_loss_values, rtol=1e-4, atol=1e-4
        )

        resumed_log_dir = (tmpdir / "resumed/").mkdir()
        resumed_log_file = gen_log_file_name(resumed_log_dir)
        shutil.rmtree((tmpdir / "epoch_1"))

        # Resume training
        cmd_2 = f"""
        tune run --nnodes 1 --nproc_per_node 2 full_finetune_distributed \
            --config {config} \
            batch_size={micro_batch_size} \
            gradient_accumulation_steps={gradient_accumulation_steps} \
            output_dir={tmpdir} \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[model.safetensors]\
            checkpointer.output_dir={tmpdir} \
            tokenizer.path='{tokenizer_path}' \
            tokenizer.prompt_template=null \
            metric_logger.filename={resumed_log_file} \
            resume_from_checkpoint=True \
            enable_async_checkpointing=True \
        """.split()

        cmd_2 = cmd_2 + self._get_test_config_overrides(epochs=3) + model_config

        if not optim_in_bwd:
            cmd_2.append("clip_grad_norm=100")
            cmd_2.append("optimizer_in_bwd=False")
        else:
            cmd_2.append("optimizer_in_bwd=True")

        monkeypatch.setattr(sys, "argv", cmd_2)
        runpy.run_path(TUNE_PATH, run_name="__main__")

        # Second epoch only
        # Validate that the expected loss values are close to the ones observed after the resume
        resumed_loss_values = get_loss_values_from_metric_logger(resumed_log_file)
        torch.testing.assert_close(
            resumed_loss_values[:2], expected_loss_values[2:], rtol=1e-3, atol=1e-3
        )
