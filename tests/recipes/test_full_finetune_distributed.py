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

from torchtune.training.checkpointing._utils import (
    get_largest_iter_folder,
    RECIPE_STATE_DIRNAME,
    SHARD_FNAME,
)


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
            "llama2": [10.5209, 10.5217, 10.4945, 10.5136],
            "llama3": [11.9839, 11.9684, 11.9596, 11.93656],
        }
        return loss_values_map[model_type]

    def _fetch_expected_loss_values_single_rank(self, model_type):
        loss_values_map = {
            "llama2": [10.5051, 10.5572, 10.4780, 10.5678],
            "llama3": [11.9742, 12.0049, 11.9382, 12.0464],
        }
        return loss_values_map[model_type]

    @pytest.mark.integration_test
    @pytest.mark.parametrize(
        "config, model_type, ckpt_type, micro_batch_size, gradient_accumulation_steps, optim_in_bwd",
        [
            ("llama2/7B_full", "llama2", "hf", 1, 4, False),
            ("llama3/8B_full", "llama3", "tune", 1, 4, False),
            ("llama3/8B_full", "llama3", "tune", 4, 1, True),
        ],
    )
    @gpu_test(gpu_count=2)
    def test_loss(
        self,
        micro_batch_size,
        gradient_accumulation_steps,
        config,
        model_type,
        ckpt_type,
        optim_in_bwd,
        tmpdir,
        monkeypatch,
    ):
        ckpt_component = CKPT_COMPONENT_MAP[ckpt_type]
        ckpt = model_type + "_" + ckpt_type
        ckpt_path = Path(CKPT_MODEL_PATHS[ckpt])
        tokenizer_path = Path(TOKENIZER_PATHS[model_type])
        ckpt_dir = ckpt_path.parent
        log_file = gen_log_file_name(tmpdir)

        # Config file needed for model conversion.
        write_hf_ckpt_config(ckpt_dir)

        cmd = f"""
        tune run --nnodes 1 --nproc_per_node 2 full_finetune_distributed \
            --config {config} \
            batch_size={micro_batch_size} \
            gradient_accumulation_steps={gradient_accumulation_steps} \
            output_dir={tmpdir} \
            checkpointer._component_={ckpt_component} \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[{ckpt_path}]\
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type={model_type.upper()} \
            tokenizer.path='{tokenizer_path}' \
            tokenizer.prompt_template=null \
            metric_logger.filename={log_file} \
        """.split()
        model_config = MODEL_TEST_CONFIGS[model_type]
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
        expected_loss_values = self._fetch_expected_loss_values_multi_rank(model_type)
        torch.testing.assert_close(
            loss_values, expected_loss_values, rtol=1e-4, atol=1e-4
        )

    @pytest.mark.skipif(
        torch.__version__ < "2.8.0", reason="2D parallel test requires PyTorch >= 2.8"
    )
    @pytest.mark.integration_test
    @pytest.mark.parametrize(
        "config, model_type, ckpt_type, micro_batch_size, gradient_accumulation_steps, optim_in_bwd, tensor_parallel_dim",
        [
            ("llama3/8B_full", "llama3", "tune", 4, 1, True, 2),
            ("llama3/8B_full", "llama3", "tune", 4, 1, True, 4),
        ],
    )
    @gpu_test(gpu_count=4)
    def test_loss_2d_parallel(
        self,
        micro_batch_size,
        gradient_accumulation_steps,
        config,
        model_type,
        ckpt_type,
        optim_in_bwd,
        tensor_parallel_dim,
        tmpdir,
        monkeypatch,
    ):
        ckpt_component = CKPT_COMPONENT_MAP[ckpt_type]
        ckpt = model_type + "_" + ckpt_type
        ckpt_path = Path(CKPT_MODEL_PATHS[ckpt])
        tokenizer_path = Path(TOKENIZER_PATHS[model_type])
        ckpt_dir = ckpt_path.parent
        log_file = gen_log_file_name(tmpdir)
        tp_plan = "torchtune.models.llama3.base_llama_tp_plan"

        # Config file needed for model conversion.
        write_hf_ckpt_config(ckpt_dir)

        cmd = f"""
        tune run --nnodes 1 --nproc_per_node 4 full_finetune_distributed \
            --config {config} \
            batch_size={micro_batch_size} \
            gradient_accumulation_steps={gradient_accumulation_steps} \
            output_dir={tmpdir} \
            checkpointer._component_={ckpt_component} \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[{ckpt_path}]\
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type={model_type.upper()} \
            tokenizer.path='{tokenizer_path}' \
            tokenizer.prompt_template=null \
            tensor_parallel_dim={tensor_parallel_dim} \
            tensor_parallel_plan._component_={tp_plan} \
            metric_logger.filename={log_file} \
        """.split()
        model_config = MODEL_TEST_CONFIGS[model_type]
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
            self._fetch_expected_loss_values_multi_rank(model_type)
            if tensor_parallel_dim == 2
            else self._fetch_expected_loss_values_single_rank(model_type)
        )

        torch.testing.assert_close(
            loss_values, expected_loss_values, rtol=1e-4, atol=1e-4
        )

    @pytest.mark.integration_test
    @pytest.mark.parametrize(
        "config, model_type, ckpt_type, micro_batch_size, gradient_accumulation_steps, optim_in_bwd",
        [
            ("llama2/7B_full", "llama2", "hf", 1, 4, False),
            ("llama3/8B_full", "llama3", "tune", 1, 4, False),
            ("llama3/8B_full", "llama3", "tune", 4, 1, True),
        ],
    )
    @gpu_test(gpu_count=1)
    def test_loss_single_rank(
        self,
        micro_batch_size,
        gradient_accumulation_steps,
        config,
        model_type,
        ckpt_type,
        optim_in_bwd,
        tmpdir,
        monkeypatch,
    ):
        ckpt_component = CKPT_COMPONENT_MAP[ckpt_type]
        ckpt = model_type + "_" + ckpt_type
        ckpt_path = Path(CKPT_MODEL_PATHS[ckpt])
        tokenizer_path = Path(TOKENIZER_PATHS[model_type])
        ckpt_dir = ckpt_path.parent
        log_file = gen_log_file_name(tmpdir)

        # Config file needed for model conversion.
        write_hf_ckpt_config(ckpt_dir)

        cmd = f"""
        tune run --nnodes 1 --nproc_per_node 1 full_finetune_distributed \
            --config {config} \
            batch_size={micro_batch_size} \
            gradient_accumulation_steps={gradient_accumulation_steps} \
            output_dir={tmpdir} \
            checkpointer._component_={ckpt_component} \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[{ckpt_path}]\
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type={model_type.upper()} \
            tokenizer.path='{tokenizer_path}' \
            tokenizer.prompt_template=null \
            metric_logger.filename={log_file} \
        """.split()
        model_config = MODEL_TEST_CONFIGS[model_type]
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
        expected_loss_values = self._fetch_expected_loss_values_single_rank(model_type)
        torch.testing.assert_close(
            loss_values, expected_loss_values, rtol=1e-4, atol=1e-4
        )

    @pytest.mark.skipif(
        torch.__version__ < "2.7.0", reason="Test requires at least PyTorch 2.7"
    )
    @pytest.mark.integration_test
    @pytest.mark.parametrize(
        "config, model_type, ckpt_type, micro_batch_size, gradient_accumulation_steps, optim_in_bwd",
        [
            ("llama3/8B_full", "llama3", "tune", 1, 4, False),
            ("llama3/8B_full", "llama3", "tune", 4, 1, True),
        ],
    )
    @gpu_test(gpu_count=2)
    def test_training_state_on_resume(
        self,
        micro_batch_size,
        gradient_accumulation_steps,
        config,
        model_type,
        ckpt_type,
        optim_in_bwd,
        tmpdir,
        monkeypatch,
    ):
        ckpt_component = CKPT_COMPONENT_MAP[ckpt_type]
        ckpt = model_type + "_" + ckpt_type
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
        tune run --nnodes 1 --nproc_per_node 2 full_finetune_distributed \
            --config {config} \
            batch_size={micro_batch_size} \
            gradient_accumulation_steps={gradient_accumulation_steps} \
            output_dir={tmpdir} \
            checkpointer._component_={ckpt_component} \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[{ckpt_path}]\
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type={model_type.upper()} \
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

        model_config = MODEL_TEST_CONFIGS[model_type]
        cmd_1 = cmd_1 + self._get_test_config_overrides() + model_config

        monkeypatch.setattr(sys, "argv", cmd_1)
        runpy.run_path(TUNE_PATH, run_name="__main__")

        # Resume training
        epoch_folder = get_largest_iter_folder(tmpdir)
        epoch_folder_minus_one = f"epoch_{int(epoch_folder.split('_')[-1]) - 1}"
        suffix = ".safetensors" if ckpt_type == "hf" else ".bin"
        model_ckpt_fname = (
            SHARD_FNAME.format(cpt_idx="1".zfill(5), num_shards="1".zfill(5)) + suffix
        )
        cmd_2 = f"""
        tune run --nnodes 1 --nproc_per_node 2 full_finetune_distributed \
            --config {config} \
            batch_size={micro_batch_size} \
            gradient_accumulation_steps={gradient_accumulation_steps} \
            output_dir={tmpdir} \
            checkpointer._component_={ckpt_component} \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[{os.path.join(epoch_folder_minus_one, model_ckpt_fname)}]\
            checkpointer.recipe_checkpoint={os.path.join(RECIPE_STATE_DIRNAME, "recipe_state.pt")}\
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type={model_type.upper()} \
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

        expected_loss_values = self._fetch_expected_loss_values_multi_rank(model_type)[
            2:
        ]

        loss_values = get_loss_values_from_metric_logger(log_file)
        torch.testing.assert_close(
            loss_values, expected_loss_values, rtol=1e-4, atol=1e-4
        )

    @pytest.mark.integration_test
    @pytest.mark.parametrize(
        "config, model_type, ckpt_type, micro_batch_size, gradient_accumulation_steps, optim_in_bwd",
        [
            ("llama2/7B_full", "llama2", "hf", 1, 4, False),
            ("llama3/8B_full", "llama3", "tune", 1, 4, False),
            ("llama3/8B_full", "llama3", "tune", 4, 1, True),
        ],
    )
    @gpu_test(gpu_count=1)
    def test_training_state_on_resume_from_distributed_checkpoint_single_rank(
        self,
        micro_batch_size,
        gradient_accumulation_steps,
        config,
        model_type,
        ckpt_type,
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

        ckpt_component = CKPT_COMPONENT_MAP[ckpt_type]
        ckpt = model_type + "_" + ckpt_type
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
        tune run --nnodes 1 --nproc_per_node 1 full_finetune_distributed \
            --config {config} \
            batch_size={micro_batch_size} \
            gradient_accumulation_steps={gradient_accumulation_steps} \
            output_dir={tmpdir} \
            checkpointer._component_={ckpt_component} \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[{ckpt_path}]\
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type={model_type.upper()} \
            tokenizer.path='{tokenizer_path}' \
            tokenizer.prompt_template=null \
            metric_logger.filename={log_file} \
            enable_async_checkpointing=True \
        """.split()

        model_config = MODEL_TEST_CONFIGS[model_type]
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

        resumed_log_dir = (tmpdir / "resumed/").mkdir()
        resumed_log_file = gen_log_file_name(resumed_log_dir)

        # Resume training
        cmd_2 = f"""
        tune run --nnodes 1 --nproc_per_node 1 full_finetune_distributed \
            --config {config} \
            batch_size={micro_batch_size} \
            gradient_accumulation_steps={gradient_accumulation_steps} \
            output_dir={tmpdir} \
            checkpointer._component_={ckpt_component} \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[{ckpt_path}]\
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type={model_type.upper()} \
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

        # Validate that the expected loss values are close to the ones observed in the first run
        expected_loss_values = self._fetch_expected_loss_values_single_rank(model_type)
        torch.testing.assert_close(
            expected_loss_values_first_run, expected_loss_values, rtol=1e-4, atol=1e-4
        )

        # Second epoch only
        # Validate that the expected loss values are close to the ones observed after the resume
        resumed_loss_values = get_loss_values_from_metric_logger(resumed_log_file)
        torch.testing.assert_close(
            resumed_loss_values[:2], expected_loss_values[2:], rtol=1e-4, atol=1e-4
        )

    @pytest.mark.integration_test
    @pytest.mark.parametrize(
        "config, model_type, ckpt_type, micro_batch_size, gradient_accumulation_steps, optim_in_bwd",
        [
            ("llama2/7B_full", "llama2", "hf", 1, 4, False),
            ("llama3/8B_full", "llama3", "tune", 1, 4, False),
            ("llama3/8B_full", "llama3", "tune", 4, 1, True),
        ],
    )
    @gpu_test(gpu_count=2)
    def test_training_state_on_resume_from_distributed_checkpoint_multi_rank(
        self,
        micro_batch_size,
        gradient_accumulation_steps,
        config,
        model_type,
        ckpt_type,
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

        ckpt_component = CKPT_COMPONENT_MAP[ckpt_type]
        ckpt = model_type + "_" + ckpt_type
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
        tune run --nnodes 1 --nproc_per_node 2 full_finetune_distributed \
            --config {config} \
            batch_size={micro_batch_size} \
            gradient_accumulation_steps={gradient_accumulation_steps} \
            output_dir={tmpdir} \
            checkpointer._component_={ckpt_component} \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[{ckpt_path}]\
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type={model_type.upper()} \
            tokenizer.path='{tokenizer_path}' \
            tokenizer.prompt_template=null \
            metric_logger.filename={log_file} \
            enable_async_checkpointing=True \
        """.split()

        model_config = MODEL_TEST_CONFIGS[model_type]
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

        resumed_log_dir = (tmpdir / "resumed/").mkdir()
        resumed_log_file = gen_log_file_name(resumed_log_dir)

        # Resume training
        cmd_2 = f"""
        tune run --nnodes 1 --nproc_per_node 2 full_finetune_distributed \
            --config {config} \
            batch_size={micro_batch_size} \
            gradient_accumulation_steps={gradient_accumulation_steps} \
            output_dir={tmpdir} \
            checkpointer._component_={ckpt_component} \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[{ckpt_path}]\
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type={model_type.upper()} \
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

        # Validate that the expected loss values are close to the ones observed in the first run
        expected_loss_values = self._fetch_expected_loss_values_multi_rank(model_type)
        torch.testing.assert_close(
            expected_loss_values_first_run, expected_loss_values, rtol=1e-4, atol=1e-4
        )

        # Second epoch only
        # Validate that the expected loss values are close to the ones observed after the resume
        resumed_loss_values = get_loss_values_from_metric_logger(resumed_log_file)
        torch.testing.assert_close(
            resumed_loss_values[:2], expected_loss_values[2:], rtol=1e-4, atol=1e-4
        )
