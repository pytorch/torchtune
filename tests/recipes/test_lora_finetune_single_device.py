# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import runpy
import shutil
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

from torchtune.training.checkpointing._utils import (
    ADAPTER_MODEL_FNAME,
    get_largest_iter_folder,
    safe_torch_load,
    SHARD_FNAME,
)


class TestLoRAFinetuneSingleDeviceRecipe:
    def _get_test_config_overrides(self, dtype_str: str = "fp32", epochs: int = 2):
        return [
            f"dtype={dtype_str}",
            "dataset.train_on_input=False",
            "seed=9",
            f"epochs={epochs}",
            "max_steps_per_epoch=2",
            "optimizer.lr=2e-5",
            "log_every_n_steps=1",
            "clip_grad_norm=100",
        ] + dummy_alpaca_dataset_config()

    def _fetch_expected_loss_values(self, model_type):
        loss_values_map = {
            "llama3": [11.9838, 11.9691, 11.9616, 11.9383],
        }
        return loss_values_map[model_type]

    def _fetch_qlora_expected_loss_values(self, dtype):
        if dtype == "bf16":
            return [11.9857, 11.9711, 11.9619, 11.9407]
        return [11.9857, 11.9712, 11.9613, 11.9408]

    @pytest.mark.integration_test
    @pytest.mark.parametrize(
        "config, model_type, ckpt_type, micro_batch_size, gradient_accumulation_steps, compile",
        [
            ("llama3/8B_lora_single_device", "llama3", "tune", 2, 4, True),
            ("llama3/8B_lora_single_device", "llama3", "tune", 2, 4, False),
        ],
    )
    @gpu_test(gpu_count=1)
    def test_loss(
        self,
        compile,
        micro_batch_size,
        gradient_accumulation_steps,
        config,
        model_type,
        ckpt_type,
        tmpdir,
        monkeypatch,
    ):
        ckpt_component = CKPT_COMPONENT_MAP[ckpt_type]
        ckpt = model_type + "_" + ckpt_type
        ckpt_path = Path(CKPT_MODEL_PATHS[ckpt])
        tokenizer_path = Path(TOKENIZER_PATHS[model_type])
        ckpt_dir = ckpt_path.parent
        log_file = gen_log_file_name(tmpdir)

        cmd = f"""
        tune run lora_finetune_single_device \
            --config {config} \
            batch_size={micro_batch_size} \
            gradient_accumulation_steps={gradient_accumulation_steps} \
            output_dir={tmpdir} \
            model.lora_attn_modules=['q_proj','v_proj'] \
            model.apply_lora_to_mlp=False \
            checkpointer._component_={ckpt_component} \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[{ckpt_path}] \
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type={model_type.upper()} \
            tokenizer.path='{tokenizer_path}' \
            tokenizer.prompt_template=null \
            metric_logger.filename={log_file} \
            compile={compile} \
        """.split()

        model_config = MODEL_TEST_CONFIGS[model_type + "_lora"]

        cmd = cmd + self._get_test_config_overrides(dtype_str="fp32") + model_config
        monkeypatch.setattr(sys, "argv", cmd)
        with pytest.raises(SystemExit, match=""):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        # Make sure to clear compile state in between tests
        if compile:
            torch._dynamo.reset()

        loss_values = get_loss_values_from_metric_logger(log_file)
        expected_loss_values = self._fetch_expected_loss_values(model_type)
        torch.testing.assert_close(
            loss_values, expected_loss_values, rtol=1e-5, atol=1e-5
        )

    @pytest.mark.integration_test
    @pytest.mark.parametrize(
        "dtype, compile, micro_batch_size, gradient_accumulation_steps",
        [
            ("fp32", True, 8, 1),
            ("bf16", True, 2, 4),
            ("fp32", False, 4, 2),
            ("bf16", False, 8, 1),
        ],
    )
    @gpu_test(gpu_count=1)
    def test_loss_qlora(
        self,
        dtype,
        compile,
        micro_batch_size,
        gradient_accumulation_steps,
        tmpdir,
        monkeypatch,
    ):
        ckpt = "llama3_tune"
        ckpt_path = Path(CKPT_MODEL_PATHS[ckpt])
        ckpt_dir = ckpt_path.parent
        log_file = gen_log_file_name(tmpdir)

        cmd = f"""
        tune run lora_finetune_single_device
            --config llama3/8B_qlora_single_device \
            batch_size={micro_batch_size} \
            gradient_accumulation_steps={gradient_accumulation_steps} \
            output_dir={tmpdir} \
            model.lora_attn_modules=['q_proj','v_proj','k_proj','output_proj'] \
            model.apply_lora_to_mlp=True \
            checkpointer=torchtune.training.FullModelTorchTuneCheckpointer
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[{ckpt_path}]\
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type=LLAMA3 \
            metric_logger.filename={log_file} \
            tokenizer.path=/tmp/test-artifacts/tokenizer_llama3.model \
            tokenizer.prompt_template=null \
            compile={compile} \
            enable_activation_checkpointing=False \
            enable_activation_offloading=False \
        """.split()

        model_config = MODEL_TEST_CONFIGS["llama3_qlora"]

        cmd = cmd + self._get_test_config_overrides(dtype_str=dtype) + model_config
        monkeypatch.setattr(sys, "argv", cmd)
        with pytest.raises(SystemExit):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        # Make sure to clear compile state in between tests
        if compile:
            torch._dynamo.reset()

        loss_values = get_loss_values_from_metric_logger(log_file)
        expected_loss_values = self._fetch_qlora_expected_loss_values(dtype=dtype)
        torch.testing.assert_close(
            loss_values, expected_loss_values, rtol=1e-4, atol=1e-4
        )

    @pytest.mark.parametrize("save_adapter_weights_only", [False, True])
    @pytest.mark.integration_test
    @gpu_test(gpu_count=1)
    def test_training_state_on_resume(
        self, tmpdir, monkeypatch, save_adapter_weights_only
    ):
        """Test whether the recipe state is correctly updated on resume. Since this
        is model agnostic, we should run this on the small model only. The test
        consists of three stages:
            - Train a model for 2 epochs
            - Resume training after epoch 1
            - Make sure final loss matches the expected value of a model successfully resumed from a ckpt
        """

        ckpt = "llama3_tune"
        ckpt_path = Path(CKPT_MODEL_PATHS[ckpt])
        ckpt_dir = ckpt_path.parent
        log_file = gen_log_file_name(tmpdir)

        # Config file needed for model conversion.
        # Create a second copy for training resume
        write_hf_ckpt_config(ckpt_dir)
        write_hf_ckpt_config(tmpdir)

        # Train for two epochs
        cmd_1 = f"""
        tune run lora_finetune_single_device \
            --config llama3/8B_lora_single_device \
            batch_size=8 \
            gradient_accumulation_steps=1 \
            output_dir={tmpdir} \
            model.lora_attn_modules=['q_proj','v_proj','k_proj','output_proj'] \
            model.apply_lora_to_mlp=True \
            checkpointer=torchtune.training.FullModelTorchTuneCheckpointer \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[{ckpt_path}]\
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type=LLAMA3 \
            tokenizer.path=/tmp/test-artifacts/tokenizer_llama3.model \
            tokenizer.prompt_template=null \
            save_adapter_weights_only={save_adapter_weights_only} \
            enable_activation_checkpointing=True \
            enable_activation_offloading=False \
        """.split()

        model_config = MODEL_TEST_CONFIGS["llama3_lora"]

        cmd_1 = cmd_1 + self._get_test_config_overrides() + model_config
        monkeypatch.setattr(sys, "argv", cmd_1)
        with pytest.raises(SystemExit, match=""):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        # Resume training
        epoch_folder = get_largest_iter_folder(tmpdir)
        epoch_folder_minus_one = f"epoch_{int(epoch_folder.split('_')[-1]) - 1}"
        cmd_2 = f"""
        tune run lora_finetune_single_device \
            --config llama3/8B_lora_single_device \
            batch_size=8 \
            gradient_accumulation_steps=1 \
            output_dir={tmpdir} \
            model.lora_attn_modules=['q_proj','v_proj','k_proj','output_proj'] \
            model.apply_lora_to_mlp=True \
            checkpointer=torchtune.training.FullModelTorchTuneCheckpointer \
            checkpointer.checkpoint_dir={os.path.join(tmpdir, epoch_folder_minus_one)} \
            checkpointer.checkpoint_files=[{ckpt_path}]\
            checkpointer.adapter_checkpoint={os.path.join(epoch_folder_minus_one, f"{ADAPTER_MODEL_FNAME}.pt")}
            checkpointer.recipe_checkpoint={os.path.join(tmpdir, epoch_folder_minus_one, "recipe_state.pt")}
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type=LLAMA3 \
            resume_from_checkpoint=True \
            metric_logger.filename={log_file} \
            tokenizer.path=/tmp/test-artifacts/tokenizer_llama3.model \
            tokenizer.prompt_template=null \
            enable_activation_checkpointing=True \
            enable_activation_offloading=False \
        """.split()
        cmd_2 = cmd_2 + self._get_test_config_overrides() + model_config
        monkeypatch.setattr(sys, "argv", cmd_2)
        with pytest.raises(SystemExit, match=""):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        # Second epoch only
        expected_loss_values = self._fetch_expected_loss_values("llama3")
        loss_values = get_loss_values_from_metric_logger(log_file)

        torch.testing.assert_close(
            loss_values, expected_loss_values, rtol=1e-5, atol=1e-5
        )

    @pytest.mark.parametrize("save_adapter_weights_only", [False, True])
    @pytest.mark.integration_test
    def test_training_state_on_resume_with_async_checkpointing(
        self, tmpdir, monkeypatch, save_adapter_weights_only
    ):
        """Test whether the recipe state is correctly updated on resume. Since this
        is model agnostic, we should run this on the small model only. The test
        consists of three stages:
            - Train a model for 2 epochs
            - Resume training after epoch 1
            - Make sure final loss matches the expected value of a model successfully resumed from a ckpt
        """

        ckpt = "llama3_tune"
        ckpt_path = Path(CKPT_MODEL_PATHS[ckpt])
        ckpt_dir = ckpt_path.parent
        log_file = gen_log_file_name(tmpdir)

        # Config file needed for model conversion.
        # Create a second copy for training resume
        write_hf_ckpt_config(ckpt_dir)
        write_hf_ckpt_config(tmpdir)

        # Train for two epochs
        cmd_1 = f"""
        tune run lora_finetune_single_device \
            --config llama3/8B_lora_single_device \
            batch_size=8 \
            gradient_accumulation_steps=1 \
            output_dir={tmpdir} \
            model.lora_attn_modules=['q_proj','v_proj','k_proj','output_proj'] \
            model.apply_lora_to_mlp=True \
            checkpointer=torchtune.training.FullModelTorchTuneCheckpointer \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[{ckpt_path}]\
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type=LLAMA3 \
            tokenizer.path=/tmp/test-artifacts/tokenizer_llama3.model \
            tokenizer.prompt_template=null \
            save_adapter_weights_only={save_adapter_weights_only} \
            enable_activation_checkpointing=True \
            enable_activation_offloading=False \
            enable_async_checkpointing=True \
        """.split()

        model_config = MODEL_TEST_CONFIGS["llama3_lora"]

        cmd_1 = cmd_1 + self._get_test_config_overrides() + model_config
        monkeypatch.setattr(sys, "argv", cmd_1)
        with pytest.raises(SystemExit, match=""):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        # Resume training
        shutil.rmtree(tmpdir / "epoch_1")

        cmd_2 = f"""
        tune run lora_finetune_single_device \
            --config llama3/8B_lora_single_device \
            batch_size=8 \
            gradient_accumulation_steps=1 \
            output_dir={tmpdir} \
            model.lora_attn_modules=['q_proj','v_proj','k_proj','output_proj'] \
            model.apply_lora_to_mlp=True \
            checkpointer=torchtune.training.FullModelTorchTuneCheckpointer \
            checkpointer.checkpoint_dir={ckpt_dir} \
            checkpointer.checkpoint_files=[{ckpt_path}]\
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type=LLAMA3 \
            resume_from_checkpoint=True \
            metric_logger.filename={log_file} \
            tokenizer.path=/tmp/test-artifacts/tokenizer_llama3.model \
            tokenizer.prompt_template=null \
            enable_activation_checkpointing=True \
            enable_activation_offloading=False \
            enable_async_checkpointing=True \
        """.split()
        cmd_2 = cmd_2 + self._get_test_config_overrides() + model_config
        monkeypatch.setattr(sys, "argv", cmd_2)
        with pytest.raises(SystemExit, match=""):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        # Second epoch only
        expected_loss_values = self._fetch_expected_loss_values("llama3")
        loss_values = get_loss_values_from_metric_logger(log_file)

        torch.testing.assert_close(
            loss_values, expected_loss_values, rtol=1e-5, atol=1e-5
        )

    @pytest.mark.parametrize(
        "epochs_to_save, expected_folders",
        [
            ("all", ["epoch_0", "epoch_1", "epoch_2"]),
            ("none", []),
            ("1,3", ["epoch_0", "epoch_2"]),
        ],
    )
    @pytest.mark.integration_test
    @gpu_test(gpu_count=1)
    def test_epochs_to_save(
        self, tmpdir, monkeypatch, epochs_to_save, expected_folders
    ):
        """Test that epochs_to_save parameter controls which epoch folders are saved.
        The test checks if the specified epochs are saved after training a model for 3 epochs.
        """

        ckpt = "llama3_tune"
        ckpt_path = Path(CKPT_MODEL_PATHS[ckpt])
        ckpt_dir = ckpt_path.parent
        log_file = gen_log_file_name(tmpdir)

        # Config file needed for model conversion.
        write_hf_ckpt_config(ckpt_dir)
        write_hf_ckpt_config(tmpdir)

        # Train for three epochs
        cmd = f"""
        tune run lora_finetune_single_device \
            --config llama3/8B_lora_single_device \
            batch_size=8 \
            gradient_accumulation_steps=1 \
            output_dir={tmpdir} \
            model.lora_attn_modules=['q_proj','v_proj','k_proj','output_proj'] \
            model.apply_lora_to_mlp=False \
            checkpointer=torchtune.training.FullModelTorchTuneCheckpointer \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[{ckpt_path}] \
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type=LLAMA3 \
            tokenizer.path=/tmp/test-artifacts/tokenizer_llama3.model \
            tokenizer.prompt_template=null \
            epochs_to_save={epochs_to_save} \
            save_last_epoch_only=False \
            enable_activation_checkpointing=True \
            enable_activation_offloading=False \
            enable_async_checkpointing=False \
        """.split()

        model_config = MODEL_TEST_CONFIGS["llama3_lora"]

        cmd = cmd + self._get_test_config_overrides(epochs=3) + model_config
        monkeypatch.setattr(sys, "argv", cmd)
        with pytest.raises(SystemExit, match=""):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        # Verify the checkpointing behavior
        # Check if the expected epoch folders are created
        saved_epoch_folders = sorted(
            [f for f in os.listdir(tmpdir) if f.startswith("epoch_")]
        )

        assert (
            saved_epoch_folders == expected_folders
        ), f"With epochs_to_save={epochs_to_save}, expected epoch folders {expected_folders}, got {saved_epoch_folders}"

    @pytest.mark.parametrize(
        "epochs_to_save, expected_folders",
        [
            ("all", ["epoch_0", "epoch_1", "epoch_2"]),
            ("none", []),
            ("1,3", ["epoch_0", "epoch_2"]),
        ],
    )
    @pytest.mark.integration_test
    @gpu_test(gpu_count=1)
    def test_epochs_to_save_with_async_checkpointing(
        self, tmpdir, monkeypatch, epochs_to_save, expected_folders
    ):
        """Test that epochs_to_save parameter controls which epoch folders are saved with async checkpointing.
        The test checks if the specified epochs are saved after training a model for 3 epochs.
        """

        ckpt = "llama3_tune"
        ckpt_path = Path(CKPT_MODEL_PATHS[ckpt])
        ckpt_dir = ckpt_path.parent
        log_file = gen_log_file_name(tmpdir)

        # Config file needed for model conversion.
        write_hf_ckpt_config(ckpt_dir)
        write_hf_ckpt_config(tmpdir)

        # Train for three epochs
        cmd = f"""
        tune run lora_finetune_single_device \
            --config llama3/8B_lora_single_device \
            batch_size=8 \
            gradient_accumulation_steps=1 \
            output_dir={tmpdir} \
            model.lora_attn_modules=['q_proj','v_proj','k_proj','output_proj'] \
            model.apply_lora_to_mlp=False \
            checkpointer=torchtune.training.FullModelTorchTuneCheckpointer \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[{ckpt_path}] \
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type=LLAMA3 \
            tokenizer.path=/tmp/test-artifacts/tokenizer_llama3.model \
            tokenizer.prompt_template=null \
            epochs_to_save={epochs_to_save} \
            save_last_epoch_only=False \
            enable_activation_checkpointing=True \
            enable_activation_offloading=False \
            enable_async_checkpointing=True \
        """.split()

        model_config = MODEL_TEST_CONFIGS["llama3_lora"]

        cmd = cmd + self._get_test_config_overrides(epochs=3) + model_config
        monkeypatch.setattr(sys, "argv", cmd)
        with pytest.raises(SystemExit, match=""):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        # Verify the checkpointing behavior
        # Check if the expected epoch folders are created
        saved_epoch_folders = sorted(
            [f for f in os.listdir(tmpdir) if f.startswith("epoch_")]
        )

        assert (
            saved_epoch_folders == expected_folders
        ), f"With epochs_to_save={epochs_to_save}, expected epoch folders {expected_folders}, got {saved_epoch_folders}"

    @pytest.mark.parametrize(
        "save_last_epoch_only, expected_folders",
        [
            (True, ["epoch_2"]),
            (False, ["epoch_0", "epoch_1"]),
        ],
    )
    @pytest.mark.integration_test
    @gpu_test(gpu_count=1)
    def test_save_last_epoch_only(
        self, tmpdir, monkeypatch, save_last_epoch_only, expected_folders
    ):
        """Test that save_last_epoch_only parameter controls checkpoint saving behavior.
        The test checks if the last epoch is saved when save_last_epoch_only is True
        after training a model for 3 epochs and if it correctly overrides epochs_to_save.
        """

        ckpt = "llama3_tune"
        ckpt_path = Path(CKPT_MODEL_PATHS[ckpt])
        ckpt_dir = ckpt_path.parent
        log_file = gen_log_file_name(tmpdir)

        # Config file needed for model conversion.
        write_hf_ckpt_config(ckpt_dir)
        write_hf_ckpt_config(tmpdir)

        # Train for three epochs
        cmd = f"""
        tune run lora_finetune_single_device \
            --config llama3/8B_lora_single_device \
            batch_size=8 \
            gradient_accumulation_steps=1 \
            output_dir={tmpdir} \
            model.lora_attn_modules=['q_proj','v_proj','k_proj','output_proj'] \
            model.apply_lora_to_mlp=False \
            checkpointer=torchtune.training.FullModelTorchTuneCheckpointer \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[{ckpt_path}] \
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type=LLAMA3 \
            tokenizer.path=/tmp/test-artifacts/tokenizer_llama3.model \
            tokenizer.prompt_template=null \
            epochs_to_save='1,2' \
            save_last_epoch_only={save_last_epoch_only} \
            enable_activation_checkpointing=True \
            enable_activation_offloading=False \
            enable_async_checkpointing=True \
        """.split()

        model_config = MODEL_TEST_CONFIGS["llama3_lora"]

        cmd = cmd + self._get_test_config_overrides(epochs=3) + model_config
        monkeypatch.setattr(sys, "argv", cmd)
        with pytest.raises(SystemExit, match=""):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        # Verify the checkpointing behavior
        # Check if the expected epoch folders are created
        saved_epoch_folders = sorted(
            [f for f in os.listdir(tmpdir) if f.startswith("epoch_")]
        )

        assert (
            saved_epoch_folders == expected_folders
        ), f"With save_last_epoch_only={save_last_epoch_only}, expected epoch folders {expected_folders}, got {saved_epoch_folders}"

    @pytest.mark.parametrize(
        "save_last_epoch_only, expected_folders",
        [
            (True, ["epoch_2"]),
            (False, ["epoch_0", "epoch_1"]),
        ],
    )
    @pytest.mark.integration_test
    @gpu_test(gpu_count=1)
    def test_save_last_epoch_only_with_async_checkpointing(
        self, tmpdir, monkeypatch, save_last_epoch_only, expected_folders
    ):
        """Test that save_last_epoch_only parameter controls checkpoint saving behavior with async checkpointing.
        The test checks if the last epoch is saved when save_last_epoch_only is True
        after training a model for 3 epochs and if it correctly overrides epochs_to_save.
        """

        ckpt = "llama3_tune"
        ckpt_path = Path(CKPT_MODEL_PATHS[ckpt])
        ckpt_dir = ckpt_path.parent
        log_file = gen_log_file_name(tmpdir)

        # Config file needed for model conversion.
        write_hf_ckpt_config(ckpt_dir)
        write_hf_ckpt_config(tmpdir)

        # Train for three epochs
        cmd = f"""
        tune run lora_finetune_single_device \
            --config llama3/8B_lora_single_device \
            batch_size=8 \
            gradient_accumulation_steps=1 \
            output_dir={tmpdir} \
            model.lora_attn_modules=['q_proj','v_proj','k_proj','output_proj'] \
            model.apply_lora_to_mlp=False \
            checkpointer=torchtune.training.FullModelTorchTuneCheckpointer \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[{ckpt_path}] \
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type=LLAMA3 \
            tokenizer.path=/tmp/test-artifacts/tokenizer_llama3.model \
            tokenizer.prompt_template=null \
            epochs_to_save='1,2' \
            save_last_epoch_only={save_last_epoch_only} \
            enable_activation_checkpointing=True \
            enable_activation_offloading=False \
            enable_async_checkpointing=True \
        """.split()

        model_config = MODEL_TEST_CONFIGS["llama3_lora"]

        cmd = cmd + self._get_test_config_overrides(epochs=3) + model_config
        monkeypatch.setattr(sys, "argv", cmd)
        with pytest.raises(SystemExit, match=""):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        # Verify the checkpointing behavior
        # Check if the expected epoch folders are created
        saved_epoch_folders = sorted(
            [f for f in os.listdir(tmpdir) if f.startswith("epoch_")]
        )

        assert (
            saved_epoch_folders == expected_folders
        ), f"With save_last_epoch_only={save_last_epoch_only}, expected epoch folders {expected_folders}, got {saved_epoch_folders}"

    @pytest.mark.parametrize("use_dora", [False, True])
    @pytest.mark.integration_test
    @gpu_test(gpu_count=1)
    def test_save_and_load_merged_weights(self, tmpdir, monkeypatch, use_dora):
        ckpt = "llama3_tune"
        ckpt_path = Path(CKPT_MODEL_PATHS[ckpt])
        ckpt_dir = ckpt_path.parent

        cmd = f"""
        tune run lora_finetune_single_device \
            --config llama3/8B_lora_single_device \
            output_dir={tmpdir} \
            model.lora_attn_modules=['q_proj','v_proj','k_proj','output_proj'] \
            model.apply_lora_to_mlp=True \
            checkpointer=torchtune.training.FullModelTorchTuneCheckpointer \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[{ckpt_path}]\
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type=LLAMA3 \
            tokenizer.path=/tmp/test-artifacts/tokenizer_llama3.model \
            tokenizer.prompt_template=null \
            enable_activation_checkpointing=True \
            enable_activation_offloading=False \
        """.split()

        if use_dora:
            model_config = MODEL_TEST_CONFIGS["llama3_dora"]
        else:
            model_config = MODEL_TEST_CONFIGS["llama3_lora"]

        cmd = cmd + self._get_test_config_overrides() + model_config
        monkeypatch.setattr(sys, "argv", cmd)
        with pytest.raises(SystemExit, match=""):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        # Next load both the merged weights in a Llama3 base model
        # and the base model weights + trained adapter weights in the LoRA Llama 2 model
        # The results of calling forward on dummy inputs should be the same.
        inputs = torch.randint(low=0, high=32_000, size=(2, 100))

        # Build LoRA model for loading base + adapter weights separately
        lora_model = config.instantiate(OmegaConf.from_dotlist(model_config).model)

        # Build base llama3 model for loading merged weights
        base_llama3_config = MODEL_TEST_CONFIGS["llama3"]
        llama3_model = config.instantiate(
            OmegaConf.from_dotlist(base_llama3_config).model
        )

        # Load base model and trained adapter weights into LoRA model and call fwd
        epoch_folder = get_largest_iter_folder(tmpdir)
        adpt_path = os.path.join(tmpdir, epoch_folder, f"{ADAPTER_MODEL_FNAME}.pt")
        lora_sd = safe_torch_load(adpt_path, weights_only=True)

        with open(ckpt_path, "rb") as f:
            base_model_sd = torch.load(f, weights_only=True)
        lora_model.load_state_dict(lora_sd, strict=False)
        lora_model.load_state_dict(base_model_sd, strict=False)
        baseline_out = lora_model(inputs)

        # Load merged final ckpt directly into llama3 and call fwd
        model_ckpt_fname = (
            SHARD_FNAME.format(cpt_idx="1".zfill(5), num_shards="1".zfill(5)) + ".bin"
        )
        model_path = os.path.join(tmpdir, epoch_folder, model_ckpt_fname)
        sd = safe_torch_load(model_path, weights_only=True)

        llama3_model.load_state_dict(sd)
        merged_ckpt_out = llama3_model(inputs)
        torch.testing.assert_close(baseline_out, merged_ckpt_out, rtol=1e-5, atol=1e-5)
