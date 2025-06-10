# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import logging
import runpy
import sys
from pathlib import Path

import pytest
import torch
from tests.common import TUNE_PATH

from tests.recipes.utils import (
    dummy_text_completion_alpaca_dataset_config,
    # llama2_classifier_test_config,
    # llama2_test_config,
    write_hf_ckpt_config,
    write_llama3_hf_ckpt_config,
    MODEL_TEST_CONFIGS
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


logger = logging.getLogger(__name__)


class TestPPOFullFinetuneSingleDeviceRecipe:
    def _get_test_config_overrides(self):
        return [
            "batch_size=4",
            "forward_batch_size=4",
            "ppo_batch_size=4",
            "ppo_epochs=1",
            "num_steps=16",
            "temperature=1.0",
            "gradient_accumulation_steps=1",
            "dtype=fp32",
            "enable_activation_checkpointing=False",
            "enable_activation_offloading=False",
            f"tokenizer.path={TOKENIZER_PATHS['llama3']}",
            "tokenizer._component_=torchtune.models.llama3.llama3_tokenizer",
            "tokenizer.prompt_template=null",
            "tokenizer.max_seq_len=64",
            "seed=9",
            "optimizer=torch.optim.AdamW",
            "optimizer.lr=2e-5",
            "lr_scheduler.num_warmup_steps=0",
            "lr_scheduler.num_cycles=0",
            "log_every_n_steps=1",
            "compile=False",
        ] + dummy_text_completion_alpaca_dataset_config()

    # Unfortunately we get different values on different hardware.
    # This is a hack to allow us to run CI on T4s/A10Gs and still run tests locally
    def _get_expected_loss_values(self, device_capability):
        if device_capability == (7, 5):
            return [
                1.0030436515808105,
                0.9150941967964172,
                0.8794946074485779,
                1.0626529455184937,
                0.964613676071167,
                0.980392575263977,
                1.0056356191635132,
                0.9202911853790283,
                0.8534448146820068,
                1.045704960823059,
                0.9574834704399109,
                0.8822144865989685,
            ]
        elif device_capability == (8, 6):
            return [
                1.1089695692062378,
                1.0091122388839722,
                0.9985737800598145,
                1.076175570487976,
                0.9825485348701477,
                0.9362708926200867,
                1.0785716772079468,
                0.9799201488494873,
                0.9865158200263977,
                1.0669920444488525,
                0.976087749004364,
                0.9090427756309509
            ]
        elif device_capability == (9, 0):
            return [
                1.0266655683517456,
                0.9376769661903381,
                0.8898855447769165,
                1.0626059770584106,
                0.966614842414856,
                0.9599114656448364,
                1.0275567770004272,
                0.9341378211975098,
                0.9341893196105957,
                1.0539714097976685,
                0.9588900208473206,
                0.950813889503479,
            ]
        else:
            raise ValueError("Unsupported device")

    @pytest.mark.integration_test
    @pytest.mark.skipif(
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability() not in ((7, 5), (8, 6), (9, 0)),
        reason="Unexpected device type",
    )
    @gpu_test(gpu_count=1)
    def test_loss(self, tmpdir, monkeypatch):
        reward_ckpt_path = Path(CKPT_MODEL_PATHS["llama3_reward_hf"])
        policy_ckpt_path = Path(CKPT_MODEL_PATHS["llama3_tune"])

        ckpt_dir = policy_ckpt_path.parent
        log_file = gen_log_file_name(tmpdir)
        policy_tmpdir = (tmpdir / "policy").mkdir()
        value_tmpdir = (tmpdir / "value").mkdir()

        write_llama3_hf_ckpt_config(ckpt_dir)
        cmd_1 = f"""
        tune run ppo_full_finetune_single_device \
            --config mistral/7B_full_ppo_low_memory \
            output_dir={tmpdir} \
            checkpointer._component_=torchtune.training.FullModelTorchTuneCheckpointer \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[{policy_ckpt_path}]\
            checkpointer.output_dir={policy_tmpdir} \
            checkpointer.model_type=LLAMA3 \

            ref_policy_checkpointer._component_=torchtune.training.FullModelTorchTuneCheckpointer \
            ref_policy_checkpointer.checkpoint_dir='{ckpt_dir}' \
            ref_policy_checkpointer.checkpoint_files=[{policy_ckpt_path}]\
            ref_policy_checkpointer.model_type=LLAMA3 \

            value_checkpointer.checkpoint_dir='{ckpt_dir}' \
            value_checkpointer.checkpoint_files=[{reward_ckpt_path}]\
            value_checkpointer.output_dir={value_tmpdir} \

            reward_checkpointer.checkpoint_dir='{ckpt_dir}' \
            reward_checkpointer.checkpoint_files=[{reward_ckpt_path}]\

            metric_logger._component_=torchtune.training.metric_logging.DiskLogger \
            metric_logger.filename={log_file} \
        """.split()

        model_config = MODEL_TEST_CONFIGS["llama3"]
        model_config = [k.replace("model.", "policy_model.") for k in model_config]

        reward_and_value_model_config = MODEL_TEST_CONFIGS["llama3_classifier"]
        reward_and_value_model_config = [
            k.replace("model.", "reward_and_value_model.")
            for k in reward_and_value_model_config
        ]
        cmd_1 = (
            cmd_1
            + self._get_test_config_overrides()
            + model_config
            + reward_and_value_model_config
        )

        monkeypatch.setattr(sys, "argv", cmd_1)
        with pytest.raises(SystemExit, match=""):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        loss_values = get_loss_values_from_metric_logger(log_file)

        logger.error(f"Loss values: {loss_values}")

        expected_loss_values = self._get_expected_loss_values(
            torch.cuda.get_device_capability()
        )
        torch.testing.assert_close(
            loss_values, expected_loss_values, atol=1e-4, rtol=1e-5
        )

    @pytest.mark.integration_test
    @gpu_test(gpu_count=1)
    def test_training_state_on_resume(self, tmpdir, monkeypatch):
        """Test whether the recipe state correctly saved and restored after training."""

        reward_ckpt_path = Path(CKPT_MODEL_PATHS["llama3_reward_hf"])
        policy_ckpt_path = Path(CKPT_MODEL_PATHS["llama3_tune"])

        ckpt_dir = policy_ckpt_path.parent
        log_file = gen_log_file_name(tmpdir)
        policy_tmpdir = (tmpdir / "policy").mkdir()
        value_tmpdir = (tmpdir / "value").mkdir()

        # Config file needed for model conversion.
        # Create a second copy for training resume
        write_llama3_hf_ckpt_config(ckpt_dir)
        write_llama3_hf_ckpt_config(policy_tmpdir)
        write_llama3_hf_ckpt_config(value_tmpdir)

        # There are 4 steps in total (num_steps / batch size)
        # and the dataset has 8 samples, so each epoch will be 2 batches
        # a single step is a single batch update, and we checkpoint at every epoch (2 steps)
        # so we're expecting an intermediate checkpoint at step 2. The idea here is to train for 4 steps,
        # resume after 2, and ensure the losses for the final two steps after resuming are identical
        cmd_1 = f"""
        tune run ppo_full_finetune_single_device \
            --config mistral/7B_full_ppo_low_memory \
            output_dir={tmpdir} \
            checkpointer._component_=torchtune.training.FullModelTorchTuneCheckpointer \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[{policy_ckpt_path}]\
            checkpointer.output_dir={policy_tmpdir} \
            checkpointer.model_type=LLAMA2 \

            ref_policy_checkpointer._component_=torchtune.training.FullModelTorchTuneCheckpointer \
            ref_policy_checkpointer.checkpoint_dir='{ckpt_dir}' \
            ref_policy_checkpointer.checkpoint_files=[{policy_ckpt_path}]\
            ref_policy_checkpointer.model_type=LLAMA3 \

            value_checkpointer.checkpoint_dir='{ckpt_dir}' \
            value_checkpointer.checkpoint_files=[{reward_ckpt_path}]\
            value_checkpointer.output_dir={value_tmpdir} \

            reward_checkpointer.checkpoint_dir='{ckpt_dir}' \
            reward_checkpointer.checkpoint_files=[{reward_ckpt_path}]\

            metric_logger._component_=torchtune.training.metric_logging.DiskLogger \
            metric_logger.filename={log_file} \
        """.split()

        model_config = MODEL_TEST_CONFIGS["llama3"]
        model_config = [k.replace("model.", "policy_model.") for k in model_config]

        reward_and_value_model_config = MODEL_TEST_CONFIGS["llama3_classifier"]
        reward_and_value_model_config = [
            k.replace("model.", "reward_and_value_model.")
            for k in reward_and_value_model_config
        ]
        cmd_1 = (
            cmd_1
            + self._get_test_config_overrides()
            + model_config
            + reward_and_value_model_config
        )

        monkeypatch.setattr(sys, "argv", cmd_1)
        with pytest.raises(SystemExit, match=""):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        loss_values = get_loss_values_from_metric_logger(log_file)

        # Resume training at step 2
        resumed_log_dir = (tmpdir / "resumed/").mkdir()
        resumed_log_file = gen_log_file_name(resumed_log_dir)

        epoch_folder = get_largest_iter_folder(value_tmpdir)
        epoch_folder_minus_one = f"epoch_{int(epoch_folder.split('_')[-1]) - 1}"
        suffix = ".safetensors"
        model_ckpt_fname = (
            SHARD_FNAME.format(cpt_idx="1".zfill(5), num_shards="1".zfill(5)) + suffix
        )
        cmd_2 = f"""
        tune run ppo_full_finetune_single_device \
            --config mistral/7B_full_ppo_low_memory \
            output_dir={tmpdir} \
            checkpointer._component_=torchtune.training.FullModelTorchTuneCheckpointer \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[{os.path.join(epoch_folder_minus_one, model_ckpt_fname)}]\
            checkpointer.recipe_checkpoint={os.path.join(RECIPE_STATE_DIRNAME, "recipe_state.pt")}\
            checkpointer.output_dir={policy_tmpdir} \
            checkpointer.model_type=LLAMA3 \

            ref_policy_checkpointer._component_=torchtune.training.FullModelTorchTuneCheckpointer \
            ref_policy_checkpointer.checkpoint_dir='{ckpt_dir}' \
            ref_policy_checkpointer.checkpoint_files=[{policy_ckpt_path}]\
            ref_policy_checkpointer.model_type=LLAMA3 \

            value_checkpointer.checkpoint_dir='{ckpt_dir}' \
            value_checkpointer.checkpoint_files=[{os.path.join(value_tmpdir, epoch_folder_minus_one, model_ckpt_fname)}]\
            value_checkpointer.output_dir={value_tmpdir} \

            reward_checkpointer.checkpoint_dir='{ckpt_dir}' \
            reward_checkpointer.checkpoint_files=[{reward_ckpt_path}]\

            resume_from_checkpoint=True \
            metric_logger._component_=torchtune.training.metric_logging.DiskLogger \
            metric_logger.filename={resumed_log_file} \
        """.split()

        cmd_2 = (
            cmd_2
            + self._get_test_config_overrides()
            + model_config
            + reward_and_value_model_config
        )

        monkeypatch.setattr(sys, "argv", cmd_2)
        with pytest.raises(SystemExit, match=""):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        resumed_loss_values = get_loss_values_from_metric_logger(resumed_log_file)

        # losses at each step are (loss, policy_loss, value_loss)
        torch.testing.assert_close(
            loss_values[6:], resumed_loss_values, rtol=1e-4, atol=1e-4
        )

    @pytest.mark.integration_test
    @gpu_test(gpu_count=1)
    def test_training_state_on_resume_with_optimizer_in_bwd(self, tmpdir, monkeypatch):
        """Test whether the recipe state correctly saves and restores optimizer state
        when using ``optimizer_in_bwd``, since the optimizer checkpoint dict will include
        parameters for two models.

        This is identical to ``test_training_state_on_resume``, but adds optimizer_in_bwd.
        """

        reward_ckpt_path = Path(CKPT_MODEL_PATHS["llama3_reward_hf"])
        policy_ckpt_path = Path(CKPT_MODEL_PATHS["llama3_tune"])

        ckpt_dir = policy_ckpt_path.parent
        log_file = gen_log_file_name(tmpdir)
        policy_tmpdir = (tmpdir / "policy").mkdir()
        value_tmpdir = (tmpdir / "value").mkdir()

        # Config file needed for model conversion.
        # Create a second copy for training resume
        write_llama3_hf_ckpt_config(ckpt_dir)
        write_llama3_hf_ckpt_config(policy_tmpdir)
        write_llama3_hf_ckpt_config(value_tmpdir)
        cmd_1 = f"""
        tune run ppo_full_finetune_single_device \
            --config mistral/7B_full_ppo_low_memory \
            output_dir={tmpdir} \
            checkpointer._component_=torchtune.training.FullModelTorchTuneCheckpointer \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[{policy_ckpt_path}]\
            checkpointer.output_dir={policy_tmpdir} \
            
            checkpointer.model_type=LLAMA3 \
            ref_policy_checkpointer._component_=torchtune.training.FullModelTorchTuneCheckpointer \
            ref_policy_checkpointer.checkpoint_dir='{ckpt_dir}' \
            ref_policy_checkpointer.checkpoint_files=[{policy_ckpt_path}]\
            ref_policy_checkpointer.model_type=LLAMA3 \

            value_checkpointer.checkpoint_dir='{ckpt_dir}' \
            value_checkpointer.checkpoint_files=[{reward_ckpt_path}]\
            value_checkpointer.output_dir={value_tmpdir} \

            reward_checkpointer.checkpoint_dir='{ckpt_dir}' \
            reward_checkpointer.checkpoint_files=[{reward_ckpt_path}]\

            metric_logger._component_=torchtune.training.metric_logging.DiskLogger \
            metric_logger.filename={log_file} \

            optimizer_in_bwd=True
        """.split()

        model_config = MODEL_TEST_CONFIGS["llama3"]
        model_config = [k.replace("model.", "policy_model.") for k in model_config]

        reward_and_value_model_config = MODEL_TEST_CONFIGS["llama3_classifier"]
        reward_and_value_model_config = [
            k.replace("model.", "reward_and_value_model.")
            for k in reward_and_value_model_config
        ]

        cmd_1 = (
            cmd_1
            + self._get_test_config_overrides()
            + model_config
            + reward_and_value_model_config
        )

        monkeypatch.setattr(sys, "argv", cmd_1)
        with pytest.raises(SystemExit, match=""):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        loss_values = get_loss_values_from_metric_logger(log_file)

        # Resume training at step 2
        resumed_log_dir = (tmpdir / "resumed/").mkdir()
        resumed_log_file = gen_log_file_name(resumed_log_dir)

        epoch_folder = get_largest_iter_folder(value_tmpdir)
        epoch_folder_minus_one = f"epoch_{int(epoch_folder.split('_')[-1]) - 1}"
        suffix = ".safetensors"
        model_ckpt_fname = (
            SHARD_FNAME.format(cpt_idx="1".zfill(5), num_shards="1".zfill(5)) + suffix
        )
        cmd_2 = f"""
        tune run ppo_full_finetune_single_device \
            --config mistral/7B_full_ppo_low_memory \
            output_dir={tmpdir} \
            checkpointer._component_=torchtune.training.FullModelHFCheckpointer \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[{os.path.join(epoch_folder_minus_one, model_ckpt_fname)}]\
            checkpointer.recipe_checkpoint={os.path.join(RECIPE_STATE_DIRNAME, "recipe_state.pt")}\
            checkpointer.output_dir={policy_tmpdir} \
            checkpointer.model_type=LLAMA3 \

            ref_policy_checkpointer._component_=torchtune.training.FullModelTorchTuneCheckpointer \
            ref_policy_checkpointer.checkpoint_dir='{ckpt_dir}' \
            ref_policy_checkpointer.checkpoint_files=[{policy_ckpt_path}]\
            ref_policy_checkpointer.model_type=LLAMA3 \

            value_checkpointer.checkpoint_dir='{ckpt_dir}' \
            value_checkpointer.checkpoint_files=[{os.path.join(value_tmpdir, epoch_folder_minus_one, model_ckpt_fname)}]\
            value_checkpointer.output_dir={value_tmpdir} \

            reward_checkpointer.checkpoint_dir='{ckpt_dir}' \
            reward_checkpointer.checkpoint_files=[{reward_ckpt_path}]\

            resume_from_checkpoint=True \
            metric_logger._component_=torchtune.training.metric_logging.DiskLogger \
            metric_logger.filename={resumed_log_file} \

            optimizer_in_bwd=True
        """.split()

        cmd_2 = (
            cmd_2
            + self._get_test_config_overrides()
            + model_config
            + reward_and_value_model_config
        )

        monkeypatch.setattr(sys, "argv", cmd_2)
        with pytest.raises(SystemExit, match=""):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        resumed_loss_values = get_loss_values_from_metric_logger(resumed_log_file)

        # losses at each step are (loss, policy_loss, value_loss)
        torch.testing.assert_close(
            loss_values[6:], resumed_loss_values, rtol=1e-4, atol=1e-4
        )
