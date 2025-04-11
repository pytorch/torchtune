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
    dummy_text_completion_alpaca_dataset_config,
    llama2_classifier_test_config,
    llama2_test_config,
    write_hf_ckpt_config,
)
from tests.test_utils import (
    CKPT_MODEL_PATHS,
    gen_log_file_name,
    get_loss_values_from_metric_logger,
    mps_ignored_test,
)

from torchtune.training.checkpointing._utils import (
    get_largest_iter_folder,
    RECIPE_STATE_DIRNAME,
    SHARD_FNAME,
)


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
            "device=cpu",
            "dtype=fp32",
            "enable_activation_checkpointing=False",
            "enable_activation_offloading=False",
            "tokenizer.path=/tmp/test-artifacts/tokenizer.model",
            "tokenizer._component_=torchtune.models.llama2.llama2_tokenizer",
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

    @pytest.mark.integration_test
    @mps_ignored_test()
    def test_loss(self, tmpdir, monkeypatch):

        reward_ckpt = "llama2_reward_hf"
        policy_ckpt = "llama2_hf"
        reward_ckpt_path = Path(CKPT_MODEL_PATHS[reward_ckpt])
        policy_ckpt_path = Path(CKPT_MODEL_PATHS[policy_ckpt])

        ckpt_dir = policy_ckpt_path.parent
        log_file = gen_log_file_name(tmpdir)
        policy_tmpdir = (tmpdir / "policy").mkdir()
        value_tmpdir = (tmpdir / "value").mkdir()

        write_hf_ckpt_config(ckpt_dir)
        cmd_1 = f"""
        tune run ppo_full_finetune_single_device \
            --config mistral/7B_full_ppo_low_memory \
            output_dir={tmpdir} \
            checkpointer._component_=torchtune.training.FullModelHFCheckpointer \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[{policy_ckpt_path}]\
            checkpointer.output_dir={policy_tmpdir} \
            checkpointer.model_type=LLAMA2 \

            ref_policy_checkpointer.checkpoint_dir='{ckpt_dir}' \
            ref_policy_checkpointer.checkpoint_files=[{policy_ckpt_path}]\

            value_checkpointer.checkpoint_dir='{ckpt_dir}' \
            value_checkpointer.checkpoint_files=[{reward_ckpt_path}]\
            value_checkpointer.output_dir={value_tmpdir} \

            reward_checkpointer.checkpoint_dir='{ckpt_dir}' \
            reward_checkpointer.checkpoint_files=[{reward_ckpt_path}]\

            metric_logger._component_=torchtune.training.metric_logging.DiskLogger \
            metric_logger.filename={log_file} \
        """.split()

        model_config = llama2_test_config()
        model_config = [k.replace("model.", "policy_model.") for k in model_config]
        model_config += ["policy_model.intermediate_dim=null"]

        reward_and_value_model_config = llama2_classifier_test_config()
        reward_and_value_model_config = [
            k.replace("model.", "reward_and_value_model.")
            for k in reward_and_value_model_config
        ]
        reward_and_value_model_config += [
            "reward_and_value_model.intermediate_dim=null"
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
        expected_loss_values = [
            1.0403811931610107,
            0.9495404958724976,
            0.90840744972229,
            1.0494484901428223,
            0.9609842896461487,
            0.8846424221992493,
            1.0282150506973267,
            0.939055323600769,
            0.891597330570221,
            1.0166869163513184,
            0.9231593608856201,
            0.9352758526802063,
        ]

        torch.testing.assert_close(
            loss_values, expected_loss_values, atol=1e-4, rtol=1e-5
        )

    @pytest.mark.integration_test
    def test_training_state_on_resume(self, tmpdir, monkeypatch):
        """Test whether the recipe state correctly saved and restored after training."""

        reward_ckpt = "llama2_reward_hf"
        policy_ckpt = "llama2_hf"
        reward_ckpt_path = Path(CKPT_MODEL_PATHS[reward_ckpt])
        policy_ckpt_path = Path(CKPT_MODEL_PATHS[policy_ckpt])

        ckpt_dir = policy_ckpt_path.parent
        log_file = gen_log_file_name(tmpdir)
        policy_tmpdir = (tmpdir / "policy").mkdir()
        value_tmpdir = (tmpdir / "value").mkdir()

        # Config file needed for model conversion.
        # Create a second copy for training resume
        write_hf_ckpt_config(ckpt_dir)
        write_hf_ckpt_config(policy_tmpdir)
        write_hf_ckpt_config(value_tmpdir)

        # There are 4 steps in total (num_steps / batch size)
        # and the dataset has 8 samples, so each epoch will be 2 batches
        # a single step is a single batch update, and we checkpoint at every epoch (2 steps)
        # so we're expecting an intermediate checkpoint at step 2. The idea here is to train for 4 steps,
        # resume after 2, and ensure the losses for the final two steps after resuming are identical
        cmd_1 = f"""
        tune run ppo_full_finetune_single_device \
            --config mistral/7B_full_ppo_low_memory \
            output_dir={tmpdir} \
            checkpointer._component_=torchtune.training.FullModelHFCheckpointer \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[{policy_ckpt_path}]\
            checkpointer.output_dir={policy_tmpdir} \
            checkpointer.model_type=LLAMA2 \

            ref_policy_checkpointer.checkpoint_dir='{ckpt_dir}' \
            ref_policy_checkpointer.checkpoint_files=[{policy_ckpt_path}]\

            value_checkpointer.checkpoint_dir='{ckpt_dir}' \
            value_checkpointer.checkpoint_files=[{reward_ckpt_path}]\
            value_checkpointer.output_dir={value_tmpdir} \

            reward_checkpointer.checkpoint_dir='{ckpt_dir}' \
            reward_checkpointer.checkpoint_files=[{reward_ckpt_path}]\

            metric_logger._component_=torchtune.training.metric_logging.DiskLogger \
            metric_logger.filename={log_file} \
        """.split()

        model_config = llama2_test_config()
        model_config = [k.replace("model.", "policy_model.") for k in model_config]
        model_config += ["policy_model.intermediate_dim=null"]

        reward_and_value_model_config = llama2_classifier_test_config()
        reward_and_value_model_config = [
            k.replace("model.", "reward_and_value_model.")
            for k in reward_and_value_model_config
        ]
        reward_and_value_model_config += [
            "reward_and_value_model.intermediate_dim=null"
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
            checkpointer.model_type=LLAMA2 \

            ref_policy_checkpointer.checkpoint_dir='{ckpt_dir}' \
            ref_policy_checkpointer.checkpoint_files=[{policy_ckpt_path}]\

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
    def test_training_state_on_resume_with_optimizer_in_bwd(self, tmpdir, monkeypatch):
        """Test whether the recipe state correctly saves and restores optimizer state
        when using ``optimizer_in_bwd``, since the optimizer checkpoint dict will include
        parameters for two models.

        This is identical to ``test_training_state_on_resume``, but adds optimizer_in_bwd.
        """

        reward_ckpt = "llama2_reward_hf"
        policy_ckpt = "llama2_hf"
        reward_ckpt_path = Path(CKPT_MODEL_PATHS[reward_ckpt])
        policy_ckpt_path = Path(CKPT_MODEL_PATHS[policy_ckpt])

        ckpt_dir = policy_ckpt_path.parent
        log_file = gen_log_file_name(tmpdir)
        policy_tmpdir = (tmpdir / "policy").mkdir()
        value_tmpdir = (tmpdir / "value").mkdir()

        # Config file needed for model conversion.
        # Create a second copy for training resume
        write_hf_ckpt_config(ckpt_dir)
        write_hf_ckpt_config(policy_tmpdir)
        write_hf_ckpt_config(value_tmpdir)
        cmd_1 = f"""
        tune run ppo_full_finetune_single_device \
            --config mistral/7B_full_ppo_low_memory \
            output_dir={tmpdir} \
            checkpointer._component_=torchtune.training.FullModelHFCheckpointer \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[{policy_ckpt_path}]\
            checkpointer.output_dir={policy_tmpdir} \
            checkpointer.model_type=LLAMA2 \

            ref_policy_checkpointer.checkpoint_dir='{ckpt_dir}' \
            ref_policy_checkpointer.checkpoint_files=[{policy_ckpt_path}]\

            value_checkpointer.checkpoint_dir='{ckpt_dir}' \
            value_checkpointer.checkpoint_files=[{reward_ckpt_path}]\
            value_checkpointer.output_dir={value_tmpdir} \

            reward_checkpointer.checkpoint_dir='{ckpt_dir}' \
            reward_checkpointer.checkpoint_files=[{reward_ckpt_path}]\

            metric_logger._component_=torchtune.training.metric_logging.DiskLogger \
            metric_logger.filename={log_file} \

            optimizer_in_bwd=True
        """.split()

        model_config = llama2_test_config()
        model_config = [k.replace("model.", "policy_model.") for k in model_config]
        model_config += ["policy_model.intermediate_dim=null"]

        reward_and_value_model_config = llama2_classifier_test_config()
        reward_and_value_model_config = [
            k.replace("model.", "reward_and_value_model.")
            for k in reward_and_value_model_config
        ]
        reward_and_value_model_config += [
            "reward_and_value_model.intermediate_dim=null"
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
            checkpointer.model_type=LLAMA2 \

            ref_policy_checkpointer.checkpoint_dir='{ckpt_dir}' \
            ref_policy_checkpointer.checkpoint_files=[{policy_ckpt_path}]\

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
