# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import re
import runpy
import sys
from pathlib import Path

import pytest

import ray
import torch

from tests.common import TUNE_PATH
from tests.recipes.utils import MODEL_TEST_CONFIGS, write_hf_ckpt_config
from tests.test_utils import (
    CKPT_MODEL_PATHS,
    gen_log_file_name,
    get_loss_values_from_metric_logger,
    gpu_test,
    TOKENIZER_PATHS,
)


def get_reward_values_from_metric_logger(log_file_path: str) -> list[float]:
    """
    Given an output directory containing metric logger .txt file,
    parse the .txt and return a list of reward values from each logged iteration.
    """
    with open(log_file_path, "r") as f:
        logs = f.read()
    rewards = [float(x) for x in re.findall(r"rewards:(\d+\.\d+)", logs)]
    return rewards


class TestAsyncGRPOFullFinetuneDistributedRecipe:
    """Test suite for the async_grpo_full_finetune_distributed recipe."""

    def _get_test_config_overrides(self, tmpdir):
        """Get CLI overrides for testing the async GRPO recipe."""
        log_file = gen_log_file_name(tmpdir)

        # Basic overrides for testing
        overrides = [
            f"output_dir={tmpdir}",
            f"name=test_async_grpo_qwen",
            "dtype=fp32",
            "seed=42",
            # Configure for 4 GPUs
            "vllm.num_workers=1",
            "vllm.tp_size=1",
            "num_ref_workers=1",  # Use 1 reference worker
            "num_fsdp_workers=2",  # 2 FSDP worker
            "vllm.queue_maxsize=1",
            "total_inference_batch_size=2",
            "replay_buffer_size=2",
            "batch_size=2",
            "grpo_samples=2",
            # Reduce computation for testing
            "max_generated_tokens=32",
            "num_steps=2",
            "epochs=1",
            "steps_before_sync=1",
            # Use a mock base model path for testing
            "base_model_path=/tmp/test-artifacts",
            # Logging
            "metric_logger._component_=torchtune.training.metric_logging.DiskLogger",
            f"metric_logger.filename={log_file}",
            "log_every_n_steps=1",
            # Memory management
            "enable_activation_checkpointing=False",
            "enable_activation_offloading=False",
            # Mock tokenizer for testing - remove any merges_file parameter
            "tokenizer._component_=torchtune.models.llama3.llama3_tokenizer",
            f"tokenizer.path={TOKENIZER_PATHS['llama3']}",
            "tokenizer.max_seq_len=512",
            # Override the original config's tokenizer parameters that might cause issues
            "~tokenizer.merges_file",
            # Use a dataset that provides tokens and answers
            "dataset._component_=torchtune.dev.grpo.gsm8k.gsm8k_dataset",
            "dataset.partition=1-9/10",
            # Mock model for testing
            "model._component_=torchtune.models.llama3.llama3",
        ]

        # Add model-specific configs
        overrides.extend(MODEL_TEST_CONFIGS["llama3"])

        # Mock checkpointers for testing
        ckpt_path = Path(CKPT_MODEL_PATHS["llama3_tune"])
        ckpt_dir = ckpt_path.parent

        overrides.extend(
            [
                "checkpointer._component_=torchtune.training.FullModelTorchTuneCheckpointer",
                f"checkpointer.checkpoint_dir={ckpt_dir}",
                f"checkpointer.checkpoint_files=[{ckpt_path}]",
                f"checkpointer.output_dir={tmpdir}",
                "checkpointer.model_type=LLAMA3",
                "ref_checkpointer._component_=torchtune.training.FullModelTorchTuneCheckpointer",
                f"ref_checkpointer.checkpoint_dir={ckpt_dir}",
                f"ref_checkpointer.checkpoint_files=[{ckpt_path}]",
                f"ref_checkpointer.output_dir={tmpdir}/ref",
                "ref_checkpointer.model_type=LLAMA3",
            ]
        )

        return overrides

    @pytest.mark.integration_test
    @gpu_test(gpu_count=4)
    def test_basic_run(self, tmpdir, monkeypatch):
        """Test that the recipe runs without errors with minimal configuration."""
        # Config file needed for model conversion
        write_hf_ckpt_config(Path(CKPT_MODEL_PATHS["llama3_tune"]).parent)

        # Set up command to run the recipe with the default config and overrides
        cmd = [
            "tune",
            "run",
            "dev/async_grpo_full_finetune_distributed",
            "--config",
            "recipes/configs/dev/qwen3B_async_grpo.yaml",
        ]

        # Add CLI overrides
        cmd.extend(self._get_test_config_overrides(tmpdir))
        try:
            monkeypatch.setattr(sys, "argv", cmd)
            runpy.run_path(TUNE_PATH, run_name="__main__")
            # If we get here, the recipe ran without errors
            # assert True
        finally:
            # Make sure Ray is shut down even if the test fails
            if "ray" in sys.modules and ray.is_initialized():
                ray.shutdown()

    @pytest.mark.integration_test
    @gpu_test(gpu_count=4)
    def test_loss_and_reward_values(self, tmpdir, monkeypatch):
        """Test that the recipe produces expected loss and reward values."""
        # Config file needed for model conversion
        write_hf_ckpt_config(Path(CKPT_MODEL_PATHS["llama3_tune"]).parent)

        # Create log file
        log_file = gen_log_file_name(tmpdir)

        # Get basic overrides
        overrides = self._get_test_config_overrides(tmpdir)

        # Add specific overrides for consistent loss/reward testing
        test_overrides = [
            f"metric_logger.filename={log_file}",
            "seed=42",  # Ensure deterministic results
            "num_steps=4",  # Run for more steps to get more metrics
            "log_every_n_steps=1",  # Log every step
        ]

        # Set up command to run the recipe with the default config and overrides
        cmd = [
            "tune",
            "run",
            "dev/async_grpo_full_finetune_distributed",
            "--config",
            "recipes/configs/dev/qwen3B_async_grpo.yaml",
        ]

        # Add all overrides
        cmd.extend(overrides)
        cmd.extend(test_overrides)

        monkeypatch.setattr(sys, "argv", cmd)

        try:
            # Run the recipe
            runpy.run_path(TUNE_PATH, run_name="__main__")

            # Extract loss and reward values from the metric logger
            loss_values = get_loss_values_from_metric_logger(log_file)
            reward_values = get_reward_values_from_metric_logger(log_file)

            # Expected values (placeholder values - user will fill these in)
            expected_loss_values = [10.5, 10.2, 9.8, 9.5]
            expected_reward_values = [0.2, 0.3, 0.4, 0.5]

            # Check that we have the expected number of values
            assert len(loss_values) > 0, "No loss values found in the log file"
            assert len(reward_values) > 0, "No reward values found in the log file"

            # Compare with expected values
            # Note: Using a high tolerance since these are placeholder values
            torch.testing.assert_close(
                torch.tensor(loss_values),
                torch.tensor(expected_loss_values[: len(loss_values)]),
                rtol=1e-1,
                atol=1e-1,
                msg=f"Loss values don't match expected values. Got {loss_values}, expected {expected_loss_values[:len(loss_values)]}",
            )

            torch.testing.assert_close(
                torch.tensor(reward_values),
                torch.tensor(expected_reward_values[: len(reward_values)]),
                rtol=1e-1,
                atol=1e-1,
                msg=f"Reward values don't match expected values. Got {reward_values}, expected {expected_reward_values[:len(reward_values)]}",
            )
        finally:
            # Make sure Ray is shut down even if the test fails
            if "ray" in sys.modules and ray.is_initialized():
                ray.shutdown()

    @pytest.mark.integration_test
    @gpu_test(gpu_count=4)
    def test_with_custom_gpu_allocation(self, tmpdir, monkeypatch):
        """Test that the recipe runs with custom GPU allocation."""
        # Config file needed for model conversion
        write_hf_ckpt_config(Path(CKPT_MODEL_PATHS["llama3_tune"]).parent)

        # Get basic overrides
        overrides = self._get_test_config_overrides(tmpdir)

        # Add custom GPU allocation overrides
        custom_overrides = [
            "vllm.num_workers=2",
            "vllm.tp_size=2",
            "num_ref_workers=1",
            "num_fsdp_workers=1",
            "total_inference_batch_size=1",
        ]

        # Set up command to run the recipe with the default config and overrides
        cmd = [
            "tune",
            "run",
            "dev/async_grpo_full_finetune_distributed",
            "--config",
            "recipes/configs/dev/qwen3B_async_grpo.yaml",
        ]

        # Add all overrides
        cmd.extend(overrides)
        cmd.extend(custom_overrides)

        monkeypatch.setattr(sys, "argv", cmd)

        # We're not checking specific loss values here, just that the recipe runs without errors
        try:
            runpy.run_path(TUNE_PATH, run_name="__main__")
            # If we get here, the recipe ran without errors
            assert True
        except Exception as e:
            pytest.fail(f"Recipe failed to run: {e}")
        finally:
            # Make sure Ray is shut down even if the test fails
            if "ray" in sys.modules and ray.is_initialized():
                ray.shutdown()
