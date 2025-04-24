# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import runpy
import sys
from pathlib import Path

import pytest

from tests.common import TUNE_PATH
from tests.recipes.utils import MODEL_TEST_CONFIGS
from tests.test_utils import (
    CKPT_MODEL_PATHS,
    gen_log_file_name,
    get_loss_values_from_metric_logger,
    gpu_test,
    TOKENIZER_PATHS,
)


class TestAsyncGRPOFullFinetuneDistributedRecipe:
    """Test suite for the async_grpo_full_finetune_distributed recipe."""

    def _get_test_config_overrides(self, output_dir, log_file):
        """Get CLI overrides for testing the async GRPO recipe."""
        # Basic overrides for testing
        overrides = [
            f"output_dir={output_dir}",
            "name=test_async_grpo_qwen",
            "dtype=fp32",
            "seed=42",
            # Configure for 4 GPUs
            "num_rollout_workers=1",  # 1 vLLM worker
            "rollout_tensor_parallel_dim=1",
            "num_ref_workers=1",  # Use 1 reference model worker
            "num_trainer_workers=2",  # 2 FSDP trainer workers
            "rollout_queue_maxsize=1",
            "total_inference_batch_size=2",
            "replay_buffer_size=2",
            "batch_size=2",
            "grpo_samples=2",
            # Reduce computation for testing
            "max_generated_tokens=32",
            "num_steps=2",
            "save_every_n_steps=2",
            "num_train_steps_before_sync=1",
            # Logging
            "metric_logger._component_=torchtune.training.metric_logging.DiskLogger",
            f"metric_logger.filename={log_file}",
        ]
        return overrides

    @pytest.mark.integration_test
    @gpu_test(gpu_count=4)
    def test_basic_run(self, tmpdir, monkeypatch):
        """Test that the recipe runs without errors with minimal configuration."""
        import ray

        # Set up command to run the recipe with the default config
        cmd = [
            "tune",
            "run",
            "dev/async_grpo_full_finetune_distributed",
            "--config",
            "recipes/configs/dev/qwen3B_async_grpo.yaml",
        ]
        # Add generic CLI overrides
        log_file = gen_log_file_name(tmpdir)
        cmd.extend(
            self._get_test_config_overrides(output_dir=tmpdir, log_file=log_file)
        )
        # Add model-specific overrides
        model = "llama3_137M"
        cmd.extend(MODEL_TEST_CONFIGS[model])
        model_ckpt = CKPT_MODEL_PATHS[model]
        cmd.extend([f"base_model_path={Path(model_ckpt).parent}"])
        # Add tokenizer overrides
        tokenizer_model = "llama3"
        cmd.extend(
            [
                "tokenizer._component_=torchtune.models.llama3.llama3_tokenizer",
                f"tokenizer.path={TOKENIZER_PATHS[tokenizer_model]}",
                "tokenizer.max_seq_len=1024",
                # Override the original config's tokenizer parameters that might cause issues
                "~tokenizer.merges_file",
            ]
        )
        # Add checkpointer overrides
        cmd.extend(
            [
                "checkpointer._component_=torchtune.training.FullModelHFCheckpointer",
                f"checkpointer.checkpoint_files=[{model_ckpt}]",
                f"checkpointer.output_dir={tmpdir}",
                "checkpointer.model_type=LLAMA3",
                "ref_checkpointer._component_=torchtune.training.FullModelHFCheckpointer",
                f"ref_checkpointer.checkpoint_files=[{model_ckpt}]",
                "ref_checkpointer.model_type=LLAMA3",
            ]
        )
        try:
            monkeypatch.setattr(sys, "argv", cmd)
            runpy.run_path(TUNE_PATH, run_name="__main__")
            # If we get here, the recipe ran without errors
            # Let's make sure there are some loss values logged
            loss_values = get_loss_values_from_metric_logger(log_file)
            assert loss_values is not None
            # And also make sure there was a checkpoint saved
            output_ckpt_path = (
                Path(tmpdir) / "step_2" / "model-00001-of-00001.safetensors"
            )  # Hardcoded for now, should change eventually
            assert output_ckpt_path.exists()
        finally:
            # Make sure Ray is shut down even if the test fails
            if "ray" in sys.modules and ray.is_initialized():
                ray.shutdown()
