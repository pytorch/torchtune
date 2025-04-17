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
    dummy_stack_exchange_dataset_config,
    MODEL_TEST_CONFIGS,
    write_hf_ckpt_config,
)
from tests.test_utils import (
    CKPT_MODEL_PATHS,
    gen_log_file_name,
    get_loss_values_from_metric_logger,
    gpu_test,
)
from torchtune import config

from torchtune.training.checkpointing._utils import (
    ADAPTER_MODEL_FNAME,
    get_largest_iter_folder,
    RECIPE_STATE_DIRNAME,
    safe_torch_load,
    SHARD_FNAME,
)


class TestLoRADPODistributedRecipe:
    def _get_test_config_overrides(self, dtype_str: str = "fp32", epochs: int = 2):
        return [
            "batch_size=8",
            "device=cuda",
            f"dtype={dtype_str}",
            "dataset.train_on_input=False",
            "seed=9",
            f"epochs={epochs}",
            "max_steps_per_epoch=2",
            "optimizer.lr=2e-5",
            "log_every_n_steps=1",
            "gradient_accumulation_steps=1",
            "clip_grad_norm=100",
            "tokenizer.max_seq_len=512",
        ] + dummy_stack_exchange_dataset_config()

    @pytest.mark.parametrize("save_adapter_weights_only", [False, True])
    @gpu_test(gpu_count=4)
    @pytest.mark.integration_test
    def test_training_state_on_resume(
        self, tmpdir, monkeypatch, save_adapter_weights_only
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

        ckpt = "llama2_hf"
        ckpt_path = Path(CKPT_MODEL_PATHS[ckpt])
        ckpt_dir = ckpt_path.parent
        log_file = gen_log_file_name(tmpdir)

        # Config file needed for model conversion.
        # Create a second copy for training resume
        write_hf_ckpt_config(ckpt_dir)
        write_hf_ckpt_config(tmpdir)

        # Train for two epochs
        cmd_1 = f"""
        tune run  --nnodes 1 --nproc_per_node 4 lora_dpo_distributed \
            --config llama2/7B_lora_dpo \
            output_dir={tmpdir} \
            model.lora_attn_modules=['q_proj','v_proj'] \
            model.apply_lora_to_mlp=False \
            checkpointer=torchtune.training.FullModelHFCheckpointer \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[{ckpt_path}]\
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type=LLAMA2 \
            tokenizer.path=/tmp/test-artifacts/tokenizer.model \
            tokenizer.prompt_template=null \
            save_adapter_weights_only={save_adapter_weights_only} \
            metric_logger.filename={log_file} \
            enable_activation_checkpointing=True \
            enable_activation_offloading=False \
        """.split()

        model_config = MODEL_TEST_CONFIGS["llama2_lora"]

        cmd_1 = cmd_1 + self._get_test_config_overrides() + model_config
        monkeypatch.setattr(sys, "argv", cmd_1)
        runpy.run_path(TUNE_PATH, run_name="__main__")

        expected_loss_values = get_loss_values_from_metric_logger(log_file)

        resumed_log_dir = (tmpdir / "resumed/").mkdir()
        resumed_log_file = gen_log_file_name(resumed_log_dir)

        # Resume training
        epoch_folder = get_largest_iter_folder(tmpdir)
        epoch_folder_minus_one = f"epoch_{int(epoch_folder.split('_')[-1]) - 1}"
        cmd_2 = f"""
        tune run  --nnodes 1 --nproc_per_node 4 lora_dpo_distributed \
            --config llama2/7B_lora_dpo \
            output_dir={tmpdir} \
            model.lora_attn_modules=['q_proj','v_proj'] \
            model.apply_lora_to_mlp=False \
            checkpointer=torchtune.training.FullModelHFCheckpointer \
            checkpointer.checkpoint_dir={ckpt_dir} \
            checkpointer.checkpoint_files=[{ckpt_path}]\
            checkpointer.adapter_checkpoint={os.path.join(tmpdir, epoch_folder_minus_one, f"{ADAPTER_MODEL_FNAME}.pt")}
            checkpointer.recipe_checkpoint={os.path.join(tmpdir, RECIPE_STATE_DIRNAME, "recipe_state.pt")}
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type=LLAMA2 \
            resume_from_checkpoint=True \
            metric_logger.filename={resumed_log_file} \
            tokenizer.path=/tmp/test-artifacts/tokenizer.model \
            tokenizer.prompt_template=null \
            enable_activation_checkpointing=True \
            enable_activation_offloading=False \
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
    def test_save_and_load_merged_weights(self, tmpdir, monkeypatch):
        ckpt = "llama2_tune"
        ckpt_path = Path(CKPT_MODEL_PATHS[ckpt])
        ckpt_dir = ckpt_path.parent

        cmd = f"""
        tune run  --nnodes 1 --nproc_per_node 4 lora_dpo_distributed \
            --config llama2/7B_lora_dpo \
            output_dir={tmpdir} \
            model.lora_attn_modules=['q_proj','v_proj'] \
            model.apply_lora_to_mlp=False \
            checkpointer=torchtune.training.FullModelTorchTuneCheckpointer \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[{ckpt_path}]\
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type=LLAMA2 \
            tokenizer.path=/tmp/test-artifacts/tokenizer.model \
            tokenizer.prompt_template=null \
            enable_activation_checkpointing=False \
            enable_activation_offloading=False \
        """.split()

        model_config = MODEL_TEST_CONFIGS["llama2_lora"]

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
        base_llama2_config = MODEL_TEST_CONFIGS["llama2"]
        llama2_model = config.instantiate(
            OmegaConf.from_dotlist(base_llama2_config).model
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

        # Load merged final ckpt directly into llama2 and call fwd
        suffix = ".bin"
        model_ckpt_fname = (
            SHARD_FNAME.format(cpt_idx="1".zfill(5), num_shards="1".zfill(5)) + suffix
        )
        model_path = os.path.join(tmpdir, epoch_folder, model_ckpt_fname)
        sd = safe_torch_load(model_path, weights_only=True)

        llama2_model.load_state_dict(sd)
        merged_ckpt_out = llama2_model(inputs)
        torch.testing.assert_close(baseline_out, merged_ckpt_out, rtol=1e-5, atol=1e-5)
