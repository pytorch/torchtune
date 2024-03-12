import logging
import runpy
import tempfile
from pathlib import Path

import pytest
from torchtune import models

logging.basicConfig(level=logging.INFO)
import sys
from functools import partial

import torch
from tests.common import TUNE_PATH

# TODO: is there a better way to do this?
FULL_EVAL_CONFIG_PATH = (
    Path(__file__).parent.parent.parent
    / "torchtune/_cli/eval_configs/full_finetune_eval_config.yaml"
)
LORA_EVAL_CONFIG_PATH = (
    Path(__file__).parent.parent.parent
    / "torchtune/_cli/eval_configs/loar_finetune_eval_config.yaml"
)

# TODO - should probably go into a general directory
from tests.recipes.utils import (
    fetch_ckpt_model_path,
    llama2_small_test_ckpt,
    lora_llama2_small_test_ckpt,
)

models.small_test_ckpt_tune = llama2_small_test_ckpt
test_lora_attn_modules = ["q_proj", "v_proj"]
models.lora_small_test_ckpt = partial(
    lora_llama2_small_test_ckpt,
    lora_attn_modules=test_lora_attn_modules,
    apply_lora_to_mlp=False,
    apply_lora_to_output=False,
    lora_rank=8,
    lora_alpha=16,
)


class TestEval:
    def dsddddtest_full_model_eval_result(self, capsys, monkeypatch):
        model_ckpt = "small_test_ckpt_tune"
        ckpt_path = Path(fetch_ckpt_model_path(model_ckpt))
        ckpt_dir = ckpt_path.parent
        cpu_device_str = "cpu"
        cmd = f"""
        tune eval \
            --config {FULL_EVAL_CONFIG_PATH} \
            --override \
            model._component_=torchtune.models.{model_ckpt} \
            model_checkpoint={ckpt_path} \
            limit={10} \
            device={cpu_device_str} \
        """.split()
        monkeypatch.setattr(sys, "argv", cmd)
        with pytest.raises(SystemExit):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        out_err = capsys.readouterr()
        assert "'acc,none': 0.3" in out_err.out

    def test_lora_eval_result(self, capsys, monkeypatch):
        # TODO: this is a hacky way of getting the LoRA adapter checkpoint. We should
        # generate the adapter checkpoints and store them instead.
        lora_model = models.lora_small_test_ckpt()
        lora_sd = lora_model.state_dict()
        lora_sd = {k: v for k, v in lora_sd.items() if "lora_a" in k or "lora_b" in k}
        with tempfile.NamedTemporaryFile() as f:
            torch.save(lora_sd, f.name)
            lora_checkpoint_path = Path(f.name)
            model_ckpt = "lora_small_test_ckpt"
            ckpt_path = Path(fetch_ckpt_model_path(model_ckpt))
            ckpt_dir = ckpt_path.parent
            cpu_device_str = "cpu"
            cmd = f"""
            tune eval \
                --config {FULL_EVAL_CONFIG_PATH} \
                --override \
                model._component_=torchtune.models.{model_ckpt} \
                model_checkpoint={ckpt_path} \
                lora_checkpoint={lora_checkpoint_path} \
                limit={10} \
                device={cpu_device_str} \
            """.split()
            monkeypatch.setattr(sys, "argv", cmd)
            with pytest.raises(SystemExit):
                runpy.run_path(TUNE_PATH, run_name="__main__")

            out_err = capsys.readouterr()
            assert "'acc,none': 0.3" in out_err.out
