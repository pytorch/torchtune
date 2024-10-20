# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import re
import runpy
import sys
from pathlib import Path

import pytest

from tests.common import TUNE_PATH
from tests.recipes.utils import llama2_test_config, write_hf_ckpt_config
from tests.test_utils import CKPT_MODEL_PATHS


class TestEleutherEval:
    @pytest.mark.parametrize(
        "eval_name, expected_acc, bsz",
        [
            ("truthfulqa_gen", 0.1, 4),
            ("truthfulqa_gen", 0.1, 1),
            ("truthfulqa_mc2", 0.4, 4),
        ],
    )
    @pytest.mark.integration_test
    def test_torchtune_checkpoint_eval_results(
        self, caplog, monkeypatch, tmpdir, eval_name, expected_acc, bsz
    ):
        ckpt = "llama2_tune"
        ckpt_path = Path(CKPT_MODEL_PATHS[ckpt])
        ckpt_dir = ckpt_path.parent

        # explicitly setting limit to an odd number here to ensure generation tasks
        # work with KV-cacheing + bsz > 1 - we'll receive batches of size 4, 4, 3
        cmd = f"""
        tune run eleuther_eval \
            --config eleuther_evaluation \
            output_dir={tmpdir} \
            checkpointer=torchtune.training.FullModelTorchTuneCheckpointer \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[{ckpt_path}]\
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type=LLAMA2 \
            tokenizer.path=/tmp/test-artifacts/tokenizer.model \
            tokenizer.prompt_template=null \
            limit=11 \
            dtype=fp32 \
            device=cpu \
            tasks=[{eval_name}]\
            batch_size={bsz} \
        """.split()

        model_config = llama2_test_config()
        cmd = cmd + model_config

        monkeypatch.setattr(sys, "argv", cmd)
        with pytest.raises(SystemExit, match=""):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        out = caplog.text

        # Format of output is:
        # |    Tasks     |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
        # |--------------|------:|------|-----:|------|---|-----:|---|-----:|
        # |truthfulqa_mc2|      2|none  |     0|acc   |↑  |0.4497|±  |0.1067|
        search_results = re.search(
            r"acc(?:_norm)?\s*\|?\s*(?:\↑\s*\|?)?([\d.]+)", out.strip()
        )
        assert search_results is not None
        acc_result = float(search_results.group(1))
        assert math.isclose(acc_result, expected_acc, abs_tol=0.05)

    @pytest.fixture
    def hide_correct_version_number(self, monkeypatch):
        import importlib.metadata

        import_orig = importlib.metadata.version

        def mocked_import(name, *args, **kwargs):
            if name == "lm-eval":
                return "0.4.4"  # Hardcode wrong version number
            return import_orig(name, *args, **kwargs)

        monkeypatch.setattr(importlib.metadata, "version", mocked_import)

    @pytest.mark.integration_test
    @pytest.mark.usefixtures("hide_correct_version_number")
    def test_eval_recipe_errors_without_lm_eval(self, capsys, monkeypatch, tmpdir):
        ckpt = "llama2_tune"
        ckpt_path = Path(CKPT_MODEL_PATHS[ckpt])
        ckpt_dir = ckpt_path.parent

        cmd = f"""
        tune run eleuther_eval \
            --config eleuther_evaluation \
            output_dir={tmpdir} \
            checkpointer=torchtune.training.FullModelTorchTuneCheckpointer \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[{ckpt_path}]\
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type=LLAMA2 \
            tokenizer.path=/tmp/test-artifacts/tokenizer.model \
            tokenizer.prompt_template=null \
            limit=1 \
            dtype=fp32 \
            device=cpu \
        """.split()

        model_config = llama2_test_config()
        cmd = cmd + model_config

        monkeypatch.setattr(sys, "argv", cmd)
        with pytest.raises(
            RuntimeError,
            match="This recipe requires EleutherAI Eval Harness v0.4.5. "
            "Please install with `pip install lm-eval==0.4.5`",
        ):
            runpy.run_path(TUNE_PATH, run_name="__main__")

    @pytest.mark.integration_test
    def test_eval_recipe_errors_with_quantization_hf_checkpointer(
        self, capsys, monkeypatch, tmpdir
    ):
        ckpt = "llama2_hf"
        ckpt_path = Path(CKPT_MODEL_PATHS[ckpt])
        ckpt_dir = ckpt_path.parent

        # Config file needed for model conversion.
        write_hf_ckpt_config(ckpt_dir)

        cmd = f"""
        tune run eleuther_eval \
            --config eleuther_evaluation \
            output_dir={tmpdir} \
            checkpointer=torchtune.training.FullModelHFCheckpointer \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[{ckpt_path}]\
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type=LLAMA2 \
            tokenizer.path=/tmp/test-artifacts/tokenizer.model \
            tokenizer.prompt_template=null \
            limit=1 \
            dtype=fp32 \
            device=cpu \
            quantizer._component_=torchtune.training.quantization.Int8DynActInt4WeightQuantizer \
            quantizer.groupsize=256 \
        """.split()

        model_config = llama2_test_config()
        cmd = cmd + model_config

        monkeypatch.setattr(sys, "argv", cmd)
        with pytest.raises(
            ValueError,
            match="Quantization is only supported for models quantized and saved with the "
            "FullModelTorchTuneCheckpointer",
        ):
            runpy.run_path(TUNE_PATH, run_name="__main__")

    @pytest.mark.integration_test
    def test_eval_recipe_errors_with_qat_quantizer(self, capsys, monkeypatch, tmpdir):
        ckpt = "llama2_tune"
        ckpt_path = Path(CKPT_MODEL_PATHS[ckpt])
        ckpt_dir = ckpt_path.parent

        cmd = f"""
        tune run eleuther_eval \
            --config eleuther_evaluation \
            output_dir={tmpdir} \
            checkpointer=torchtune.training.FullModelTorchTuneCheckpointer \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[{ckpt_path}]\
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type=LLAMA2 \
            tokenizer.path=/tmp/test-artifacts/tokenizer.model \
            tokenizer.prompt_template=null \
            limit=1 \
            dtype=fp32 \
            device=cpu \
            quantizer._component_=torchtune.training.quantization.Int8DynActInt4WeightQATQuantizer \
            quantizer.groupsize=32\
        """.split()

        model_config = llama2_test_config()
        cmd = cmd + model_config

        monkeypatch.setattr(sys, "argv", cmd)
        with pytest.raises(
            ValueError,
            match="QAT quantizers should only be used during quantization aware training",
        ):
            runpy.run_path(TUNE_PATH, run_name="__main__")
