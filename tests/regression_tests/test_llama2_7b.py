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
import torchtune
from tests.common import TUNE_PATH
from tests.test_utils import CKPT_MODEL_PATHS, gpu_test

from torchtune.training.checkpointing._utils import get_largest_iter_folder, SHARD_FNAME

CKPT = "llama2_7b"

# TODO: remove this once we have eval configs exposed properly
pkg_path = Path(torchtune.__file__).parent.absolute()
EVAL_CONFIG_PATH = Path.joinpath(
    pkg_path, "_cli", "eval_configs", "default_eval_config.yaml"
)


@gpu_test(gpu_count=2)
class TestLoRA7BDistributedFinetuneEval:
    @pytest.mark.slow_integration_test
    def test_finetune_and_eval(self, tmpdir, capsys, monkeypatch):

        ckpt_path = Path(CKPT_MODEL_PATHS[CKPT])
        ckpt_dir = ckpt_path.parent

        # Run on prod LoRA FT config but with only 10 steps for now
        ft_cmd = f"""
        tune run --nnodes 1 --nproc_per_node 2 lora_finetune_distributed
            --config llama2/7B_lora \
            model.lora_attn_modules=['q_proj','v_proj'] \
            model.apply_lora_to_mlp=False \
            output_dir={tmpdir} \
            checkpointer=torchtune.training.FullModelTorchTuneCheckpointer
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[{ckpt_path}]\
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type=LLAMA2 \
            tokenizer.path=/tmp/test-artifacts/tokenizer.model \
            max_steps_per_epoch=10 \
        """.split()

        monkeypatch.setattr(sys, "argv", ft_cmd)
        runpy.run_path(TUNE_PATH, run_name="__main__")

        epoch_folder = get_largest_iter_folder(tmpdir)
        suffix = ".bin"
        model_ckpt_fname = (
            SHARD_FNAME.format(cpt_idx="1".zfill(5), num_shards="1".zfill(5)) + suffix
        )
        eval_cmd = f"""
        tune run eleuther_eval \
            --config eleuther_evaluation \
            output_dir={tmpdir} \
            checkpointer=torchtune.training.FullModelTorchTuneCheckpointer \
            checkpointer.checkpoint_dir='{tmpdir}' \
            checkpointer.checkpoint_files=[{os.path.join(tmpdir, epoch_folder, model_ckpt_fname)}]\
            checkpointer.output_dir={tmpdir} \
            tokenizer.path=/tmp/test-artifacts/tokenizer.model \
            tasks=['truthfulqa_mc2']
            limit=10 \
            device=cuda \
        """.split()
        monkeypatch.setattr(sys, "argv", eval_cmd)
        with pytest.raises(SystemExit):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        out = capsys.readouterr().out
        search_results = re.search(
            r"acc(?:_norm)?\s*\|?\s*(?:\↑\s*\|?)?([\d.]+)", out.strip()
        )
        assert search_results is not None
        acc_result = float(search_results.group(1))
        assert acc_result >= 0.4
