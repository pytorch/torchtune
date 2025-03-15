# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import re
import runpy
import sys
from pathlib import Path

import pytest
from tests.common import TUNE_PATH
from tests.test_utils import CKPT_MODEL_PATHS, TOKENIZER_PATHS, gpu_test

from torchtune.training.checkpointing._utils import get_largest_iter_folder

CKPT = "qwen2_5_3b"
        
@gpu_test(gpu_count=2)
class TestFull3BDistributedFinetuneEval:
    @pytest.mark.slow_integration_test
    def test_finetune_and_eval(self, tmpdir, caplog, monkeypatch):
        ckpt_path = Path(CKPT_MODEL_PATHS[CKPT])

        # Run on prod Full FT config but with only 20 steps for now
        ft_cmd = f"""
        tune run --nnodes 1 --nproc_per_node 2 full_finetune_distributed
            --config qwen2_5/3B_full \
            output_dir={tmpdir} \
            checkpointer=torchtune.training.FullModelHFCheckpointer
            checkpointer.checkpoint_dir='{ckpt_path}' \
            checkpointer.output_dir={tmpdir} \
            tokenizer.path={TOKENIZER_PATHS[CKPT]} \
            tokenizer.merges_file={Path.joinpath(Path(TOKENIZER_PATHS[CKPT]).parent, "merges.txt")}
            max_steps_per_epoch=20 \
        """.split()

        monkeypatch.setattr(sys, "argv", ft_cmd)
        runpy.run_path(TUNE_PATH, run_name="__main__")
        epoch_folder = get_largest_iter_folder(tmpdir)
 
        eval_cmd = f"""
        tune run eleuther_eval \
            --config qwen2_5/evaluation \
            output_dir={tmpdir} \
            model=torchtune.models.qwen2_5.qwen2_5_3b \
            checkpointer=torchtune.training.FullModelHFCheckpointer \
            checkpointer.checkpoint_dir='{Path.joinpath(Path(tmpdir), epoch_folder)}' \
            checkpointer.output_dir={tmpdir} \
            tokenizer.path={TOKENIZER_PATHS[CKPT]} \
            tokenizer.merges_file={Path.joinpath(Path(TOKENIZER_PATHS[CKPT]).parent, "merges.txt")} \
            tasks=['truthfulqa_mc2']
            limit=10 \
            device=cuda \
        """.split()
        
        monkeypatch.setattr(sys, "argv", eval_cmd)
        
         
        with pytest.raises(SystemExit):
            runpy.run_path(TUNE_PATH, run_name="__main__")
        
        out = caplog.text
        
        assert "acc" in out
        search_results = re.search(
            r"acc(?:_norm)?\s*\|?\s*(?:\↑\s*\|?)?([\d.]+)", out.strip()
        )
        assert search_results is not None
        acc_result = float(search_results.group(1))
        assert acc_result >= 0.62


@gpu_test(gpu_count=2)
class TestLora3BDistributedFinetuneEval:
    @pytest.mark.slow_integration_test
    def test_finetune_and_eval(self, tmpdir, caplog, monkeypatch):
        ckpt_path = Path(CKPT_MODEL_PATHS[CKPT])

        # Run on prod LoRA FT config but with only 20 steps for now
        ft_cmd = f"""
        tune run --nnodes 1 --nproc_per_node 2 lora_finetune_distributed
            --config qwen2_5/3B_lora \
            output_dir={tmpdir} \
            checkpointer=torchtune.training.FullModelHFCheckpointer
            checkpointer.checkpoint_dir='{ckpt_path}' \
            checkpointer.output_dir={tmpdir} \
            tokenizer.path={TOKENIZER_PATHS[CKPT]} \
            tokenizer.merges_file={Path.joinpath(Path(TOKENIZER_PATHS[CKPT]).parent, "merges.txt")}
            max_steps_per_epoch=20 \
        """.split()

        monkeypatch.setattr(sys, "argv", ft_cmd)
        runpy.run_path(TUNE_PATH, run_name="__main__")
        epoch_folder = get_largest_iter_folder(tmpdir)
 
        eval_cmd = f"""
        tune run eleuther_eval \
            --config qwen2_5/evaluation \
            output_dir={tmpdir} \
            model=torchtune.models.qwen2_5.qwen2_5_3b \
            checkpointer=torchtune.training.FullModelHFCheckpointer \
            checkpointer.checkpoint_dir='{Path.joinpath(Path(tmpdir), epoch_folder)}' \
            checkpointer.output_dir={tmpdir} \
            tokenizer.path={TOKENIZER_PATHS[CKPT]} \
            tokenizer.merges_file={Path.joinpath(Path(TOKENIZER_PATHS[CKPT]).parent, "merges.txt")} \
            tasks=['truthfulqa_mc2']
            limit=10 \
            device=cuda \
        """.split()
        
        monkeypatch.setattr(sys, "argv", eval_cmd)
        
         
        with pytest.raises(SystemExit):
            runpy.run_path(TUNE_PATH, run_name="__main__")
        
        out = caplog.text
        
        assert "acc" in out
        search_results = re.search(
            r"acc(?:_norm)?\s*\|?\s*(?:\↑\s*\|?)?([\d.]+)", out.strip()
        )
        assert search_results is not None
        acc_result = float(search_results.group(1))
        assert acc_result >= 0.62


@gpu_test(gpu_count=2)
class TestFull3BDistributedFinetuneDPO:
    @pytest.mark.slow_integration_test
    def test_finetune_and_eval(self, tmpdir, caplog, monkeypatch):
        ckpt_path = Path(CKPT_MODEL_PATHS[CKPT])

        # Run on prod Full DPO config but with only 20 steps for now
        ft_cmd = f"""
        tune run --nnodes 1 --nproc_per_node 2 full_dpo_distributed
            --config llama3_1/8B_full_dpo \
            output_dir={tmpdir} \
            model=torchtune.models.qwen2_5.qwen2_5_3b \
            checkpointer=torchtune.training.FullModelHFCheckpointer
            checkpointer.checkpoint_dir='{ckpt_path}' \
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type='QWEN2' \
            checkpointer.model_files=['model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors',] \
            ref_checkpointer.model_type='QWEN2' \
            ref_checkpointer.checkpoint_dir='{ckpt_path}' \
            ref_checkpointer.output_dir={tmpdir} \
            ref_checkpointer.model_files=['model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors',] \
            tokenizer=torchtune.models.qwen2_5.qwen2_5_tokenizer \
            tokenizer.path={TOKENIZER_PATHS[CKPT]} \
            tokenizer.merges_file={Path.joinpath(Path(TOKENIZER_PATHS[CKPT]).parent, "merges.txt")}
            max_steps_per_epoch=20 \
        """.split()

        monkeypatch.setattr(sys, "argv", ft_cmd)
        runpy.run_path(TUNE_PATH, run_name="__main__")
        epoch_folder = get_largest_iter_folder(tmpdir)
 
        eval_cmd = f"""
        tune run eleuther_eval \
            --config qwen2_5/evaluation \
            output_dir={tmpdir} \
            model=torchtune.models.qwen2_5.qwen2_5_3b \
            checkpointer=torchtune.training.FullModelHFCheckpointer \
            checkpointer.checkpoint_dir='{Path.joinpath(Path(tmpdir), epoch_folder)}' \
            checkpointer.output_dir={tmpdir} \
            tokenizer.path={TOKENIZER_PATHS[CKPT]} \
            tokenizer.merges_file={Path.joinpath(Path(TOKENIZER_PATHS[CKPT]).parent, "merges.txt")} \
            tasks=['truthfulqa_mc2']
            limit=10 \
            device=cuda \
        """.split()
        
        monkeypatch.setattr(sys, "argv", eval_cmd)
        
         
        with pytest.raises(SystemExit):
            runpy.run_path(TUNE_PATH, run_name="__main__")
        
        out = caplog.text
        
        assert "acc" in out
        search_results = re.search(
            r"acc(?:_norm)?\s*\|?\s*(?:\↑\s*\|?)?([\d.]+)", out.strip()
        )
        assert search_results is not None
        acc_result = float(search_results.group(1))
        assert acc_result >= 0.62
        