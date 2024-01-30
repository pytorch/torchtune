# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import runpy
import sys
import tempfile

from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torchtune.models.llama2 import llama2

from tests.scripts.common import TUNE_PATH
from tests.test_utils import assert_expected
from tests.torchtune.models.llama2.scripts.compare_decoder import Transformer

ASSETS = Path(__file__).parent.parent / "assets"

# Generating `tiny_state_dict_with_one_key.pt`
# >>> import torch
# >>> state_dict = {"test_key": torch.randn(10, 10)}
# >>> torch.save(state_dict, "tiny_state_dict_with_one_key.pt")

# Generating `tiny_fair_checkpoint.pt`
# >>> from tests.torchtune.models.llama2.scripts.compare_decoder import Transformer
# >>> from tests.test_utils import init_weights_with_constant
# >>> import torch
# >>> tiny_fair_transfomer = Transformer(
#     vocab_size=500,
#     n_layers=2,
#     n_heads=4,
#     dim=32,
#     max_seq_len=64,
#     n_kv_heads=4,
# )
# >>> init_weights_with_constant(tiny_fair_transfomer, constant=0.2)
# >>> torch.save(tiny_fair_transfomer.state_dict(), "tiny_fair_checkpoint.pt")


class TestTuneCLIWithConvertCheckpointScript:
    def test_convert_checkpoint_errors_on_bad_conversion(self, capsys):
        incorrect_state_dict_loc = ASSETS / "tiny_state_dict_with_one_key.pt"
        testargs = f"tune convert_checkpoint --checkpoint-path {incorrect_state_dict_loc}".split()
        with patch.object(sys, "argv", testargs):
            with pytest.raises(
                Exception, match=r".*Are you sure this is a FAIR Llama2 checkpoint.*"
            ) as e:
                runpy.run_path(TUNE_PATH, run_name="__main__")

    def test_convert_checkpoint_matches_fair_model(self, capsys):
        tiny_fair_ckpt = ASSETS / "tiny_fair_checkpoint.pt"
        output_path = tempfile.NamedTemporaryFile(delete=True).name
        testargs = f"tune convert_checkpoint --checkpoint-path {tiny_fair_ckpt} --output-path {output_path}".split()
        with patch.object(sys, "argv", testargs):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        output = capsys.readouterr()
        assert (
            "Succesfully wrote PyTorch-native model checkpoint"
            in output.out.rstrip("\n").split("\n")[-1]
        )

        tiny_fair_transformer = Transformer(
            vocab_size=500,
            n_layers=2,
            n_heads=4,
            dim=32,
            max_seq_len=64,
            n_kv_heads=4,
        )
        tiny_fair_state_dict = torch.load(tiny_fair_ckpt, weights_only=True)
        tiny_fair_transformer.load_state_dict(tiny_fair_state_dict, strict=True)

        tiny_llama2 = llama2(
            vocab_size=500,
            num_layers=2,
            num_heads=4,
            embed_dim=32,
            max_seq_len=64,
            num_kv_heads=4,
        )
        tiny_llama2_state_dict = torch.load(output_path, weights_only=True)
        tiny_llama2.load_state_dict(tiny_llama2_state_dict, strict=True)

        for i in range(10):
            toks = torch.randint(low=0, high=500, size=(16, 64))
            tiny_fair_transformer_out = tiny_fair_transformer(toks)
            tiny_llama2_out = tiny_llama2(toks)
            assert_expected(tiny_fair_transformer_out.sum(), tiny_llama2_out.sum())
