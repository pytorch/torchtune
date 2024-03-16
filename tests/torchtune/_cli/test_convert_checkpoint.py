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

from tests.common import TUNE_PATH
from tests.test_utils import assert_expected
from tests.torchtune.models.llama2.scripts.compare_decoder import Transformer
from torchtune.models.llama2 import llama2

ASSETS = Path(__file__).parent.parent.parent / "assets"

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
        testargs = (
            f"tune convert_checkpoint --checkpoint-path {incorrect_state_dict_loc} --model llama2 --train-type full"
        ).split()
        with patch.object(sys, "argv", testargs):
            with pytest.raises(
                Exception, match=r".*Error converting the original Llama2.*"
            ) as e:
                runpy.run_path(TUNE_PATH, run_name="__main__")

    def _tiny_fair_transformer(self, ckpt):
        tiny_fair_transfomer = Transformer(
            vocab_size=500,
            n_layers=2,
            n_heads=4,
            dim=32,
            max_seq_len=64,
            n_kv_heads=4,
        )
        tiny_fair_state_dict = torch.load(ckpt, weights_only=True)
        tiny_fair_transfomer.load_state_dict(tiny_fair_state_dict, strict=True)
        return tiny_fair_transfomer

    def _tiny_native_transformer(self, ckpt):
        tiny_native_transfomer = llama2(
            vocab_size=500,
            num_layers=2,
            num_heads=4,
            embed_dim=32,
            max_seq_len=64,
            num_kv_heads=4,
        )
        tiny_native_state_dict = torch.load(ckpt, weights_only=True)
        tiny_native_transfomer.load_state_dict(
            tiny_native_state_dict["model"], strict=False
        )
        return tiny_native_transfomer

    def _llama2_7b_fair_transformer(self, ckpt):
        llama2_7b_fair_transformer = Transformer(
            vocab_size=32_000,
            n_layers=32,
            n_heads=32,
            dim=4096,
            max_seq_len=2048,
            n_kv_heads=32,
        )
        llama2_7b_fair_state_dict = torch.load(ckpt, weights_only=True)
        llama2_7b_fair_transformer.load_state_dict(
            llama2_7b_fair_state_dict, strict=False
        )
        llama2_7b_fair_transformer.eval()
        return llama2_7b_fair_transformer

    def _llama2_7b_native_transformer(self, ckpt):
        llama2_7b_native_transformer = llama2(
            vocab_size=32_000,
            num_layers=32,
            num_heads=32,
            embed_dim=4096,
            max_seq_len=2048,
            num_kv_heads=32,
        )
        llama2_7b_native_state_dict = torch.load(ckpt, weights_only=True)
        llama2_7b_native_transformer.load_state_dict(
            llama2_7b_native_state_dict["model"], strict=True
        )
        llama2_7b_native_transformer.eval()
        return llama2_7b_native_transformer

    def _generate_toks_for_tiny(self):
        return torch.randint(low=0, high=500, size=(16, 64))

    def _generate_toks_for_llama2_7b(self):
        return torch.randint(low=0, high=32_000, size=(16, 128))

    def test_convert_checkpoint_matches_fair_model(self, caplog, pytestconfig):
        is_large_scale_test = pytestconfig.getoption("--large-scale")

        if is_large_scale_test:
            ckpt = "/tmp/test-artifacts/llama2-7b-fair"
            fair_transformer = self._llama2_7b_fair_transformer(ckpt)
        else:
            ckpt = ASSETS / "tiny_fair_checkpoint.pt"
            fair_transformer = self._tiny_fair_transformer(ckpt)

        output_path = tempfile.NamedTemporaryFile(delete=True).name
        testargs = (
            f"tune convert_checkpoint --checkpoint-path {ckpt} --output-path {output_path} --model llama2 --train-type lora"
        ).split()
        with patch.object(sys, "argv", testargs):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        output = caplog.text
        assert "Succesfully wrote PyTorch-native model checkpoint" in output

        native_transformer = (
            self._llama2_7b_native_transformer(output_path)
            if is_large_scale_test
            else self._tiny_native_transformer(output_path)
        )

        with torch.no_grad():
            for i in range(10):
                toks = (
                    self._generate_toks_for_llama2_7b()
                    if is_large_scale_test
                    else self._generate_toks_for_tiny()
                )
                fair_out = fair_transformer(toks)
                native_out = native_transformer(toks)
                assert_expected(fair_out.sum(), native_out.sum())
