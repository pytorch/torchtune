# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Tuple

import pytest

import torch

from llm.llama2.transformer import TransformerDecoder, TransformerDecoderLayer
from tests.test_utils import generate
from transformers import LlamaForCausalLM

from tests.test_utils import assert_expected, init_weights_with_constant, set_rng_seed

from torch import Tensor


@pytest.fixture(autouse=True)
def random():
    set_rng_seed(16)


class TestTransformerDecoder:
    """
    Class for testing parity between HF Llama2 and TransformerDecoder implementation.
    Tests currently cover training, inference, generations, and kv-caching.
    """

    @pytest.fixture
    def checkpoint_path(self) -> str:
        if os.environ.get("NATIVE_CKPT_PATH", None) is None:
            raise RuntimeError(
                "Tests require env var NATIVE_CKPT_PATH to point to a native checkpoint file."
            )

        return os.environ["NATIVE_CKPT_PATH"]

    @pytest.fixture
    def decoder(self, checkpoint_path) -> TransformerDecoder:
        args = args_7b()
        decoder = TransformerDecoder(
            vocab_size=llama_7b_args.vocab_size,
            num_layers=llama_7b_args.num_layers,
            num_heads=llama_7b_args.num_heads,
            num_kv_heads=llama_7b_args.num_kv_heads,
            embed_dim=llama_7b_args.embed_dim,
            max_seq_len=llama_7b_args.max_seq_len,
            norm_eps=1e-5,
            max_batch_size=None,
        )
        missing, unexpected = decoder.load_state_dict(checkpoint_path, strict=False)
        assert not missing and not unexpected, f"missing/unexpected keys: {missing}, {unexpected}"
        return decoder

    @pytest.fixture
    def hf_decoder(self) -> Llama

    def test_forward(
        self,
        input: Tensor,
        input_params: Tuple[int, int, int],
        decoder: TransformerDecoder,
    ) -> None:
        batch_size, seq_len, vocab_size = input_params
        with torch.no_grad():
            output = decoder(input)
        assert_expected(output.mean(), torch.tensor(163.8399), atol=1e-8, rtol=1e-6)
        assert_expected(output.shape, torch.Size([batch_size, seq_len, vocab_size]))

    def test_max_seq_len_exceeded(
        self,
        input_max_len_exceeded: Tensor,
        decoder: TransformerDecoder,
    ) -> None:
        with pytest.raises(Exception):
            output = decoder(input_max_len_exceeded)
