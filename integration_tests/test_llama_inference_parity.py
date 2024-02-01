# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Callable, List, Optional

import pytest
import torch

from tests.test_utils import assert_expected
from torch import nn
from torchtune.models.llama2 import llama2, llama2_tokenizer
from torchtune.modules import Tokenizer
from torchtune.utils.generation import GenerationUtils
from torchtune.utils.seed import set_seed

from transformers import LlamaForCausalLM


@pytest.fixture(autouse=True)
def default_seed():
    set_seed(42)


class TestLlama2InferenceParity:
    @pytest.fixture()
    def prompts(self) -> List[str]:
        return [
            # Few shot prompt (providing a few examples before asking model to complete more);
            """Translate English to French:

            sea otter => loutre de mer
            peppermint => menthe poivrée
            plush girafe => girafe peluche
            cheese =>""",
        ]

    def hugging_face_hub_token(self):
        if "HUGGING_FACE_HUB_TOKEN" not in os.environ:
            raise ValueError("Please set environment variable HUGGING_FACE_HUB_TOKEN")
        return os.environ["HUGGING_FACE_HUB_TOKEN"]

    def generate(
        self,
        decoder: nn.Module,
        tokenizer: Tokenizer,
        prompt_tokens: List[List[int]],
        incremental_decode: bool = False,
        logits_accessor: Optional[Callable] = None,
    ):
        with torch.no_grad():
            generations, _ = GenerationUtils(
                decoder_lm=decoder,
                eos_id=tokenizer.eos_id,
                pad_id=tokenizer.pad_id,
            ).generate(
                prompt_tokens=prompt_tokens,
                incremental_decode=incremental_decode,
                min_gen_len=1,
                max_gen_len=64,
                top_p=0,
                top_k=1,
                temperature=1.0,
                device=torch.device("cpu"),
                logits_accessor=logits_accessor,
            )

            return generations

    def llama2_7b(self, max_batch_size: Optional[int] = None):
        # TODO: Replace with definition in torchtune/models/llama2.py once defaults are reset
        return llama2(
            vocab_size=32_000,
            num_layers=32,
            num_heads=32,
            max_seq_len=4096,
            num_kv_heads=32,
            embed_dim=4096,
            norm_eps=1e-5,
            max_batch_size=max_batch_size,
        )

    def test_parity_with_huggingface(
        self, prompts: List[str], llama2_path: str, tokenizer_path: str
    ):
        tokenizer: Tokenizer = llama2_tokenizer(tokenizer_path)
        tokens = torch.tensor(
            tokenizer.encode(prompts[0], add_eos=False), dtype=torch.long
        )
        token_for_generation: List[List[int]] = [
            tokenizer.encode(prompt, add_eos=False) for prompt in prompts
        ]

        with torch.device("cpu"):
            decoder = self.llama2_7b()

        state_dict = torch.load(
            llama2_path,
            weights_only=True,
            map_location=torch.device("cpu"),
        )
        missing, unexpected = decoder.load_state_dict(state_dict["model"], strict=False)
        assert (
            not missing and not unexpected
        ), f"Missing {missing} and Unexpected: {unexpected}"

        decoder.eval()
        generations_no_kv_cache = self.generate(
            decoder, tokenizer, token_for_generation, incremental_decode=False
        )
        del decoder

        with torch.device("cpu"):
            decoder_kv = self.llama2_7b(max_batch_size=2)

        missing, unexpected = decoder_kv.load_state_dict(
            state_dict["model"], strict=False
        )
        for key in missing:
            assert "kv_cache" in key, f"{key}"

        decoder_kv.eval()
        # incremental_decode is True because we want to use the cache
        generations_kv_cache = self.generate(
            decoder_kv, tokenizer, token_for_generation, incremental_decode=True
        )
        del decoder_kv

        assert torch.allclose(generations_kv_cache, generations_no_kv_cache)

        with torch.device("cpu"):
            hf_decoder = LlamaForCausalLM.from_pretrained(  # pyre-ignore[16]
                "meta-llama/Llama-2-7b-hf",
                use_auth_token=self.hugging_face_hub_token(),
                token=None,
            )

        generations_hf = self.generate(
            hf_decoder,
            tokenizer,
            token_for_generation,
            incremental_decode=False,
            logits_accessor=lambda o: o.logits,
        )

        # check generation parity
        assert_expected(generations_hf, generations_kv_cache)
        assert_expected(generations_hf, generations_no_kv_cache)
