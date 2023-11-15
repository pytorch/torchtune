# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
import argparse

import functools

import logging

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from llm.llama2.tokenizer import Tokenizer
from llm.llama2.transformer import TransformerDecoder
from tests.llm.llama2.scripts.compare_decoder import Transformer
from tests.test_utils import generate
from transformers import LlamaForCausalLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LlamaArgs:
    """
    Dataclass encapsulating various args to instantiate a Llama-2 decoder. The defaults
    are those of a 7b parameter model with a max_seq_len of 2048.

    Args:
        vocab_size (int): Number of entries in vocabulary (default: 32_000)
        embed_dim: (int): Embedding dimension (default: 4096)
        num_layers: (int): Number of Transformer layers (default: 32)
        num_heads (int): Number of attention heads (per layer). (default: 32)
        num_kv_heads: (Optional[int]): Number of key and value heads. This needs to
            be < num_heads and num_heads % num_kv_heads must be 0. `num_kv_heads` can be
            modified to implement GQA or MHA. The default is `None`, in which case
            `num_kv_heads` is set to `num_heads` and MHA is used. Please see
            llm.llama2.attention.LlamaSelfAttention for details.
        max_seq_len: int: Maximum sequence length that this model accepts. Default: 2048
    """

    vocab_size: int = 32_000
    embed_dim: int = 4096
    num_layers: int = 32
    num_heads: int = 32
    num_kv_heads: Optional[int] = None
    max_seq_len: int = 2048


def args_7b() -> LlamaArgs:
    return LlamaArgs(
        vocab_size=32_000,
        embed_dim=4096,
        num_layers=32,
        num_heads=32,
        num_kv_heads=None,
        max_seq_len=2048,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        Example 7B native Llama-2 inference.
        """
    )
    parser.add_argument(
        "--native-checkpoint-path", type=str, help="Path to native checkpoint file."
    )
    parser.add_argument("--tokenizer-path", type=str, help="Path to tokenization file.")

    args = parser.parse_args()
    # Initialize a decoder w/no kv-caching
    llama_7b_args = args_7b()
    # import pdb ; p
    # decoder = TransformerDecoder(vocab_size=llama_7b_args.vocab_size,num_layers=llama_7b_args.num_layers,num_heads=llama_7b_args.num_heads,num_kv_heads=llama_7b_args.num_kv_heads,embed_dim=llama_7b_args.embed_dim,max_seq_len=llama_7b_args.max_seq_len,norm_eps=1e-5,max_batch_size=2)
    with torch.device("cuda:0"):
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

    # Load state_dict into decoder
    native_state_dict = torch.load(args.native_checkpoint_path)
    missing, unexpected = decoder.load_state_dict(native_state_dict, strict=False)
    # Nothing should be missing or unexpected
    assert not missing and not unexpected
    decoder.eval()
    # Do the same initialization process, but with a kv-caching decoder.
    with torch.device("cuda:1"):
        decoder_kv = TransformerDecoder(
            vocab_size=llama_7b_args.vocab_size,
            num_layers=llama_7b_args.num_layers,
            num_heads=llama_7b_args.num_heads,
            num_kv_heads=llama_7b_args.num_kv_heads,
            embed_dim=llama_7b_args.embed_dim,
            max_seq_len=llama_7b_args.max_seq_len,
            norm_eps=1e-5,
            max_batch_size=2,
        )
    missing, unexpected = decoder_kv.load_state_dict(native_state_dict, strict=False)
    # Only kv_cache stuff should be missing
    for key in missing:
        assert "kv_cache" in key, f"{key}"
    decoder_kv.eval()
    print(f"RV: tbd missing unexpected {missing} {unexpected}")
    # print(f"RV: fair missing & unexpected: {m2} {u2}")
    # print(f"RV: mm missing unexpected: {me} {une}")
    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    auth_token = "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"
    # Initialize a HF model
    with torch.device("cuda:2"):
        model_hf = LlamaForCausalLM.from_pretrained(  # pyre-ignore[16]
            "meta-llama/Llama-2-7b-hf",
            use_auth_token=auth_token,
            token=None,
        )
    # Inference
    prompts = [
        # Few shot prompt (providing a few examples before asking model to complete more);
        """Translate English to French:

        sea otter => loutre de mer
        peppermint => menthe poivrée
        plush girafe => girafe peluche
        cheese =>""",
    ]

    tokens = torch.tensor(tokenizer.encode(prompts[0], add_eos=False), dtype=torch.long)
    with torch.no_grad():
        decoder_out = decoder(tokens.unsqueeze(0).cuda(0), 0).sum()
        decoder_kv_out = decoder_kv(tokens.unsqueeze(0).cuda(1), 0).sum().cuda(0)
        hf_out = model_hf(tokens.unsqueeze(0).cuda(2)).logits.sum().cuda(0)

    assert torch.allclose(decoder_out, hf_out)
    assert torch.allclose(decoder_kv_out, hf_out)

    # Check generation parity
    toks = [tokenizer.encode(prompt, add_eos=False) for prompt in prompts]
    with torch.no_grad():
        generations_no_kv_cache, _ = generate(
            decoder_lm=decoder,
            prompt_tokens=toks,
            incremental_decode=False,  # Since we aren't caching past keys and values
            min_gen_len=1,
            max_gen_len=64,
            top_p=0,
            top_k=0,
            temperature=1.0,
            eos_token_id=tokenizer.eos_id,
            pad_token_id=tokenizer.pad_id,
            device=torch.device("cuda:0"),
            decoder_lm_kwargs=True,
        )
        generations_kv_cache, _ = generate(
            decoder_lm=decoder_kv,
            prompt_tokens=toks,
            incremental_decode=True,
            min_gen_len=1,
            max_gen_len=64,
            top_p=0,
            top_k=0,
            temperature=1.0,
            eos_token_id=tokenizer.eos_id,
            pad_token_id=tokenizer.pad_id,
            decoder_lm_kwargs=True,
            device=torch.device("cuda:1"),
        )
        generations_hf, _ = generate(
            decoder_lm=model_hf,
            prompt_tokens=toks,
            incremental_decode=False,
            min_gen_len=1,
            max_gen_len=64,
            top_p=0,
            top_k=0,
            temperature=1.0,
            eos_token_id=tokenizer.eos_id,
            pad_token_id=tokenizer.pad_id,
            device=torch.device("cuda:2"),
            decoder_lm_kwargs=False,
            logits_accessor=lambda o: o.logits,
        )

    assert torch.allclose(generations_kv_cache.cuda(0), generations_no_kv_cache)
    assert torch.allclose(generations_kv_cache.cuda(0), generations_hf.cuda(0))
