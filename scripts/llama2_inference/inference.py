# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import logging

import torch

from tests.test_utils import set_rng_seed
from torchtune.models.llama2.tokenizer import Tokenizer
from torchtune.models.llama2.transformer import TransformerDecoder
from torchtune.models.llama2.utils import llama_7b_args as args_7b
from torchtune.utils.generation import GenerationUtils
from transformers import LlamaForCausalLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    parser.add_argument("--hf-auth-token", type=str, help="HuggingFace auth token")
    args = parser.parse_args()

    # Inference setup
    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    prompts = [
        # Few shot prompt (providing a few examples before asking model to complete more);
        """Translate English to French:

        sea otter => loutre de mer
        peppermint => menthe poivrée
        plush girafe => girafe peluche
        cheese =>""",
    ]
    tokens = torch.tensor(tokenizer.encode(prompts[0], add_eos=False), dtype=torch.long)
    token_for_generation = [
        tokenizer.encode(prompt, add_eos=False) for prompt in prompts
    ]

    set_rng_seed(0)

    # --------- Initialize a decoder w/o kv-caching -------- #
    llama_7b_args = args_7b()
    with torch.device("cuda"):
        decoder = TransformerDecoder(
            vocab_size=llama_7b_args.vocab_size,
            num_layers=llama_7b_args.num_layers,
            num_heads=llama_7b_args.num_heads,
            num_kv_heads=llama_7b_args.num_kv_heads,
            embed_dim=llama_7b_args.embed_dim,
            max_seq_len=llama_7b_args.max_seq_len,
            norm_eps=1e-5,
        )

    # Load state_dict into decoder
    native_state_dict = torch.load(args.native_checkpoint_path, weights_only=True)
    missing, unexpected = decoder.load_state_dict(native_state_dict, strict=False)
    # Nothing should be missing or unexpected
    assert not missing and not unexpected
    decoder.eval()

    with torch.no_grad():
        decoder_out = decoder(tokens.unsqueeze(0).cuda(), 0).sum()
        generations_no_kv_cache, _ = GenerationUtils(
            decoder_lm=decoder,
            eos_id=tokenizer.eos_id,
            pad_id=tokenizer.pad_id,
        ).generate(
            prompt_tokens=token_for_generation,
            incremental_decode=False,  # Since we aren't caching past keys and values
            min_gen_len=1,
            max_gen_len=64,
            top_p=0,
            top_k=1,
            temperature=1.0,
            device=torch.device("cuda"),
        )

    # Remove from memory to allow to run on single A100
    del decoder

    # --------- Do the same initialization process, but with a kv-caching decoder. ------- #
    with torch.device("cuda"):
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

    with torch.no_grad():
        decoder_kv_out = decoder_kv(tokens.unsqueeze(0).cuda(), 0).sum()
        generations_kv_cache, _ = GenerationUtils(
            decoder_lm=decoder_kv, eos_id=tokenizer.eos_id, pad_id=tokenizer.pad_id
        ).generate(
            prompt_tokens=token_for_generation,
            incremental_decode=True,
            min_gen_len=1,
            max_gen_len=64,
            top_p=0,
            top_k=1,
            temperature=1.0,
            device=torch.device("cuda"),
        )

    del decoder_kv

    # --------- Initialize a HF model --------- #
    with torch.device("cuda"):
        auth_token = args.hf_auth_token
        model_hf = LlamaForCausalLM.from_pretrained(  # pyre-ignore[16]
            "meta-llama/Llama-2-7b-hf",
            use_auth_token=auth_token,
            token=None,
        )

    with torch.no_grad():
        hf_out = model_hf(tokens.unsqueeze(0).cuda()).logits.sum()
        generations_hf, _ = GenerationUtils(
            decoder_lm=model_hf, eos_id=tokenizer.eos_id, pad_id=tokenizer.pad_id
        ).generate(
            prompt_tokens=token_for_generation,
            incremental_decode=False,
            min_gen_len=1,
            max_gen_len=64,
            top_p=0,
            top_k=1,
            temperature=1.0,
            device=torch.device("cuda"),
            logits_accessor=lambda o: o.logits,
        )

    # Check that outputs are identical
    assert torch.allclose(decoder_out, hf_out)
    assert torch.allclose(decoder_kv_out, hf_out)

    # Check generation parity
    assert torch.allclose(generations_kv_cache, generations_no_kv_cache)
    assert torch.allclose(generations_kv_cache, generations_hf)
    print("All parity checks passed!", flush=True)
