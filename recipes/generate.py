# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging

import torch

from torchtune.models.llama2.models import llama2_7b, llama2_tokenizer
from torchtune.utils.env import seed
from torchtune.utils.generation import GenerationUtils

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
    args = parser.parse_args()

    # Inference setup
    tokenizer = llama2_tokenizer(args.tokenizer_path)

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

    seed(0)

    # --------- Initialize a decoder w/o kv-caching -------- #
    with torch.device("cuda"):
        decoder = llama2_7b(vocab_size=tokenizer.vocab_size)

    # Load state_dict into decoder
    native_state_dict = torch.load(args.native_checkpoint_path, weights_only=True)
    # Note: If using model finetuned with finetune_llm recipe, replace native_state_dict with native_state_dict["model"]
    missing, unexpected = decoder.load_state_dict(native_state_dict, strict=False)
    # Nothing should be missing or unexpected
    assert not missing, missing
    assert not unexpected, unexpected
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
        print(generations_no_kv_cache)
        generated_tokens = tokenizer.decode(generations_no_kv_cache.tolist()[0:1])
    print(generated_tokens)
