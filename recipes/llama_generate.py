# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch

from torchtune.models.llama2 import llama2_7b, llama2_tokenizer
from torchtune.utils import get_device, set_seed, TuneArgumentParser
from torchtune.utils.generation import GenerationUtils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def recipe(model_checkpoint, tokenizer_checkpoint, prompt, max_gen_len):
    # Inference setup
    tokenizer = llama2_tokenizer(tokenizer_checkpoint)

    token_for_generation = [tokenizer.encode(prompt, add_eos=False)]

    seed = set_seed()

    device = get_device()

    with device:
        decoder = llama2_7b(max_batch_size=1)

    # Load state_dict into decoder
    native_state_dict = torch.load(model_checkpoint, weights_only=True)
    # Note: If using pretrained model, replace native_state_dict["model"] with native_state_dict
    missing, unexpected = decoder.load_state_dict(native_state_dict, strict=False)

    decoder.eval()

    with torch.no_grad():
        generations, _ = GenerationUtils(
            decoder_lm=decoder,
            eos_id=tokenizer.eos_id,
            pad_id=tokenizer.pad_id,
        ).generate(
            prompt_tokens=token_for_generation,
            incremental_decode=True,
            min_gen_len=1,
            max_gen_len=max_gen_len,
            top_p=0,
            top_k=1,
            temperature=1.0,
            device=device,
        )

        generated_tokens = tokenizer.decode(generations.tolist())
    print(generated_tokens[0])


if __name__ == "__main__":
    parser = TuneArgumentParser(description="Example 7B native Llama-2 inference.")
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        default="/tmp/llama2-7b",
        help="Path to native checkpoint file.",
    )
    parser.add_argument(
        "--tokenizer-checkpoint",
        type=str,
        default="/tmp/tokenizer.model",
        help="Path to tokenization file.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Input to the model",
        required=True,
        # for alpaca format see: https://github.com/tatsu-lab/stanford_alpaca?tab=readme-ov-file#data-release
    )
    parser.add_argument(
        "--max-gen-len",
        type=int,
        default=64,
        help="Max number of tokens to generate",
    )

    kwargs = vars(parser.parse_args())
    recipe(**kwargs)
