# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging

import torch

from torchtune.models.llama2.models import llama2_7b, llama2_tokenizer
from torchtune.utils.env import _get_device_from_env, seed
from torchtune.utils.generation import GenerationUtils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConvertNewlinesAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        # Convert the string "\\n" to an actual newline character "\n"
        # Needed to allow passing newlines into argparse
        values = values.replace("\\n", "\n")
        setattr(namespace, self.dest, values)


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
    parser.add_argument(
        "--prompt",
        action=ConvertNewlinesAction,
        type=str,
        # example prompt for the alpaca dataset:
        # "Below is an instruction that describes a task, paired with an input that provides further context. "
        # "Write a response that appropriately completes the request.\n\n### Instruction:\nAnswer the question.\n\n### "
        # "Input:\nWhat is some cool music from the 1920s?\n\n### Response:"
        help="Input to the model",
    )
    parser.add_argument(
        "--max-gen-len",
        type=int,
        default=64,
        help="Max number of tokens to generate",
    )

    args = parser.parse_args()
    # Inference setup
    tokenizer = llama2_tokenizer(args.tokenizer_path)

    token_for_generation = [tokenizer.encode(args.prompt, add_eos=False)]

    seed(0)

    device = _get_device_from_env()
    # --------- Initialize a decoder w/o kv-caching -------- #
    with device:
        decoder = llama2_7b(vocab_size=tokenizer.vocab_size, max_batch_size=1)

    # Load state_dict into decoder
    native_state_dict = torch.load(args.native_checkpoint_path, weights_only=True)
    # Note: If using pretrained model, replace native_state_dict["model"] with native_state_dict
    missing, unexpected = decoder.load_state_dict(
        native_state_dict["model"], strict=False
    )

    decoder.eval()

    with torch.no_grad():
        generations_no_kv_cache, _ = GenerationUtils(
            decoder_lm=decoder,
            eos_id=tokenizer.eos_id,
            pad_id=tokenizer.pad_id,
        ).generate(
            prompt_tokens=token_for_generation,
            incremental_decode=True,
            min_gen_len=1,
            max_gen_len=args.max_gen_len,
            top_p=0,
            top_k=1,
            temperature=1.0,
            device=device,
        )

        generated_tokens = tokenizer.decode(generations_no_kv_cache.tolist())
    print(generated_tokens[0])
