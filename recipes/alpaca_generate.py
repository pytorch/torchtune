# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import torch

from torchtune import models
from torchtune.utils import get_device, get_logger, set_seed, TuneArgumentParser
from torchtune.utils.generation import GenerationUtils

from recipes.params.alpaca_generate import AlpacaGenerateParams

# From https://github.com/tatsu-lab/stanford_alpaca/blob/761dc5bfbdeeffa89b8bff5d038781a4055f796a/train.py#L31
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


def recipe(
    params: AlpacaGenerateParams,
):
    logger = get_logger("DEBUG")

    # Inference setup
    tokenizer = models.get_tokenizer(params.tokenizer, path=params.tokenizer_checkpoint)

    example = {"instruction": params.instruction}
    if params.input != "":
        example["input"] = params.input
        prompt = PROMPT_DICT["prompt_input"].format_map(example)
    else:
        prompt = PROMPT_DICT["prompt_no_input"].format_map(example)

    token_for_generation = [tokenizer.encode(prompt, add_eos=False)]

    set_seed()

    device = get_device()

    decoder = models.get_model(params.model, device=device, max_batch_size=1)

    # Load state_dict into decoder
    native_state_dict = torch.load(params.model_checkpoint, weights_only=True)
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
            max_gen_len=params.max_gen_len,
            top_p=0,
            top_k=1,
            temperature=1.0,
            device=device,
        )

        generated_tokens = tokenizer.decode(generations.tolist())
    logger.info(msg=generated_tokens[0])


if __name__ == "__main__":
    parser = TuneArgumentParser(
        description=AlpacaGenerateParams.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Get user-specified args from config and CLI and create params for recipe
    args, _ = parser.parse_known_args()
    args = vars(args)
    params = AlpacaGenerateParams(**args)

    logger = get_logger("DEBUG")
    logger.info(msg=f"Running alpaca_generate.py with parameters {params}")

    recipe(params)
