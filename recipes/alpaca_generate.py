# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchtune import models
from torchtune.utils import get_device, get_logger, set_seed, TuneArgumentParser
from torchtune.utils.generation import GenerationUtils

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
    model,
    model_checkpoint,
    tokenizer,
    tokenizer_checkpoint,
    instruction,
    input,
    max_gen_len,
):
    logger = get_logger("DEBUG")

    # Inference setup
    tokenizer = models.get_tokenizer(tokenizer, path=tokenizer_checkpoint)

    example = {"instruction": instruction}
    if input != "":
        example["input"] = input
        prompt = PROMPT_DICT["prompt_input"].format_map(example)
    else:
        prompt = PROMPT_DICT["prompt_no_input"].format_map(example)

    token_for_generation = [tokenizer.encode(prompt, add_eos=False)]

    set_seed()

    device = get_device()

    decoder = models.get_model(model, device=device, max_batch_size=1)

    # Load state_dict into decoder
    native_state_dict = torch.load(model_checkpoint, weights_only=True)
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
    logger.info(msg=generated_tokens[0])


if __name__ == "__main__":
    parser = TuneArgumentParser(description="Example 7B native Llama-2 inference.")
    parser.add_argument(
        "--model",
        type=str,
        default="llama2_7b",
        choices=models.list_models(),
        help="Name of the model to finetune.",
    )
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        default="/tmp/llama2-7b",
        help="Path to native checkpoint file.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="llama2_tokenizer",
        choices=models.list_tokenizers(),
        help="Name of the model tokenizer.",
    )
    parser.add_argument(
        "--tokenizer-checkpoint",
        type=str,
        default="/tmp/tokenizer.model",
        help="Path to tokenization file.",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="Answer the question.",
        help="Instruction for model to respond to.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="What is some cool music from the 1920s?",
        help='Additional optional input related to instruction. Pass in "" (empty string) for no input.',
    )
    parser.add_argument(
        "--max-gen-len",
        type=int,
        default=64,
        help="Max number of tokens to generate",
    )

    kwargs = vars(parser.parse_args())
    recipe(**kwargs)
