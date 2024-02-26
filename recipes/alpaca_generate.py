# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from omegaconf import DictConfig

from torchtune import config
from torchtune.utils import get_device, get_logger, set_seed
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


@config.parse
def recipe(
    cfg: DictConfig,
):
    logger = get_logger("DEBUG")

    # Inference setup
    tokenizer = config.instantiate(cfg.tokenizer)

    example = {"instruction": cfg.instruction}
    if cfg.input != "":
        example["input"] = cfg.input
        prompt = PROMPT_DICT["prompt_input"].format_map(example)
    else:
        prompt = PROMPT_DICT["prompt_no_input"].format_map(example)

    token_for_generation = [tokenizer.encode(prompt, add_eos=False)]

    set_seed()

    device = get_device()

    with device:
        decoder = config.instantiate(cfg.model, max_batch_size=1)

    # Load state_dict into decoder
    native_state_dict = torch.load(cfg.model_checkpoint, weights_only=True)
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
            max_gen_len=cfg.max_gen_len,
            top_p=0,
            top_k=1,
            temperature=1.0,
            device=device,
        )

        generated_tokens = tokenizer.decode(generations.tolist())
    logger.info(msg=generated_tokens[0])
