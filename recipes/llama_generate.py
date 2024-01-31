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
from transformers import LlamaForCausalLM,LlamaTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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


def recipe(model_checkpoint, tokenizer_checkpoint, max_gen_len, instruction, input):
    # Inference setup
    tokenizer = llama2_tokenizer(tokenizer_checkpoint)

    seed = set_seed()

    device = get_device()

    # with device:
    #     decoder = llama2_7b(max_batch_size=1)

    # # Load state_dict into decoder
    # native_state_dict = torch.load(model_checkpoint, weights_only=True)
    # if (
    #     "model" in native_state_dict.keys()
    # ):  # finetuned model is a dict with "model" key
    #     native_state_dict = native_state_dict["model"]
    # missing, unexpected = decoder.load_state_dict(native_state_dict, strict=False)

    # decoder.eval()
    example = {"instruction": instruction}
    if input != "":
        example["input"] = input
    if input != "":
        prompt = PROMPT_DICT["prompt_input"].format_map(example)
    else:
        prompt = PROMPT_DICT["prompt_no_input"].format_map(example)
    print(f"RV: got prompt {prompt}", flush=True)
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b", use_auth_token="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",token=None)
    model_inputs = tokenizer(prompt, return_tensors='pt').to(device)

    with device:
        model_hf = LlamaForCausalLM.from_pretrained("/tmp/foo", use_auth_token="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",token=None)

    decoder = model_hf
    decoder.eval()
    # generate 40 new tokens
    greedy_output = model_hf.generate(**model_inputs, max_new_tokens=40)

    print("Output:\n" + 100 * '-')
    print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))
    import pdb ; pdb.set_trace()
# generate 40 new tokens
# greedy_output = model.generate(**model_inputs, max_new_tokens=40)

# print("Output:\n" + 100 * '-')
# print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))
#     import pdb ; pdb.set_trace()
    with torch.no_grad():
        generations, _ = GenerationUtils(
            decoder_lm=decoder,
            eos_id=tokenizer.eos_id,
            pad_id=tokenizer.pad_id,
        ).generate(
            prompt_tokens=token_for_generation,
            incremental_decode=False,
            min_gen_len=1,
            max_gen_len=max_gen_len,
            top_p=0,
            top_k=1,
            temperature=1.0,
            device=device,
            logits_accessor=lambda m: m.logits,
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
        "--max-gen-len",
        type=int,
        default=64,
        help="Max number of tokens to generate",
    )
    parser.add_argument("--instruction", type=str, default="Answer the question.", help="Instruction for model to respond to.")
    parser.add_argument("--input", type=str, default="What is some cool music from the 1920s?", help="Additional optional input related to instruction. Pass in \"\" (empty string) for no input.")

    kwargs = vars(parser.parse_args())
    recipe(**kwargs)
