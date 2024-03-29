# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import sys

import torch
from omegaconf import DictConfig
from typing import Optional

from torchtune import config, utils
from torchtune.utils import get_device, get_logger, set_seed
from torchtune.utils.generation import GenerationUtils

import pdb

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

def multinomial_sample_one_no_sync(probs_sort): # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)

def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    # pdb.set_trace()

    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)

    return probs

def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    probs = logits_to_probs(logits[0, -1], temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    # pdb.set_trace()
    return idx_next, probs


def prefill(model, x, input_pos) -> torch.Tensor:
    # input_pos: [B, S]
    logits = model(x, input_pos)
    # import pdb; pdb.set_trace()
    return sample(logits, temperature=0.8, top_k=200)[0]

def decode_one_token(model, x, input_pos,):
    # input_pos: [B, 1]
    assert input_pos.shape[-1] == 1
    logits = model(x, input_pos)
    # import pdb; pdb.set_trace()
    return sample(logits, temperature=0.8, top_k=200)

def decode_n_tokens(model, cur_token, input_pos, num_new_tokens: int):
    new_tokens, new_probs = [], []
    for i in range(num_new_tokens):
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True): # Actually better for Inductor to codegen attention here
            next_token, next_prob = decode_one_token(
                model, cur_token, input_pos
            )
            input_pos += 1
            new_tokens.append(next_token.clone())
            new_probs.append(next_prob.clone())
            cur_token = next_token.view(1, -1)
    # import pdb; pdb.set_trace()
    return new_tokens, new_probs



def recipe(
    cfg: DictConfig,
):
    logger = get_logger("DEBUG")

    # Inference setup
    tokenizer = config.instantiate(cfg.tokenizer)

    device = utils.get_device(device=cfg.device)
    dtype = utils.get_dtype(dtype=cfg.dtype)
    checkpointer = config.instantiate(cfg.checkpointer)
    checkpoint_dict = checkpointer.load_checkpoint()

    # global decode_one_token
    # decode_one_token = torch.compile(decode_one_token, mode="reduce-overhead", fullgraph=True)

    print("Model init:")
    model = config.instantiate(cfg.model).to(torch.bfloat16).to('cuda').eval()
    model.load_state_dict(checkpoint_dict["model"], strict=False)

    with device:
        model.setup_caches(max_batch_size=1, max_seq_len=308, num_heads=40, head_dim=5120//40, dtype=dtype)

    print(model.causal_mask)

    tokens = tokenizer.encode(cfg.prompt, add_bos=True, add_eos=False)
    prompt = torch.tensor(tokens, dtype=torch.int, device=device)

    torch.manual_seed(1234)

    T = prompt.size(0)
    T_new = T + 200

    empty = torch.empty(T_new, dtype=torch.int, device=device)
    empty[:T] = prompt
    seq = empty
    input_pos = torch.arange(0, T, device=device)

    with torch.no_grad():
        next_token = prefill(model, prompt.view(1, -1), input_pos)
        seq[T] = next_token
        input_pos = torch.tensor([T], device=device, dtype=torch.int)

        generated_tokens, _ = decode_n_tokens(
            model,
            next_token.view(1, -1),
            input_pos,
            200 - 1
        )
        seq[T + 1:] = torch.cat(generated_tokens)

    print(seq)
    print(input_pos)

    print(tokenizer.decode(seq.tolist()))





@config.parse
def main(cfg: DictConfig) -> None:
    recipe(cfg)


if __name__ == "__main__":
    sys.exit(main())
