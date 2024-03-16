# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import List

import torch

from tests.test_utils import fixed_init_model

from torch import nn

from torchtune.models.llama2 import get_lora_module_names, llama2, lora_llama2

try:
    from peft import inject_adapter_in_model, LoraConfig
except:
    raise ImportError("Must have peft installed to run this comparison script")


def compare_lora_llama2(
    bsz: int,
    seq_len: int,
    vocab_size: int,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    embed_dim: int,
    max_seq_len: int,
    lora_modules: List[str],
    lora_in_mlp: bool,
    lora_in_output: bool,
    lora_rank: int,
    lora_alpha: float,
) -> None:

    # make sure we have the right seed for generating outputs
    # this should match up the seed value set in the corresponding
    # unit test
    torch.manual_seed(16)

    # generate input tensor used by both implementations
    x = torch.randint(low=0, high=vocab_size, size=(bsz, seq_len))

    # Our implementation
    lora_llama = lora_llama2(
        lora_attn_modules=lora_modules,
        apply_lora_to_mlp=lora_in_mlp,
        apply_lora_to_output=lora_in_output,
        vocab_size=vocab_size,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        embed_dim=embed_dim,
        max_seq_len=max_seq_len,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
    )
    # This is to make final outputs less trivial
    lora_llama.norm = nn.Identity()
    fixed_init_model(lora_llama)

    with torch.no_grad():
        out = lora_llama(x)

    # Reference implementation: wrap our native llama2 with PEFT LoRAConfig
    llama_ref = llama2(
        vocab_size=vocab_size,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        embed_dim=embed_dim,
        max_seq_len=max_seq_len,
    )

    peft_lora_modules = get_lora_module_names(lora_modules, lora_in_mlp, lora_in_output)

    lora_config_ref = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=0.0,
        r=lora_rank,
        bias="none",
        target_modules=peft_lora_modules,
    )

    lora_llama_ref = inject_adapter_in_model(lora_config_ref, llama_ref)
    lora_llama_ref.norm = nn.Identity()

    mapped_sd = {}
    for k, v in lora_llama.state_dict().items():
        new_k = k.replace("lora_a", "lora_A.default").replace(
            "lora_b", "lora_B.default"
        )
        for attn_module in lora_modules:
            if attn_module in new_k:
                new_k = new_k.replace(
                    attn_module + ".weight", attn_module + ".base_layer.weight"
                )
            if lora_in_mlp and any([f"mlp.w{i}.weight" in new_k for i in range(1, 4)]):
                new_k = new_k.replace(".weight", ".base_layer.weight")

            if lora_in_output and "output.weight" in new_k:
                new_k = new_k.replace(".weight", ".base_layer.weight")

        mapped_sd[new_k] = v

    lora_llama_ref.load_state_dict(mapped_sd)

    with torch.no_grad():
        out_ref = lora_llama_ref(x)

    print(
        lora_modules,
        lora_in_mlp,
        lora_in_output,
        out.mean(),
        out_ref.mean(),
        out.shape,
        out_ref.shape,
    )

    # output tensors should be similar
    torch.testing.assert_close(out, out_ref, atol=1e-5, rtol=1e-3)


if __name__ == "__main__":
    test_cases = [
        (["q_proj", "v_proj"], False, False),
        (["q_proj", "k_proj", "v_proj", "output_proj"], True, False),
        (["k_proj"], True, True),
    ]
    for lora_modules, lora_in_mlp, lora_in_output in test_cases:
        compare_lora_llama2(
            bsz=2,
            seq_len=32,
            vocab_size=50,
            num_layers=3,
            num_heads=4,
            num_kv_heads=2,
            embed_dim=64,
            max_seq_len=64,
            lora_modules=lora_modules,
            lora_in_mlp=lora_in_mlp,
            lora_in_output=lora_in_output,
            lora_rank=4,
            lora_alpha=1.0,
        )
