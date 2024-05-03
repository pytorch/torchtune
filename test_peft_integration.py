# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# Not for land, just for testing
import runpy
import sys

import torch
from tests.common import TUNE_PATH
from torch import nn
from torchtune.models.llama2 import llama2_7b
from torchtune.utils import FullModelHFCheckpointer
from transformers import AutoModelForCausalLM


def test_permute():
    # Defaults for Llama2-7B
    num_heads = 32
    num_kv_heads = 16
    embed_dim = 4096
    rank = 8
    # Try for both Q and K with GQA
    for n_heads in [num_heads, num_kv_heads]:
        head_dim = embed_dim // n_heads

        # Original permute method
        def _permute(t, n_heads):
            return (
                t.view(n_heads, head_dim // 2, 2, embed_dim)
                .transpose(1, 2)
                .reshape((head_dim * n_heads), embed_dim)
            )

        # Modified permute method for LoRA B matrix
        def _permute_lora_matrix(t, n_heads):
            return (
                t.view(n_heads, head_dim // 2, 2, rank)
                .transpose(1, 2)
                .reshape((head_dim * n_heads), rank)
            )

        orig_a = nn.Linear(embed_dim, rank, bias=False)
        orig_b = nn.Linear(rank, n_heads * head_dim, bias=False)
        orig_full = nn.Linear(embed_dim, n_heads * head_dim, bias=False)
        orig_full.weight = nn.Parameter(orig_b.weight @ orig_a.weight)

        remapped_weight = _permute(orig_full.weight, n_heads)
        remapped_lora_b = _permute_lora_matrix(orig_b.weight, n_heads)
        remapped_lora = remapped_lora_b @ orig_a.weight
        print(torch.max(torch.abs(remapped_weight - remapped_lora)))


def run_lora_finetune():
    tune_cmd = """tune run lora_finetune_single_device \
    --config llama2/7B_lora_single_device \
    gradient_accumulation_steps=1 \
    max_steps_per_epoch=100 \
    checkpointer.output_dir=/data/users/ebs/test_peft_integration \
    """.split()
    sys.argv = tune_cmd
    runpy.run_path(TUNE_PATH, run_name="__main__")


# First run run_lora_finetune and upload to hub
# via huggingface-cli upload ebsmothers/test-peft /data/users/ebs/test_peft_integration
def test_peft_integration():
    model_id = "meta-llama/Llama-2-7b-hf"
    peft_model_id = "ebsmothers/test-peft"

    peft_model = AutoModelForCausalLM.from_pretrained(model_id)
    peft_model.load_adapter(peft_model_id)

    vocab_size, bsz, seq_len = 32000, 2, 128
    inputs = torch.randint(0, vocab_size, (bsz, seq_len))

    tt_model = llama2_7b()
    checkpointer = FullModelHFCheckpointer(
        checkpoint_dir="/data/users/ebs/test_peft_integration",
        checkpoint_files=["hf_model_0001_0.pt", "hf_model_0002_0.pt"],
        model_type="LLAMA2",
        output_dir="/data/users/ebs",
        # adapter_checkpoint='adapter_0.pt',
    )
    checkpoint_dict = checkpointer.load_checkpoint()
    tt_model.load_state_dict(checkpoint_dict["model"])
    # tt_model.load_state_dict(checkpoint_dict["adapter"], strict=False)

    tt_model.eval()
    peft_model.eval()

    peft_out = model(inputs)
    tt_out = tt_model(inputs)
    print(torch.max(torch.abs(peft_out.logits - tt_out)))


if __name__ == "__main__":
    # test_permute()
    test_peft_integration()
