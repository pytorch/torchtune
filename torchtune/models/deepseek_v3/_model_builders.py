# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtune.models.deepseek_v3._component_builders import deepseek_v3


def deepseek_v3_6B_64e():
    """
    Builder for a DeepSeek V3 6.1B model with 64 experts.
    https://huggingface.co/smohammadi/deepseek-v3-micro
    """
    return deepseek_v3(
        vocab_size=129280,
        num_layers=16,
        num_heads=32,
        embed_dim=2048,
        max_seq_len=32768,
        mlp_hidden_dim=5632,
        rope_base=10000,
        norm_eps=1e-6,
        moe_every_n_layers=1,
        first_moe_layer=3,
        moe_hidden_dim=1024,
        num_experts=64,
        num_shared_experts=1,
        experts_per_token=8,
        num_groups=8,
        topk_groups=4,
        norm_topk_prob=True,
        routed_scaling_factor=2.5,
        q_lora_rank=256,
        kv_lora_rank=128,
        qk_rope_head_dim=64,
        qk_nope_head_dim=128,
        v_head_dim=128,
        rope_scaling_factor=40.0,
        original_max_seq_len=4096,
        beta_fast=32.0,
        beta_slow=1.0,
        mscale=1.0,
        mscale_all_dim=1.0,
    )
