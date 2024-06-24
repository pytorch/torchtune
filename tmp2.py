# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F


def lora_unmerged(x, alpha, rank, base_weight, lora_a_weight, lora_b_weight):
    base_out = F.linear(x, base_weight)
    scaling = alpha / rank
    lora_out = F.linear(F.linear(x, lora_a_weight), lora_b_weight)
    lora_out *= scaling
    out = base_out + lora_out
    return out


def lora_merged(x, alpha, rank, base_weight, lora_a_weight, lora_b_weight, cast=False):
    scaling = alpha / rank
    if cast:
        base_weight = base_weight.float()
        lora_a_weight = lora_a_weight.float()
        lora_b_weight = lora_b_weight.float()
    lora_weight = lora_b_weight @ lora_a_weight
    lora_weight *= scaling
    merged_weight = base_weight + lora_weight
    if cast:
        merged_weight = merged_weight.to(x.dtype)
    out = F.linear(x, merged_weight)
    return out


@torch.no_grad
def lora_compare(dtype, in_dim=256, out_dim=256, rank=4, bs=2, cast=False):
    kwargs = {
        "x": torch.randn(bs, in_dim, dtype=dtype),
        "alpha": 1.0,
        "rank": rank,
        "base_weight": torch.randn(out_dim, in_dim, dtype=dtype),
        "lora_a_weight": torch.randn(rank, in_dim, dtype=dtype),
        "lora_b_weight": torch.randn(out_dim, rank, dtype=dtype),
    }
    a = lora_unmerged(**kwargs)
    b = lora_merged(cast=cast, **kwargs)
    print(
        F.mse_loss(a.float(), b.float()),
        torch.allclose(a, b, atol=1e-2),
        torch.allclose(a, b, atol=1e-3),
        torch.allclose(a, b, atol=1e-4),
        torch.allclose(a, b, atol=1e-5),
        torch.allclose(a, b, atol=1e-6),
        torch.allclose(a, b, atol=1e-7),
        torch.allclose(a, b, atol=1e-8),
    )


lora_compare(torch.float32)
lora_compare(torch.bfloat16)
lora_compare(torch.bfloat16, cast=True)
