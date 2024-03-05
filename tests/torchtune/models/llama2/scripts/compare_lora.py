# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F
from tests.test_utils import fixed_init_model
from torch import nn
from torchtune.modules.peft.lora import LoRALinear

# Reference implementation from
# https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
class LoRALayer:
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class LoRALinearRef(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
        )

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize B the same way as the default for nn.Linear and A to zero
            # this is different than what is described in the paper but should not affect performance
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        def T(w):  # noqa
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):  # noqa
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            result += (
                self.lora_dropout(x)
                @ self.lora_A.transpose(0, 1)
                @ self.lora_B.transpose(0, 1)
            ) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)


def compare_lora(
    bsz: int,
    seq_len: int,
    in_dim: int,
    out_dim: int,
    rank: int,
    alpha: float,
    dropout: float,
) -> None:

    # make sure we have the right seed for generating outputs
    # this should match up the seed value set in the corresponding
    # unit test
    torch.manual_seed(16)

    # generate input tensor used by both implementations
    x_input = torch.randn(bsz, seq_len, in_dim)

    # Initialize our implementation
    lora = LoRALinear(
        in_dim=in_dim,
        out_dim=out_dim,
        rank=rank,
        alpha=alpha,
        use_bias=True,
        dropout=dropout,
    )
    fixed_init_model(lora)

    with torch.no_grad():
        output = lora(x_input)

    # Initialize reference implementation
    lora_ref = LoRALinearRef(
        in_features=in_dim,
        out_features=out_dim,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
    )

    sd_mapping = {
        "weight": "weight",
        "bias": "bias",
        "lora_a.weight": "lora_A",
        "lora_b.weight": "lora_B",
    }
    mapped_sd = {sd_mapping.get(k): v for k, v in lora.state_dict().items()}
    lora_ref.load_state_dict(mapped_sd)
    with torch.no_grad():
        output_ref = lora_ref(x_input)

    print(output_ref.mean())
    torch.testing.assert_close(output_ref, output, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare LoRA linear implementations")
    parser.add_argument("--bsz", type=int, default=2, help="Batch size of input tensor")
    parser.add_argument("--seq_len", type=int, default=32, help="Input sequence length")
    parser.add_argument(
        "--in_dim",
        type=int,
        default=64,
        help="Input embedding dimension",
    )
    parser.add_argument(
        "--out_dim",
        type=int,
        default=128,
        help="Input embedding dimension",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help="Rank of LoRA's A and B matrices",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Scaling factor for LoRA matrices",
    )
    parser.add_argument(
        "--dropout", type=int, default=0.0, help="Dropout prob after linear layer"
    )

    args = parser.parse_args()

    compare_lora(
        args.bsz,
        args.seq_len,
        args.in_dim,
        args.out_dim,
        args.rank,
        args.alpha,
        args.dropout,
    )
