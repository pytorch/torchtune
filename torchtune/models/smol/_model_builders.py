# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtune.models.smol._component_builders import smollm2

from torchtune.modules import TransformerDecoder

"""
https://huggingface.co/HuggingFaceTB/SmolLM2-135M/

SmolLM2 is a family of compact language models available in three size: 135M, 360M, 
and 1.7B parameters. They are capable of solving a wide range of tasks while being 
lightweight enough to run on-device. More details in our paper: https://arxiv.org/abs/2502.02737v1
"""


def smollm2_135m() -> TransformerDecoder:
    return smollm2(30, 9, 3, 576, 1536)


def smollm2_360m() -> TransformerDecoder:
    return smollm2(32, 15, 5, 960, 2560)


def smollm2_1_7b() -> TransformerDecoder:
    return smollm2(24, 32, 32, 2048, 8192, rope_base=130000)
