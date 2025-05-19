# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import List, Optional

from torchtune.data._prompt_templates import _get_prompt_template, _TemplateType

from torchtune.models.qwen2._component_builders import lora_qwen2, qwen2
from torchtune.models.qwen2_5._tokenizer import QWEN2_5_SPECIAL_TOKENS, Qwen2_5Tokenizer
from torchtune.modules import TransformerDecoder
from torchtune.modules.model_fusion import EarlyFusionModel
from torchtune.modules.peft import LORA_ATTN_MODULES
from torchtune.modules.transforms.tokenizers import parse_hf_tokenizer_json

"""
Model builders build specific instantiations using component builders. For example
the qwen2_5_7b model builder uses the qwen2 component builder to create the
Qwen2.5 7B model.
"""



def qwen2_5_vl_7b_base() -> EarlyFusionModel:
    """
    Builder for creating a Qwen2.5 7B base model with vision.
    """

    decoder = qwen2(
        vocab_size=152064,
        num_layers=28,
        num_heads=28,
        num_kv_heads=4,
        embed_dim=3584,
        intermediate_dim=18944,
        max_seq_len=32768,
        attn_dropout=0.0,
        norm_eps=1e-6,
        rope_base=1000000.0,
    )

    encoder = None

    return EarlyFusionModel(
        
    )


