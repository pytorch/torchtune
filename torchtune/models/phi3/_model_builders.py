from typing import List

from torchtune.models.phi3._component_builders import phi3

from torchtune.modules import TransformerDecoder
from torchtune.modules.tokenizers import SentencePieceTokenizer
from torchtune.modules.peft import LORA_ATTN_MODULES
from functools import partial

import torch


"""
Model builders build specific instantiations using component builders. For example
the ``mistral_7b`` model builder uses the ``mistral`` component builder.
"""


def phi3_mini() -> TransformerDecoder:
    """
    Builder for creating a Mistral 7B model initialized w/ the default 7b parameter values
    from https://mistral.ai/news/announcing-mistral-7b/


    Returns:
        TransformerDecoder: Instantiation of Mistral 7B model
    """
    return phi3(
        vocab_size=32_064,
        num_layers=32,
        num_heads=32,
        num_kv_heads=32,
        embed_dim=3072,
        intermediate_dim=8192,
        max_seq_len=4096,
        attn_dropout=0.0,
        norm_eps=1e-5,
    )

def phi3_tokenizer(path: str) -> SentencePieceTokenizer:
    tokenizer = SentencePieceTokenizer(path)
    # Original tokenizer has no pad_id, which causes indexing errors when batch training
    tokenizer.pad_id = 32000
    return tokenizer
