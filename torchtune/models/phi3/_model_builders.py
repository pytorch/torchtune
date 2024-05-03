from typing import List

from torchtune.models.phi3._component_builders import phi3
from torchtune.models.phi3._sentencepiece import Phi3MiniSentencePieceTokenizer

from torchtune.modules import TransformerDecoder
from torchtune.modules.peft import LORA_ATTN_MODULES
from functools import partial

import torch


"""
Model builders build specific instantiations using component builders. For example
the ``phi3_mini`` model builder uses the ``phi3`` component builder.
"""


def phi3_mini() -> TransformerDecoder:
    """
    Builder for creating the Phi3 Mini 4K Instruct Model.
    Ref: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct

    Note:
        This model does not currently support 128K context length nor optimizations
        such as sliding window attention.

    Returns:
        TransformerDecoder: Instantiation of Phi3 Mini 4K Instruct Model
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

def phi3_mini_tokenizer(path: str) -> Phi3MiniSentencePieceTokenizer:
    """Phi-3 Mini tokenizer.
    Ref: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/blob/main/tokenizer_config.json

    Args:
        path (str): Path to the SPM tokenizer model.

    Note:
        This tokenizer includes typical LM EOS and BOS tokens like
        <s>, </s>, and <unk>. However, to support chat completion,
        it is also augmented with special tokens like <|endoftext|>
        and <|assistant|>.

    Warning:
        Microsoft currently opts to ignore system messages citing better performance.
        See https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/discussions/51 for more details.

    Returns:
        Phi3MiniSentencePieceTokenizer: Instantiation of the SPM tokenizer.
    """
    tokenizer = Phi3MiniSentencePieceTokenizer(path)
    return tokenizer
