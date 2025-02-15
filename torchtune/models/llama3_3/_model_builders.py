# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional

from torchtune.models.llama3_3._tokenizer import Llama3_3Tokenizer
from torchtune.data._prompt_templates import _get_prompt_template, _TemplateType
from torchtune.modules.transforms.tokenizers import parse_hf_tokenizer_json
from torchtune.models.llama3_1._model_builders import (
    llama3_1_70b,
    lora_llama3_1_70b,
    qlora_llama3_1_70b,
)

"""
Model builders build specific instantiations using component builders. The Llama3.3 model
builders all call the Llama3.1 models as they're identical models apart from the checkpoints.
"""

llama3_3_70b = llama3_1_70b

llama3_3_70b.__doc__ = """
Builder for creating a Llama3.3 model initialized w/ the default 70B parameter values.
Please see `llama3_1_70b` for full API arguments.
"""

lora_llama3_3_70b = lora_llama3_1_70b

lora_llama3_3_70b.__doc__ = """
Builder for creating a Llama3.3 70B model with LoRA enabled.
Please see `lora_llama3_1_70b` for full API arguments.
"""

qlora_llama3_3_70b = qlora_llama3_1_70b

qlora_llama3_1_70b.__doc__ = """
Builder for creating a Llama3.3 70B model with QLoRA enabled. Base model weights in linear layers
that LoRA is applied to are quantized per the QLoRA paper: https://arxiv.org/abs/2305.14314.
Please see `lora_llama3_1_70b` for full API arguments.
"""

def llama3_3_tokenizer(
    path: str,
    special_tokens_path: Optional[str] = None,
    max_seq_len: Optional[int] = None,
    prompt_template: Optional[_TemplateType] = None,
) -> Llama3_3Tokenizer:
    """
    Tokenizer for Llama3.3.

    Args:
        path (str): path to the tokenizer
        special_tokens_path (Optional[str]): Path to ``tokenizer.json`` from Hugging Face
            model files that contains all registered special tokens, or a local json file
            structured similarly. Default is None to use the canonical Llama3 special tokens.
        max_seq_len (Optional[int]): maximum sequence length for tokenizing a single list of messages,
            after which the input will be truncated. Default is None.
        prompt_template (Optional[_TemplateType]): optional specified prompt template.
            If a string, it is assumed to be the dotpath of a :class:`~torchtune.data.PromptTemplateInterface`
            class. If a dictionary, it is assumed to be a custom prompt template mapping role to the
            prepend/append tags.

    Returns:
        Llama3_3Tokenizer: Instantiation of the Llama3.3 tokenizer
    """
    special_tokens = (
        parse_hf_tokenizer_json(special_tokens_path)
        if special_tokens_path is not None
        else None
    )
    template = (
        _get_prompt_template(prompt_template) if prompt_template is not None else None
    )
    return Llama3_3Tokenizer(
        path=path,
        special_tokens=special_tokens,
        max_seq_len=max_seq_len,
        prompt_template=template,
    )
