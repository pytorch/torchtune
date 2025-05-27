# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional

from torchtune.data import PromptTemplate
from torchtune.models.qwen2._tokenizer import (
    DEFAULT_QWEN2_TOKENIZER_BPE_CACHE_SIZE,
    ENDOFTEXT,
    IM_END,
)
from torchtune.models.qwen2_5._tokenizer import QWEN2_5_SPECIAL_TOKENS, Qwen2_5Tokenizer


QWEN3_SPECIAL_TOKENS = {
    **QWEN2_5_SPECIAL_TOKENS,
    "<tool_response>": 151665,
    "</tool_response>": 151666,
    "<think>": 151667,
    "</think>": 151668,
}


# TODO: Refactor the tokenizer classes for the Qwen2/2.5/3 family of models. We should have a single base tokenizer that
# all other tokenizers extend, move all common logic to the base so that the child tokenizers can override the
# behaviour that's specific to them.
class Qwen3Tokenizer(Qwen2_5Tokenizer):  # noqa: N801
    """This tokenizer is simply a wrapper around the Qwen2_5Tokenizer, with a few extra tokens added.

    See <https://github.com/huggingface/transformers/blob/v4.40.1/src/transformers/models/qwen2/tokenization_qwen2.py>
    and <https://huggingface.co/Qwen/Qwen3-8B/blob/main/tokenizer_config.json>.

    Args:
        path (str): Path to vocab.json file.
        merges_file (str): Path to merges.txt file.
            merges.txt contains all BPE merge operations, and this file is required to split a single word into
            byte-level BPE tokens.
        special_tokens (dict[str, int]): Special tokens to add to the tokenizer. Default is QWEN2_5_SPECIAL_TOKENS.
        max_seq_len (Optional[int]): A max sequence length to truncate tokens to.
            Default: None
        prompt_template (Optional[PromptTemplate]): template used to format the messages based on their role. This is used
            to add structured text around the actual messages. The structured text is used in three scenarios:

            - Task-specific templates to gear models for a particular task that it will expect after training
            - Model-specific templates that are required whenever the model is prompted, such as the [INST]
              tags in Llama2 and in Mistral
            - Community standardized templates, such as :class:`~torchtune.data.ChatMLTemplate`

            The extra text will still get tokenized as normal text, not as special tokens.
            Default: None
        errors (str): Paradigm to follow when decoding bytes to UTF-8. Defaults to "replace".
            See [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information.
        unk_token (Optional[str]): The unknown token. A token that is not in the vocabulary cannot be converted
            to an ID and is set to be this token instead. Defaults to ``<|endoftext|>``.
        bos_token (Optional[str]): The beginning of sequence token. Defaults to None.
        eos_token (str): The end of sequence token. Defaults to ``<|endoftext|>``.
        pad_token (Optional[str]): The token used for padding. Defaults to ``<|endoftext|>``.
        bpe_cache_size (int): BPE token cache size in Qwen2Tokenizer.
            NOTE: large cache size will speed up tokenization, but the cache object will get really
            large for long running processes (esp. for texts of language that do not use space between
            word, e.g. Chinese); technically not a memory leak but appears as one.
            By default, we set the cache size equals to size of the official Qwen2 tokenizer.
        truncation_type (str): type of truncation to apply, either "left" or "right".
            Default is "right".

    Example:
        >>> tokenizer = Qwen3Tokenizer(
                path="/path/to/vocab.json", merges_file="/path/to/merges.txt", special_tokens=QWEN3_SPECIAL_TOKENS)
        >>> tokenized_text = tokenizer.encode("Hello world!")
        >>> print(tokenized_text)
        [39, 385, 78, 675, 0, 2000]
    """

    def __init__(
        self,
        path: str,
        merges_file: str,
        special_tokens: dict[str, int] = QWEN3_SPECIAL_TOKENS,
        max_seq_len: Optional[int] = None,
        *,
        prompt_template: Optional[PromptTemplate] = None,
        errors: str = "replace",
        unk_token: Optional[str] = None,
        bos_token: Optional[str] = None,
        eos_token: str = IM_END,
        pad_token: Optional[str] = ENDOFTEXT,
        bpe_cache_size: int = DEFAULT_QWEN2_TOKENIZER_BPE_CACHE_SIZE,
        truncation_type: str = "right",
    ):
        super().__init__(
            path=path,
            merges_file=merges_file,
            special_tokens=special_tokens,
            max_seq_len=max_seq_len,
            prompt_template=prompt_template,
            errors=errors,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            bpe_cache_size=bpe_cache_size,
        )
