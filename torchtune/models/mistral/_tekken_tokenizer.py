# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from enum import Enum
from functools import cached_property
from itertools import groupby
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import tiktoken

from torchtune.data import Message, PromptTemplate
from torchtune.models.mistral._prompt_template import MistralChatTemplate
from torchtune.modules.transforms import Transform
from torchtune.modules.transforms.tokenizers import (
    ModelTokenizer,
    tokenize_messages_no_special_tokens,
)

WHITESPACE_CHARS = [" ", "\n", "\t", "\r", "\v"]


class SpecialTokenPolicy(Enum):
    """What to do with special tokens when encoding/decoding."""
    IGNORE = 0
    KEEP = 1
    RAISE = 2


def _reload_mergeable_ranks(vocab, max_vocab=None):
    """Reload mergeable ranks from vocab."""
    token2id = {}
    for i, token_info in enumerate(vocab):
        if max_vocab is not None and i >= max_vocab:
            break
        token_bytes = token_info["token_bytes"]
        # Decode base64 to bytes
        import base64
        token_bytes = base64.b64decode(token_bytes)
        token2id[token_bytes] = i
    return token2id


class MistralTekkenTokenizer(ModelTokenizer, Transform):
    """
    Mistral's implementation of the Tekken tokenizer for Nemo models

    Args:
        path (str): Path to pretrained tokenizer file (tekken.json).
        max_seq_len (Optional[int]): A max sequence length to truncate tokens to.
            Default: None
        prompt_template (Optional[PromptTemplate]): template used to format the messages based on their role.
            Default is :class:`~torchtune.models.mistral.MistralChatTemplate`.
        truncation_type (str): type of truncation to apply, either "left" or "right".
            Default is "right".

    Examples:
        >>> tokenizer = MistralTekkenTokenizer("/path/to/tekken.json")
        >>> tokenized_text = tokenizer.encode("Hello world!", add_bos=True, add_eos=True)
        >>> print(tokenized_text)
        [1, 31587, 29644, 102, 2]
    """

    # Special tokens for Tekken tokenizer
    SPECIAL_TOKENS = (
        "<unk>",
        "<s>",
        "</s>",
        "<begin_inst>",
        "<end_inst>",
        "<begin_tools>",
        "<end_tools>",
        "<begin_tool_results>",
        "<end_tool_results>",
        "<tool_calls>",
        "<img>",
        "<pad>",
        "<img_break>",
        "<img_end>",
        "<prefix>",
        "<middle>",
        "<suffix>",
        "<begin_system>",
        "<end_system>",
        "<begin_tool_content>",
    )
    SPECIAL_TOKEN_TEMPLATE = "<SPECIAL_{id}>"

    def __init__(
        self,
        path: str,
        max_seq_len: Optional[int] = None,
        prompt_template: Optional[PromptTemplate] = MistralChatTemplate(),
        truncation_type: str = "right",
    ):
        # Load the tokenizer configuration from the JSON file
        with open(path, "r") as f:
            model_data = json.load(f)
        
        # Extract configuration
        vocab = model_data["vocab"]
        pattern = model_data["config"]["pattern"]
        vocab_size = model_data["config"]["default_vocab_size"]
        num_special_tokens = model_data["config"]["default_num_special_tokens"]
        
        # Validate vocab size
        assert vocab_size <= len(vocab) + num_special_tokens, (
            vocab_size, len(vocab), num_special_tokens,
        )
        
        self._vocab_size = vocab_size
        self._path = path
        
        # Set up special tokens
        special_tokens = list(self.SPECIAL_TOKENS)
        assert len(special_tokens) == len(set(special_tokens)), f"Special tokens must be unique: {special_tokens}"
        assert len(special_tokens) < num_special_tokens
        
        # Add filler special tokens if needed
        special_filler = [
            self.SPECIAL_TOKEN_TEMPLATE.format(id=i)
            for i in range(len(special_tokens), num_special_tokens)
        ]
        
        if special_filler:
            print(f"Adding special tokens {special_filler[0]}, ..., {special_filler[-1]}")
        
        special_tokens = special_tokens + special_filler
        assert len(set(special_tokens)) == len(special_tokens) == num_special_tokens, special_tokens
        
        # Calculate inner vocab size
        inner_vocab_size = vocab_size - num_special_tokens
        
        # Reload vocab
        self._tekken_token2id_nospecial = _reload_mergeable_ranks(vocab, max_vocab=inner_vocab_size)
        assert set(range(inner_vocab_size)) == set(self._tekken_token2id_nospecial.values()), (
            inner_vocab_size, self._tekken_token2id_nospecial,
        )
        
        # Initialize tiktoken model
        self._model = tiktoken.Encoding(
            name=Path(path).stem,
            pat_str=pattern,
            mergeable_ranks=self._tekken_token2id_nospecial,
            special_tokens={},  # special tokens are handled manually
        )
        
        self._all_special_tokens = special_tokens
        self._vocab = [self.id_to_piece(i) for i in range(vocab_size)]
        self._special_token_policy = SpecialTokenPolicy.RAISE
        
        # TorchTune specific attributes
        self.max_seq_len = max_seq_len
        self.prompt_template = prompt_template
        self.truncation_type = truncation_type
        
        # During generation, stop when eos_id is encountered
        self.stop_tokens = [self.eos_id]

    def id_to_piece(self, token_id: int) -> str:
        """Convert token ID to string representation."""
        if token_id < len(self._all_special_tokens):
            return self._all_special_tokens[token_id]
        
        try:
            return self._model.decode([token_id - len(self._all_special_tokens)])
        except:
            return "<?>"

    @property
    def eos_id(self) -> int:
        """Get the end-of-sequence token ID."""
        return self.SPECIAL_TOKENS.index("</s>")

    @property
    def bos_id(self) -> int:
        """Get the beginning-of-sequence token ID."""
        return self.SPECIAL_TOKENS.index("<s>")

    @property
    def pad_id(self) -> int:
        """Get the padding token ID."""
        return self.SPECIAL_TOKENS.index("<pad>")

    @property
    def unk_id(self) -> int:
        """Get the unknown token ID."""
        return self.SPECIAL_TOKENS.index("<unk>")

    @property
    def vocab_size(self) -> int:
        """Get the vocabulary size."""
        return self._vocab_size

    @property
    def num_special_tokens(self) -> int:
        """Get the number of special tokens."""
        return len(self._all_special_tokens)

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
        trim_leading_whitespace: bool = False,
    ) -> List[int]:
        """
        Encode a string into a list of token IDs

        Args:
            text (str): The input text to be encoded, unbatched.
            add_bos (bool): Whether to prepend BOS special token (Beginning of Sentence) to the input, defaults to True.
            add_eos (bool): Whether to append EOS special token (End of Sentence) to the input, defaults to True.
            trim_leading_whitespace (bool): Whether to trim leading whitespace, defaults to False.
        Returns:
            List[int]: The encoded token IDs.
        """
        if trim_leading_whitespace and text and text[0] in WHITESPACE_CHARS:
            text = text.lstrip()
            
        tokens: List[int] = self._model.encode(text)
        tokens = [t + self.num_special_tokens for t in tokens]
        
        if add_bos:
            tokens = [self.bos_id, *tokens]
        if add_eos:
            tokens = [*tokens, self.eos_id]
            
        return tokens

    def decode(
        self,
        token_ids: List[int],
    ) -> str:
        """Decode token IDs to strings.

        Args:
            token_ids (List[int]): The input token IDs to be decoded.

        Returns:
            str: The decoded text.
        """
        return "".join(self._decode_all(token_ids, self._special_token_policy))

    def _decode_all(self, tokens: List[int], special_token_policy: SpecialTokenPolicy) -> List[str]:
        """Decode tokens with special token handling."""
        # Lump special and non-special tokens together to minimize calls to decode
        decoded: List[str] = []
        for is_special, group in groupby(tokens, lambda t: t < self.num_special_tokens):
            group_list = list(group)
            if is_special:
                if special_token_policy == SpecialTokenPolicy.RAISE:
                    raise ValueError(
                        f"Decoding `tokens` that contain special tokens ({group_list}) is not allowed. \n"
                        "Either make sure `tokens` do not include any special tokens or, "
                        "if you want to decode `tokens` that includes special tokens, "
                        "change the tokenizer's special token policy to IGNORE or KEEP."
                    )
                elif special_token_policy == SpecialTokenPolicy.KEEP:
                    decoded.extend(self._all_special_tokens[t] for t in group_list)
                elif special_token_policy == SpecialTokenPolicy.IGNORE:
                    continue
            else:
                decoded.append(self._model.decode([t - self.num_special_tokens for t in group_list]))
        return decoded

    def tokenize_messages(
        self,
        messages: List[Message],
        *,
        add_eos: bool = True,
    ) -> Tuple[List[int], List[bool]]:
        r"""Tokenize a list of messages one at a time then concatenate them,
        returning a list of tokens and a list of masks.

        Args:
            messages (List[Message]): A list of messages, each containing role, content,
                and masked attributes.
            add_eos (bool): Whether to append EOS after assistant message, default to True

        Returns:
            Tuple[List[int], List[bool]]: The tokenized messages
        """
        templated_messages = (
            self.prompt_template(messages)
            if self.prompt_template is not None
            else messages
        )
        return tokenize_messages_no_special_tokens(
            tokenizer=self,
            messages=templated_messages,
            bos_id=self.bos_id,
            eos_id=self.eos_id if add_eos else None,
            truncation_type=self.truncation_type,
        )

    def __call__(
        self, sample: Mapping[str, Any], inference: bool = False
    ) -> Mapping[str, Any]:
        """
        Apply ``tokenize_messages`` to the "messages" field in the sample.

        Args:
            sample (Mapping[str, Any]): A sample with a "messages" field containing
                a List[Message] to tokenize
            inference (bool): Whether the template is being used for inference or not.

        Returns:
            Mapping[str, Any]: The sample with added "tokens" and "mask" fields
                and the "messages" field removed.
            inference (bool): Whether the template is being used for inference or not.
        """
        messages = sample.pop("messages")
        tokens, mask = self.tokenize_messages(messages)
        sample["tokens"] = tokens
        sample["mask"] = mask
        return sample
