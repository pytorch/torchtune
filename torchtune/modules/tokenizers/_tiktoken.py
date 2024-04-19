# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Tuple

from tiktoken import Encoding
from tiktoken.load import load_tiktoken_bpe
from torchtune.data._types import Message
from torchtune.modules.tokenizers._utils import (
    _split_long_repetitions,
    Tokenizer,
    truncate,
)


CL100K_PATTERN = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""  # noqa

# bos and eos tokens
BEGIN_OF_TEXT = "<|begin_of_text|>"
END_OF_TEXT = "<|end_of_text|>"
# fill-in-the-middle tags
FIM_PREFIX = "<|fim_prefix|>"
FIM_MIDDLE = "<|fim_middle|>"
FIM_SUFFIX = "<|fim_suffix|>"
# start and end header tokens for formatting chat messages
START_HEADER_ID = "<|start_header_id|>"
END_HEADER_ID = "<|end_header_id|>"
STEP_ID = "<|step_id|>"
# different end of message tags
EOM_ID = "<|eom_id|>"
EOT_ID = "<|eot_id|>"
# special token for ipython messages
PYTHON_TAG = "<|python_tag|>"

ALL_SPECIAL_TOKENS = [
    BEGIN_OF_TEXT,
    END_OF_TEXT,
    FIM_PREFIX,
    FIM_MIDDLE,
    FIM_SUFFIX,
    STEP_ID,
    START_HEADER_ID,
    END_HEADER_ID,
    EOM_ID,
    EOT_ID,
    PYTHON_TAG,
]

PAD_ID = -1

# Constants controlling encode logic
MAX_ENCODE_CHARS = 400_000
MAX_NO_WHITESPACE_CHARS = 25_000


class TikTokenTokenizer(Tokenizer):
    """A wrapper around tiktoken Encoding.

    Args:
        path (str): Path to pretrained tokenizer checkpoint file.
        name (str): Name of the tokenizer (used by tiktoken for identification).
        pattern (str): Regex pattern used to for string parsing.
        all_special_tokens (Optional[List[str]]): List of all special tokens. First element
            must be bos token, second element must be eos token, final element must be
            python tag. All elements must be unique. Length must be at most 256.
            Default: None (will use ALL_SPECIAL_TOKENS)
        bos_token (str): Beginning of sequence token. Defaults to BEGIN_OF_TEXT.
        eos_token (str): End of sequence token. Defaults to END_OF_TEXT.
        start_header_id (str): Start header token. Defaults to START_HEADER_ID.
        end_header_id (str): End header token. Defaults to END_HEADER_ID.
        step_id (str): Step token. Defaults to STEP_ID.
        eom_id (str): End of message token. Defaults to EOM_ID.
        eot_id (str): End of turn token. Defaults to EOT_ID.
        python_tag (str): Python tag token. Defaults to PYTHON_TAG.
    """

    def __init__(
        self,
        path: str,
        *,
        name: str = "llama3_tiktoken",
        pattern: str = CL100K_PATTERN,
        all_special_tokens: Optional[List[str]] = None,
        bos_token: str = BEGIN_OF_TEXT,
        eos_token: str = END_OF_TEXT,
        start_header_id: str = START_HEADER_ID,
        end_header_id: str = END_HEADER_ID,
        step_id: str = STEP_ID,
        eom_id: str = EOM_ID,
        eot_id: str = EOT_ID,
        python_tag: str = PYTHON_TAG,
    ):
        self.path = path
        self.num_reserved_special_tokens = 256
        all_special_tokens = all_special_tokens or ALL_SPECIAL_TOKENS
        self._validate_special_tokens(
            all_special_tokens=all_special_tokens,
            bos_token=bos_token,
            eos_token=eos_token,
            step_id=step_id,
            start_header_id=start_header_id,
            end_header_id=end_header_id,
            eom_id=eom_id,
            eot_id=eot_id,
            python_tag=python_tag,
        )
        self.all_special_tokens = all_special_tokens

        mergeable_ranks = load_tiktoken_bpe(self.path)
        self.base_vocab_size = len(mergeable_ranks)
        all_special_tokens_with_ids = self._get_all_special_tokens_with_ids()
        self.tt_model = Encoding(
            name=name,
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks,
            special_tokens={**all_special_tokens_with_ids},
        )

        # Encode BOS and EOS, define pad ID
        self.bos_id = self._encode_special_token(self.all_special_tokens[0])
        self.eos_id = self._encode_special_token(self.all_special_tokens[1])
        self.pad_id = PAD_ID

        self.vocab_size = self.tt_model.n_vocab

        # Encode extra special tokens
        self.step_id = self._encode_special_token(step_id)
        self.start_header_id = self._encode_special_token(start_header_id)
        self.end_header_id = self._encode_special_token(end_header_id)
        self.eom_id = self._encode_special_token(eom_id)
        self.eot_id = self._encode_special_token(eot_id)
        self.python_tag = self._encode_special_token(python_tag)

    def _validate_special_tokens(
        self,
        *,
        all_special_tokens: List[str],
        bos_token: str,
        eos_token: str,
        step_id: str,
        start_header_id: str,
        end_header_id: str,
        eom_id: str,
        eot_id: str,
        python_tag: str,
    ):
        """
        Validate all the special tokens are as expected. Should satisfy:

        (1) bos_token, eos_token, step_id, start_header_id, end_header_id, eom_id,
            eot_id, python_tag are all in all_special_tokens,
        (2) bos_token should be first, eos_token should be second, python_tag should be last,
        (3) all special tokens are unique, and
        (4) at most 256 special tokens
        """
        for token in [
            bos_token,
            eos_token,
            step_id,
            start_header_id,
            end_header_id,
            eom_id,
            eot_id,
            python_tag,
        ]:
            assert (
                token in all_special_tokens
            ), f"{token} missing from all_special_tokens"
        assert (
            all_special_tokens[0] == bos_token
        ), f"First special token must be bos, got {all_special_tokens[0]}"
        assert (
            all_special_tokens[1] == eos_token
        ), f"Second special token must be eos, got {all_special_tokens[1]}"
        assert (
            all_special_tokens[-1] == python_tag
        ), f"Last special token must be python_tag, got {all_special_tokens[-1]}"
        assert len(set(all_special_tokens)) == len(
            all_special_tokens
        ), "Special tokens must be unique."
        assert (
            len(all_special_tokens) <= self.num_reserved_special_tokens
        ), "The total number of basic and extra special tokens cannot exceed the number of reserved tokens."

    def _get_all_special_tokens_with_ids(self) -> Dict[str, int]:
        """
        Returns a dictionary of all special tokens and their corresponding ids to be passed
        to tiktoken Encoding.

        There are 256 slots for special tokens, any remaining spaces beyond self.all_special_tokens
        will be filled with dummy reserved tokens. Tokens will be added in the order:
        (1) all special tokens but python_tag, (2) all reserved tokens, (3) python_tag.
        """
        reserved_tokens = [
            f"<|reserved_special_token_{i}|>"
            for i in range(
                self.num_reserved_special_tokens - len(self.all_special_tokens)
            )
        ]
        # Python tag special token should come last (validated in __init__)
        all_special_tokens = (
            self.all_special_tokens[:-1]
            + reserved_tokens
            + [self.all_special_tokens[-1]]
        )

        return {
            token: self.base_vocab_size + i
            for i, token in enumerate(all_special_tokens)
        }

    def _encode_special_token(self, token: str) -> int:
        """
        Encodes a special token.

        Args:
            token (str): The special token to encode.

        Returns:
            int: The encoded special token.
        """
        return self.tt_model.encode(
            token,
            allowed_special="all",
            disallowed_special=(),
        )[0]

    def encode(
        self,
        text: str,
        add_bos: bool,
        add_eos: bool,
    ) -> List[int]:
        """
        Encode a string into a list of token ids. Assumes that the string
        contains no special tokens.

        Args:
            text (str): The string to encode.
            add_bos (bool): Whether to add the beginning of sequence token.
            add_eos (bool): Whether to add the end of sequence token.

        Returns:
            List[int]: The list of token ids.
        """
        substrs: List[str] = []
        tokens = []
        for i in range(0, len(text), MAX_ENCODE_CHARS):
            substr = text[i : i + MAX_ENCODE_CHARS]
            # See https://github.com/openai/tiktoken/issues/195
            sliced_substr = _split_long_repetitions(substr, MAX_NO_WHITESPACE_CHARS)
            substrs.extend(sliced_substr)
        for substr in substrs:
            # allowed_special and disallowed_special are used by tiktoken to define
            # how special tokens are encoded. Our setting here is to encode any
            # special token as regular text and prevent tiktoken from raising errors.
            # This means we should only call encode on strings not containing special tokens.
            tokens.extend(
                self.tt_model.encode(
                    substr,
                    allowed_special=set(),
                    disallowed_special=(),
                )
            )
        if add_bos:
            tokens.insert(0, self.bos_id)
        if add_eos:
            tokens.append(self.eos_id)
        return tokens

    def decode(
        self,
        token_ids: List[int],
        truncate_at_eos: bool = True,
    ) -> str:
        """
        Decode a list of token ids into a string.

        Args:
            token_ids (List[int]): The list of token ids.
            truncate_at_eos (bool): Whether to truncate the string at the end of
                sequence token.

        Returns:
            str: The decoded string.
        """
        if truncate_at_eos:
            try:
                k = token_ids.index(self.eos_id)
            except ValueError:
                k = None
            if k:
                token_ids = token_ids[:k]
        token_ids = [token_id for token_id in token_ids if token_id != self.bos_id]
        return self.tt_model.decode(token_ids)

    def tokenize_message(
        self, message: Message, tokenize_header: bool = False
    ) -> List[int]:
        """
        Tokenize a message into a list of token ids.

        Args:
            message (Message): The message to tokenize.
            tokenize_header (bool): Whether to prepend a tokenized header to each message.

        Returns:
            List[int]: The list of token ids.
        """
        if tokenize_header:
            tokenized_header = (
                [self.start_header_id]
                + self.encode(message.role.strip(), add_bos=False, add_eos=False)
                + [self.end_header_id]
                + self.encode("\n\n", add_bos=False, add_eos=False)
            )
        else:
            tokenized_header = []
        tokenized_body = self.encode(
            message.content.strip(), add_bos=False, add_eos=False
        )
        if message.ipython:
            tokenized_body = [self.python_tag] + tokenized_body
        tokenized_message = tokenized_header + tokenized_body
        if message.eot:
            tokenized_message = tokenized_message + [self.eot_id]
        else:
            tokenized_message = tokenized_message + [self.eom_id]
        return tokenized_message

    def tokenize_messages(
        self,
        messages: List[Message],
        max_seq_len: Optional[int] = None,
        tokenize_header: bool = True,
    ) -> Tuple[List[int], List[bool]]:
        """
        Tokenize a list of messages into a list of token ids and masks.

        Args:
            messages (List[Message]): The list of messages to tokenize.
            max_seq_len (Optional[int]): The maximum sequence length.
            tokenize_header (bool): Whether to prepend a tokenized header to each message.

        Returns:
            Tuple[List[int], List[bool]]: The list of token ids and the list of masks.
        """
        tokens = [self.bos_id]
        # bos and eos are always masked
        mask = [True]
        for message in messages:
            tokenized_message = self.tokenize_message(
                message, tokenize_header=tokenize_header
            )
            tokens = tokens + tokenized_message
            mask = mask + ([message.masked] * len(tokenized_message))
            if max_seq_len and len(tokens) >= max_seq_len:
                break
        tokens = tokens + [self.eos_id]
        mask = mask + [True]
        if max_seq_len:
            tokens = truncate(tokens, max_seq_len, self.eos_id)
            mask = truncate(mask, max_seq_len, True)
        return tokens, mask
