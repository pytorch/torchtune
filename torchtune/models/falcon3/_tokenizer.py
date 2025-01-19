# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import json
import unicodedata
from functools import lru_cache
from typing import Any, Dict, List, Mapping, Optional, Tuple

import regex as re

from torchtune.data import ChatMLTemplate, Message, PromptTemplate, truncate
from torchtune.modules.tokenizers import ModelTokenizer

PRETOKENIZE_REGEX = (
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
FALCON3_SPECIAL_TOKENS = {
    ">>TITLE<<": 0,
    ">>ABSTRACT<<": 1,
    ">>INTRODUCTION<<": 2,
    ">>SUMMARY<<": 3,
    ">>COMMENT<<": 4,
    ">>ANSWER<<": 5,
    ">>QUESTION<<": 6,
    ">>DOMAIN<<": 7,
    ">>EMAIL_ADDRESS<<": 8,
    ">>IP_ADDRESS<<": 9,
    "<|startoftext|>": 10,
    "<|endoftext|>": 11,
    ">>IP_ADDRESS_0<<": 14,
    ">>IP_ADDRESS_1<<": 15,
    ">>IP_ADDRESS_2<<": 16,
    ">>IP_ADDRESS_3<<": 17,
    ">>IP_ADDRESS_4<<": 18,
    ">>IP_ADDRESS_5<<": 19,
    ">>IP_ADDRESS_6<<": 20,
    ">>IP_ADDRESS_7<<": 21,
    ">>IP_ADDRESS_8<<": 22,
    ">>IP_ADDRESS_9<<": 23,
    ">>PASSWORD<<": 24,
    ">>KEY<<": 25
}

ENDOFTEXT = "<|endoftext|>"

DEFAULT_FALCON3_TOKENIZER_BPE_CACHE_SIZE = 152064


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoid mapping to whitespace/control
    characters the bpe code barfs on.

    The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab
    if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for
    decent coverage. This is a significant percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup
    tables between utf-8 bytes and unicode strings.
    """
    bs = (
            list(range(ord("!"), ord("~") + 1))
            + list(range(ord("¡"), ord("¬") + 1))
            + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """
    Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class Falcon3Tokenizer(ModelTokenizer):
    """This class construct a Falcon3 tokenizer, based on GPT-2 byte-level BPE tokenization.

    See <https://github.com/huggingface/transformers/blob/v4.40.1/src/transformers/models/falcon3/tokenization_falcon3.py>.

    Args:
        path (str): Path to tokenizer.json file.
        special_tokens (Dict[str, int]): Special tokens to add to the tokenizer. Default is FALCON3_SPECIAL_TOKENS.
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
        bpe_cache_size (int): BPE token cache size in Falcon3Tokenizer.
            NOTE: large cache size will speed up tokenization, but the cache object will get really
            large for long running processes (esp. for texts of language that do not use space between
            word, e.g. Chinese); technically not a memory leak but appears as one.
            By default, we set the cache size equals to size of the official Falcon3 tokenizer.

    Example:
        >>> tokenizer = Falcon3Tokenizer(path="/path/to/tokenizer.json", special_tokens=FALCON3_SPECIAL_TOKENS)
        >>> tokenized_text = tokenizer.encode("Hello world!")
        >>> print(tokenized_text)
        [13955, 3139, 2024]
    """

    def __init__(
            self,
            path: str,
            special_tokens: Dict[str, int] = FALCON3_SPECIAL_TOKENS,
            max_seq_len: Optional[int] = None,
            *,
            prompt_template: Optional[PromptTemplate] = None,
            errors: str = "replace",
            unk_token: Optional[str] = None,
            bos_token: Optional[str] = None,
            eos_token: str = ENDOFTEXT,
            pad_token: Optional[str] = ENDOFTEXT,
            bpe_cache_size: int = DEFAULT_FALCON3_TOKENIZER_BPE_CACHE_SIZE,
    ):
        with open(path, encoding="utf-8") as tokenizer_file:
            tok_dict = json.load(tokenizer_file)

        self.encoder = tok_dict['model']['vocab']
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors  # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        bpe_merges = []
        merges_handle = tok_dict['model']['merges']
        for i, line in enumerate(merges_handle):
            # line = line.strip()
            bpe_merges.append(tuple(line))
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))

        self._bpe = lru_cache(maxsize=bpe_cache_size)(self._bpe_without_cache)

        self.pat = re.compile(PRETOKENIZE_REGEX)

        self.special_tokens = special_tokens
        self._special_tokens_reversed = {v: k for k, v in self.special_tokens.items()}

        self.unk_id = None if unk_token is None else self.special_tokens[unk_token]
        self.bos_id = None if bos_token is None else self.special_tokens[bos_token]
        self.eos_id = None if eos_token is None else self.special_tokens[eos_token]
        self.pad_id = None if pad_token is None else self.special_tokens[pad_token]

        # Pattern for special tokens.
        self._pattern_split_special_tokens = re.compile(r"(" + "|".join(special_tokens.keys()) + r")")

        self.max_seq_len = max_seq_len

        self.prompt_template = prompt_template

    def _bpe_without_cache(self, token):
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        return word

    def _tokenize(self, text):
        """Tokenize a string."""
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8")
            )  # Maps all our bytes to unicode strings, avoiding control tokens of the BPE (spaces in our case)
            bpe_tokens.extend(bpe_token for bpe_token in self._bpe(token).split(" "))
        return bpe_tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.encoder.get(token, self.unk_id)

    def encode(
            self, text: str, add_bos: bool = True, add_eos: bool = True
    ) -> List[int]:
        """
        Encode a string into a list of token ids.

        Args:
            text (str): The string to encode.
            add_bos (bool): (Optional) Whether to add the beginning of sequence token.
            add_eos (bool): (Optional) Whether to add the end of sequence token.

        Returns:
            List[int]: The list of token ids.

        Note:
            This method follows
            <https://github.com/huggingface/transformers/blob/v4.41.2/src/transformers/tokenization_utils.py#L541> and
        """

        text = unicodedata.normalize("NFC", text)

        tokens = self._pattern_split_special_tokens.split(text)

        tokenized_text = []
        for token in tokens:
            if not token:
                continue
            if token in self.special_tokens:
                tokenized_text.append(token)
            else:
                tokenized_text.extend(self._tokenize(token))

        # Convert tokenized text to token ids.
        token_ids = []
        if add_bos and self.bos_id is not None:
            token_ids.append(self.bos_id)
        for token in tokenized_text:
            if token in self.special_tokens:
                token_id = self.special_tokens[token]
            else:
                token_id = self._convert_token_to_id(token)
            token_ids.append(token_id)
        if add_eos and self.eos_id is not None:
            token_ids.append(self.eos_id)

        return token_ids

    def _convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) in a token (str) using the vocab."""
        token = self._special_tokens_reversed.get(index, None)
        if token is None:
            return self.decoder.get(index)
        return token

    def _convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Converts a sequence of tokens (string) in a single string."""
        text = "".join(tokens)
        text = bytearray([self.byte_decoder[c] for c in text]).decode(
            "utf-8", errors=self.errors
        )
        return text

    def decode(
            self,
            token_ids: List[int],
            skip_special_tokens: bool = False,
    ) -> str:
        """
        Decode a list of token ids into a string.

        Args:
            token_ids (List[int]): The list of token ids.
            skip_special_tokens (bool): Whether the special tokens should be removed from the decoded string.

        Returns:
            str: The decoded string.
        """
        sub_texts = []
        current_sub_text = []
        for token_id in token_ids:
            token = self._convert_id_to_token(token_id)
            if token_id in self._special_tokens_reversed:
                if current_sub_text:
                    string = self._convert_tokens_to_string(current_sub_text)
                    if string:
                        sub_texts.append(string)
                    current_sub_text = []
                if not skip_special_tokens:
                    sub_texts.append(token)
            else:
                current_sub_text.append(token)
        if current_sub_text:
            sub_texts.append(self._convert_tokens_to_string(current_sub_text))

        text = "".join(sub_texts)
        return text

    def tokenize_messages(
            self,
            messages: List[Message],
            *,
            add_eos: bool = True,
    ) -> Tuple[List[int], List[bool]]:
        """
        Given a list of messages, return a list of tokens for the concatenated
        and formatted messages.

        Args:
            messages (List[Message]): The message list to tokenize.
            add_eos (bool): Wether to add the tokenizer's eos_id at the end of the
                sequence of messages. Default is True.

        Returns:
            Tuple[List[int], List[bool]]: The list of token ids and the list of masks.

        Raises:
            RuntimeError: If a message contains non-text content
        """
        assert not isinstance(self.prompt_template, ChatMLTemplate), (
            "Using ChatMLTemplate with tokenize_messages will result in multiple <|im_*|> tokens wrapping each message."
            "Please use a different template or set to None."
        )
        templated_messages = (
            self.prompt_template(messages)
            if self.prompt_template is not None
            else messages
        )

        tokenized_messages = []
        mask = []
        for index, message in enumerate(templated_messages):
            tokens = []

            # message header
            if message.role != "ipython":
                # tokens.append(self.im_start_id)
                tokens.extend(
                    self.encode(f"{message.role}\n", add_bos=False, add_eos=False)
                )

            # message content
            for item in message.content:
                if item["type"] == "text":
                    tokens.extend(
                        self.encode(
                            item["content"],
                            add_bos=False,
                            add_eos=False,
                        )
                    )
                else:
                    raise RuntimeError(
                        f"Unsupported message content type: {item['type']}"
                    )

            # message footer
            if message.role != "ipython" and (
                    message.role != "assistant" or index != len(messages) - 1
            ):
                # tokens.append(self.im_end_id)
                tokens.extend(self.encode("\n", add_bos=False, add_eos=False))

            tokenized_messages.extend(tokens)
            mask.extend([message.masked] * len(tokens))

            # Break out early if we reach max_seq_len
            if self.max_seq_len and len(tokenized_messages) >= self.max_seq_len:
                break

        # Add the End-Of-Sequence token
        if add_eos:
            tokenized_messages.append(self.eos_id)
            mask.append(mask[-1])

        # Finally, truncate if necessary
        if self.max_seq_len:
            tokenized_messages = truncate(
                tokenized_messages, self.max_seq_len, self.eos_id if add_eos else None
            )
            mask = truncate(mask, self.max_seq_len, True if add_eos else None)

        return tokenized_messages, mask

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
