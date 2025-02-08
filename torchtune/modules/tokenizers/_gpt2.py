# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Iterator, List

import json
import re
from torchtune.modules.tokenizers._utils import BaseTokenizer
from functools import lru_cache

# Constants controlling encode logic
MAX_ENCODE_CHARS = 400_000
MAX_NO_WHITESPACE_CHARS = 25_000

@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
    characters the bpe code barfs on.

    This is standard implementation based on: https://github.com/huggingface/transformers/blob/6b550462139655d488d4c663086a63e98713c6b9/src/transformers/models/gpt2/tokenization_gpt2.py#L36
    """
    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
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


class GPT2BaseTokenizer(BaseTokenizer):
    """
    A lightweight version of the base GPT2Tokenizer.

    Args:
        path (str): Path to pretrained tokenizer checkpoint file.
        name (str): Name of the tokenizer (used by tiktoken for identification).
        pattern (str): Regex pattern used to split input text into chunks before passing
            to byte-pair encoding.
        bos_id (int): beginning-of-sequence token id. This can be present or absent in ``special_tokens``.
        eos_id (int): end-of-sequence token id. This can be present or absent in ``special_tokens``.
        special_tokens (Dict[str, int]): Mapping of special tokens to their ids.

    Examples:
        >>> tokenizer = TikTokenBaseTokenizer("/path/to/tt_model")
        >>> tokenized_text = tokenizer.encode("Hello world!", add_bos=True, add_eos=True)
        >>> print(tokenized_text)
        [1, 31587, 29644, 102, 2]
    """

    def __init__(
        self,
        vocab_path: str,
        merges_path: str,
        errors: str,
        unk_id: int,
        bos_id: int,
        eos_id: int,
        pad_id: int,
    ):
        with open(vocab_path, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors  # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        with open(merges_path, encoding="utf-8") as merges_handle:
            bpe_merges = merges_handle.read().split("\n")[1:-1]
        bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}

        # Should have added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        # We do not use external regex library, so this is slightly modified version of the original line
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?[^\s\W\d_]+|\s+(?!\S)|\s+""")

        self.unk_id = unk_id 
        self.bos_id = bos_id 
        self.eos_id = eos_id 
        self.pad_id = pad_id

    @property
    def vocab_size(self):
        return len(self.encoder)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
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
        self.cache[token] = word
        return word

    def _tokenize(
        self,
        text: str,
    ) -> List[int]:
        """Tokenize a string."""
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8")
            )  # Maps all our bytes to unicode strings, avoiding control tokens of the BPE (spaces in our case)
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
       
        return bpe_tokens
    
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.encoder.get(token, self.encoder.get(self.unk_id))

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
    ):
        bpe_tokens = list(map(self._convert_token_to_id, self._tokenize(text)))

        if add_bos:
            bpe_tokens = [self.bos_id] + bpe_tokens
        if add_eos:
            bpe_tokens = bpe_tokens + [self.eos_id]
        return bpe_tokens
