# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import html
from os import PathLike
from typing import List

import ftfy
import regex as re
import torch

from torchtune.modules.tokenizers._utils import BaseTokenizer


class CLIPTokenizer(BaseTokenizer):
    """
    Text tokenizer for CLIP.

    Based on the official implementation here:
    https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py

    Args:
        merges_path (PathLike): the path to the CLIP merges file
        max_seq_len (int): the context length (all CLIP models use 77)
        truncate (bool): whether to truncate the text when longer than max_seq_len
    """

    def __init__(
        self, merges_path: PathLike, max_seq_len: int = 77, truncate: bool = True
    ):
        self.max_seq_len = max_seq_len
        self.truncate = truncate

        self.byte_encoder = _bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        merges = []
        with open(merges_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if (i == 0 and line.startswith("#version:")) or not line:
                    continue
                merges.append(tuple(line.split()))

        vocab = list(self.byte_encoder.values())
        vocab.extend([v + "</w>" for v in vocab])
        vocab.extend(["".join(merge) for merge in merges])
        vocab.extend(["<|startoftext|>", "<|endoftext|>"])

        self.encoder = {word: i for i, word in enumerate(vocab)}
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = {merge: i for i, merge in enumerate(merges)}

        self.pat = re.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE,
        )

        self.sot_token = self.encoder["<|startoftext|>"]
        self.eot_token = self.encoder["<|endoftext|>"]
        self.pad_token = self.eot_token

        self.cache = {
            "<|startoftext|>": "<|startoftext|>",
            "<|endoftext|>": "<|endoftext|>",
        }

    def __call__(self, texts: List[str]) -> torch.Tensor:
        """
        Returns a Tensor with the tokenized representation of given input strings

        Args:
            texts (List[str]): list of input strings to tokenize

        Returns:
            torch.Tensor: int tensor with shape [len(texts), max_seq_len]
        """
        assert isinstance(texts, list)
        result = torch.full(
            (len(texts), self.max_seq_len), self.pad_token, dtype=torch.int
        )
        for i, text in enumerate(texts):
            tokens = self.encode(text)
            result[i, : len(tokens)] = torch.tensor(tokens)
        return result

    def encode(self, text: str) -> List[int]:
        """
        Given a string, return the encoded list of token ids.

        Args:
            text (str): The text to encode.

        Returns:
            List[int]: The encoded list of token ids.
        """
        text = _clean_text(text).lower()

        tokens = [self.sot_token]
        for token in re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            tokens.extend(
                self.encoder[bpe_token] for bpe_token in self._bpe(token).split(" ")
            )
            if len(tokens) >= self.max_seq_len:
                break
        tokens.append(self.eot_token)

        if len(tokens) > self.max_seq_len:
            assert self.truncate, (
                "Tokenized text is larger than the maximum sequence length but "
                "truncate is set to False."
            )
            tokens = tokens[: self.max_seq_len]
            tokens[-1] = self.eot_token

        return tokens

    def decode(self, tokens: List[int]) -> str:
        """
        Given a list of token ids, return the decoded text, optionally including special tokens.

        Args:
            tokens (List[int]): The list of token ids to decode.

        Returns:
            str: The decoded text.
        """
        text = "".join([self.decoder[token] for token in tokens])
        return (
            bytearray([self.byte_decoder[c] for c in text])
            .decode("utf-8", errors="replace")
            .replace("</w>", " ")
        )

    def _bpe(self, token):
        if token in self.cache:
            return self.cache[token]

        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = _get_pairs(word)

        if not pairs:
            return token + "</w>"

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
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = _get_pairs(word)

        word = " ".join(word)
        self.cache[token] = word
        return word


def _bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
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


def _get_pairs(word):
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


def _clean_text(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text)).strip()
    return re.sub(r"\s+", " ", text).strip()
