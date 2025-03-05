# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Any, Dict, List, Mapping, Set, Tuple

import regex as re

from torchtune.modules.transforms.tokenizers._utils import BaseTokenizer

WORD_BOUNDARY = "</w>"


class CLIPTokenizer(BaseTokenizer):
    """
    Text tokenizer for CLIP.

    Based on the official implementation here:
    https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py

    Args:
        path (str): the path to the CLIP merges file
        max_seq_len (int): the context length (all CLIP models use 77)
        truncate (bool): whether to truncate the text when longer than max_seq_len
    """

    def __init__(self, path: str, max_seq_len: int = 77, truncate: bool = True):
        self.max_seq_len = max_seq_len
        self.truncate = truncate

        self.byte_encoder = _bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        merges = _load_merges(path)

        vocab = list(self.byte_encoder.values())
        vocab.extend([v + WORD_BOUNDARY for v in vocab])
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
            .replace(WORD_BOUNDARY, " ")
        )

    def __call__(
        self, sample: Mapping[str, Any], inference: bool = False
    ) -> Mapping[str, Any]:
        """
        Tokenize the "text" field in the sample.

        Args:
            sample (Mapping[str, Any]): A sample with a "text" field containing a string to tokenize
            inference (bool): Unused by this tokenizer

        Returns:
            Mapping[str, Any]: The sample with added "tokens" field and the "messages" field removed.
        """
        text = sample.pop("text")
        sample["tokens"] = self.encode(text)
        return sample

    def _bpe(self, token: str) -> str:
        """
        Performs byte-pair encoding on a single token.
        """
        if token in self.cache:
            return self.cache[token]

        if len(token) < 2:
            return token + WORD_BOUNDARY

        # create the initial "word" (seq of "symbols" i.e. characters and merged subwords)
        # by converting the token to tuple of characters and add </w> to the last character
        word = tuple(token[:-1]) + (token[-1] + WORD_BOUNDARY,)

        # get all pairs of adjacent characters
        pairs = _get_pairs(word)

        # merge symbol pairs until there are no possible merges left
        while True:
            # find the pair with the lowest rank (highest priority to merge)
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            # end if there are no pairs to merge
            if bigram not in self.bpe_ranks:
                break

            # create the next "word" by merging any adjacent symbols that match the bigram
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                # find next potentially mergeable position and copy over any skipped characters
                # if no more merge positions found, copy remaining characters and finish
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break

                # check if we can perform a merge
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)

            # end if the new "word" is fully merged
            if len(word) == 1:
                break

            # get all pairs of adjacent symbols in the new word
            pairs = _get_pairs(word)

        word = " ".join(word)
        self.cache[token] = word
        return word


def _bytes_to_unicode() -> Dict[int, str]:
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


def _get_pairs(word: Tuple[str, ...]) -> Set[Tuple[str, str]]:
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


def _clean_text(text: str) -> str:
    """
    Minimal version of CLIP's text cleaning via the `ftfy` package.
    """
    return text.replace("’", "'")


def _load_merges(path: str) -> List[Tuple[str, str]]:
    merges = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if (i == 0 and line.startswith("#version:")) or not line:
                continue
            merges.append(tuple(line.split()))
    return merges
