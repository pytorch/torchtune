# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List, Mapping, Protocol


class Transform(Protocol):
    """
    Loose interface for all data and model transforms. Transforms operate at the
    sample level and perform operations on a sample dict which is contained in
    kwargs. Any fields that will be processed are unfolded with explicit keyword-arguments,
    then the updated dict is returned.
    """

    def __call__(self, **kwargs) -> Mapping[str, Any]:
        pass


class Compose(Transform):
    """
    Compose multiple transforms together, inspired by torchvision's ``Compose`` API

    Args:
        transforms (List[Transform]): List of transforms to compose together in sequential order.
    """

    def __init__(self, transforms: List[Transform]) -> None:
        self.transforms = transforms

    def __call__(self, **kwargs) -> Mapping[str, Any]:
        for transform in self.transforms:
            kwargs = transform(**kwargs)
        return kwargs


class TokenizeMessages(Transform):
    """
    Apply the ``tokenize_messages`` method from a given
    :class:`~torchtune.modules.tokenizers.ModelTokenizer` on the ``messages`` field of the sample.

    Args:
        tokenizer (ModelTokenizer): Tokenizer used by the model that implements
            the ``tokenize_messages`` method.
    """

    def __init__(self, tokenizer: ModelTokenizer, max_seq_len: int) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __call__(self, *, messages: List[Message], **kwargs) -> Mapping[str, Any]:
        tokenized_dict = self.tokenizer.tokenize_messages(
            messages, max_seq_len=self.max_seq_len
        )
        kwargs.update(tokenized_dict)
        return kwargs
