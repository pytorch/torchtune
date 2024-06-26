# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Literal, Union

from PIL.Image import Image

Role = Literal[
    "system",  # Origin is system prompt
    "user",  # Origin is user
    "assistant",  # Origin is the model output
    "ipython",  # Origin is return from a tool call
]


@dataclass
class Message:
    """
    This dataclass represents individual messages in an instruction or chat dataset.

    Note that the fields ipython and eot are only relevant when tokenizing with tiktoken,
    as they inform handling of special tokens in that case.

    Attributes:
        role (Role): role of the message writer. Can be "system", "user", "assistant", or "ipython".
        content (Union[str, Image]): content of the message. Can be a string or a PIL Image.
        masked (bool): whether the message is masked in the sample. Default: False
        ipython (bool): whether the message is an ipython call. Default: False
        eot (bool): whether the message corresponds to the end of a turn. Should be true
            except in the case of multiple consecutive assistant messages (i.e., tool calls
            by assistant). Default: True
    """

    role: Role
    content: Union[str, Image]
    masked: bool = False
    ipython: bool = False
    eot: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "Message":
        """
        Construct a Message from a dictionary.

        Args:
            d (dict): dictionary containing the fields of the Message.

        Returns:
            Message: constructed Message.
        """
        return cls(
            role=d["role"],
            content=d["content"],
            masked=d.get("masked", False),
            ipython=d.get("ipython", False),
            eot=d.get("eot", True),
        )
