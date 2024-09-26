# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from torchtune.data._messages import Message, Role


class ChatFormat(ABC):
    """
    Warning:
        This class is deprecated and will be removed in a future release. Please use
        :class:`~torchtune.data.PromptTemplate` for custom chat formats.

    Interface for chat formats. Each chat format should include tags for system,
    user, and assistant roles that are prepended or appended to the message
    content.
    """

    # Template should map role to a tuple containing the tag to prepend to the text
    # and tag to append to the text. Leave as empty strings to not prepend or append
    template: Dict[Role, Tuple[str, str]]

    @classmethod
    @abstractmethod
    def format(
        cls,
        sample: List[Message],
    ) -> List[Message]:
        """
        Format each role's message(s) according to the chat format

        Args:
            sample (List[Message]): a single conversation, structured as a list
                of `Message` objects

        Returns:
            The formatted list of messages
        """
        pass
