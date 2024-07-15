# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from functools import partial
from typing import Dict, List, Protocol, Tuple

from torchtune.data import Message, Role


class Template(Protocol):
    """
    Interface for prompt templates. Each prompt template can include structured
    text for system, user, and assistant roles that are prepended or appended to
    the message content.
    """

    # Template should map role to a tuple containing the tag to prepend to the text
    # and tag to append to the text. Leave as empty strings to not prepend or append
    template: Dict[Role, Tuple[str, str]]

    def __call__(
        self,
        messages: List[Message],
    ) -> List[Message]:
        """
        Format each role's message(s) according to the prompt template

        Args:
            messages (List[Message]): a single conversation, structured as a list
                of :class:`~torchtune.data.Message` objects

        Returns:
            The formatted list of messages
        """
        pass


class QuickTemplate(Template):
    def __init__(
        self,
        template: Dict[Role, Tuple[str, str]],
    ):
        self.template = template

    def __call__(self, messages: List[Message]) -> List[Message]:
        formatted_dialogue = []
        for message in messages:
            content = (
                [{"type": "text", "content": self.template[message.role][0]}]
                + message.content
                + [{"type": "text", "content": self.template[message.role][1]}]
            )
            formatted_dialogue.append(
                Message(
                    role=message.role,
                    content=content,
                    masked=message.masked,
                    ipython=message.ipython,
                    eot=message.eot,
                ),
            )
        return formatted_dialogue


GrammarErrorCorrectionTemplate = partial(
    QuickTemplate,
    template={
        "user": ("Correct this to standard English: ", "\n---\n"),
        "assistant": ("Corrected: ", ""),
    },
)
SummarizeTemplate = partial(
    QuickTemplate,
    template={
        "user": ("Summarize this dialogue:\n", "\n---\n"),
        "assistant": ("Summary:\n", ""),
    },
)
