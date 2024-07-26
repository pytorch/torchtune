# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from functools import partial
from typing import Dict, List, Protocol, Tuple

from torchtune.data import Message, Role


class PromptTemplate(Protocol):
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


class CustomPromptTemplate(PromptTemplate):
    """
    Define a custom prompt template by passing in a dictionary mapping role to
    the prepend and append tags. For example, to achieve the following prompt
    template::

        System: {content}\n
        User: {content}\n
        Assistant: {content}\n
        Tool: {content}\n

    You can define the template as follows::

        template = {
            "system": ("System: ", "\n"),
            "user": ("User: ", "\n"),
            "assistant": ("Assistant: ", "\n"),
            "ipython": ("Tool: ", "\n"),
        }

    Once instantiated, you must call the prompt template on a list of messages. It
    will return the same list of messages updated with the template.

    Args:
        template (Dict[Role, Tuple[str, str]]): a dictionary mapping role to the
            prepend and append tags
    """

    def __init__(
        self,
        template: Dict[Role, Tuple[str, str]],
    ):
        self.template = template

    def __call__(self, messages: List[Message]) -> List[Message]:
        """
        Format each role's message(s) according to the prompt template by prepending
        and appending the defined tags.

        Args:
            messages (List[Message]): list of messages to apply the template to

        Returns:
            List[Message]: The formatted list of messages
        """
        formatted_dialogue = []
        for message in messages:
            prepend_tag = self.template[message.role][0]
            append_tag = self.template[message.role][1]
            content = (
                [{"type": "text", "content": prepend_tag}]
                + message.content
                + [{"type": "text", "content": append_tag}]
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
    CustomPromptTemplate,
    template={
        "user": ("Correct this to standard English: ", "\n---\n"),
        "assistant": ("Corrected: ", ""),
    },
)
GrammarErrorCorrectionTemplate.__doc__ = """
A prompt template for grammar error correction tasks::

    Correct this to standard English: {user_message}
    ---
    Corrected: {assistant_message}

"""
SummarizeTemplate = partial(
    CustomPromptTemplate,
    template={
        "user": ("Summarize this dialogue:\n", "\n---\n"),
        "assistant": ("Summary:\n", ""),
    },
)
SummarizeTemplate.__doc__ = """
A prompt template for summarization tasks::

    Summarize this dialogue:
    {user_message}
    ---
    Summary:
    {assistant_message}

"""
