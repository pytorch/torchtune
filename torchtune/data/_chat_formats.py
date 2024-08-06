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


class Llama2ChatFormat(ChatFormat):
    """
    Chat format that formats human and system prompts with appropriate tags
    used in Llama2 pre-training. Taken from Meta's official `Llama inference
    repository <https://github.com/meta-llama/llama/blob/main/llama/generation.py>`_.

    .. code-block:: text

        "[INST] <<SYS>>
        You are a helpful, respectful and honest assistant.
        <</SYS>>"

        I am going to Paris, what should I see? [/INST] Paris, the capital of France, is known for its stunning architecture..."


    """

    template = {
        "system": ("<<SYS>>\n", "\n<</SYS>>\n\n"),
        "user": ("[INST] ", " [/INST] "),
        "assistant": ("", ""),
        "ipython": ("", ""),
    }

    @classmethod
    def format(
        cls,
        sample: List[Message],
    ) -> List[Message]:
        """
        Format user and system messages with appropriate tags.

        Args:
            sample (List[Message]): a single conversation, structured as a list
                of `Message` objects

        Returns:
            The formatted list of messages
        """
        system_message = []
        formatted_dialogue = []
        for message in sample:
            if message.role == "system":
                system_message = (
                    [{"type": "text", "content": cls.template["system"][0]}]
                    + message.content
                    + [{"type": "text", "content": cls.template["system"][1]}]
                )
                # Incorporate the system message in the user message - Llama2 only
                # looks for the <<SYS>> tags and not the explicit role so this will
                # be treated the same as an actual system message. We do this because
                # of the nesting of the system prompt in the user message.
                continue
            elif message.role == "user":
                content = (
                    [{"type": "text", "content": cls.template["user"][0]}]
                    + system_message
                    + message.content
                    + [{"type": "text", "content": cls.template["user"][1]}]
                )
            elif message.role == "assistant":
                # No special formatting needed for assistant message
                content = message.content
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


class MistralChatFormat(ChatFormat):
    """
    Formats according to `Mistral's instruct model <https://docs.mistral.ai/models/>`_.

    It is identical to :class:`Llama2ChatFormat`, except it does not support system
    prompts.

    .. code-block:: text

        "[INST] I am going to Paris, what should I see? [/INST] Paris, the capital
        of France, is known for its stunning architecture..."

    """

    template = {
        "system": None,
        "user": ("[INST] ", " [/INST] "),
        "assistant": ("", ""),
        "ipython": ("", ""),
    }

    @classmethod
    def format(
        cls,
        sample: List[Message],
    ) -> List[Message]:
        """
        Format user and system messages with appropriate tags.

        Args:
            sample (List[Message]): a single conversation, structured as a list
                of `Message` objects

        Returns:
            The formatted list of messages

        Raises:
            ValueError: If system prompts are provided
        """
        formatted_dialogue = []
        for message in sample:
            if message.role == "system":
                raise ValueError(
                    "System prompts are not supported in MistralChatFormat"
                )
            else:
                content = (
                    [{"type": "text", "content": cls.template[message.role][0]}]
                    + message.content
                    + [{"type": "text", "content": cls.template[message.role][1]}]
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


class ChatMLFormat(ChatFormat):
    """
    OpenAI's `Chat Markup Language
    <https://github.com/MicrosoftDocs/azure-docs/blob/772c14eeabfa0c0c561d5c2d34ef19341f528b7b/articles/ai-services/openai/how-to/chat-markup-language.md>`_
    used by their chat models.

    It is the default chat format used by Hugging Face models.

    .. code-block:: text

        <|im_start|>system
        Provide some context and/or instructions to the model.<|im_end|>
        <|im_start|>user
        The user’s message goes here<|im_end|>
        <|im_start|>assistant
        The assistant’s response goes here<|im_end|>

    """

    template = {
        "system": ("<|im_start|>system\n", "<|im_end|>\n"),
        "user": ("<|im_start|>user\n", "<|im_end|>\n"),
        "assistant": ("<|im_start|>assistant\n", "<|im_end|>"),
        "ipython": ("", ""),
    }

    @classmethod
    def format(
        cls,
        sample: List[Message],
    ) -> List[Message]:
        """
        Format user and system messages with appropriate tags.

        Args:
            sample (List[Message]): a single conversation, structured as a list
                of `Message` objects

        Returns:
            The formatted list of messages
        """
        formatted_dialogue = []
        for message in sample:
            content = (
                [{"type": "text", "content": cls.template[message.role][0]}]
                + message.content
                + [{"type": "text", "content": cls.template[message.role][1]}]
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
