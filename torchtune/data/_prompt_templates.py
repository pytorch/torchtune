# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from functools import partial
from typing import Dict, List, Protocol, Tuple

from torchtune.data import Message, Role


class PromptTemplateInterface(Protocol):
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


class PromptTemplate(PromptTemplateInterface):
    """
    Quickly define a custom prompt template by passing in a dictionary mapping role to
    the prepend and append tags. For example, to achieve the following prompt
    template::

        System: {content}\\n
        User: {content}\\n
        Assistant: {content}\\n
        Tool: {content}\\n

    You need to pass in a tuple for each role, where ``PREPEND_TAG`` is the string
    added before the text content and ``APPEND_TAG`` is the string added after::

        template = {role: (PREPEND_TAG, APPEND_TAG)}

    Thus, the template would be defined as follows::

        template = {
            "system": ("System: ", "\\n"),
            "user": ("User: ", "\\n"),
            "assistant": ("Assistant: ", "\\n"),
            "ipython": ("Tool: ", "\\n"),
        }

    Once instantiated, you must call the prompt template on a list of messages. It
    will return the same list of messages updated with the template.

    Note:
        Any tags prepended/appended to the assistant message will be included
        in the loss calculation. All other prepend/append tags for other roles
        (system, user, ipython) are, in most cases, not included in loss. Consider using
        the append tags for user messages for tags that need to come before the
        assistant message but should not be included in loss. For more custom masking
        and prompt templating, you can create your own class based off the
        :class:`~torchtune.data.PromptTemplate` interface.

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
            if message.role in self.template:
                prepend_tag = self.template[message.role][0]
                append_tag = self.template[message.role][1]
                content = (
                    [{"type": "text", "content": prepend_tag}]
                    + message.content
                    + [{"type": "text", "content": append_tag}]
                )
            else:
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

class Llama2ChatTemplate(PromptTemplateInterface):
    """
    Prompt template that formats chat data of human and system prompts with appropriate tags
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

    def __call__(
        self,
        messages: List[Message],
    ) -> List[Message]:
        """
        Format user and system messages with appropriate tags.

        Args:
            messages (List[Message]): a single conversation, structured as a list
                of `Message` objects

        Returns:
            The formatted list of messages
        """
        system_message = []
        formatted_dialogue = []
        for message in messages:
            if message.role == "system":
                system_message = (
                    [{"type": "text", "content": self.template["system"][0]}]
                    + message.content
                    + [{"type": "text", "content": self.template["system"][1]}]
                )
                # Incorporate the system message in the user message - Llama2 only
                # looks for the <<SYS>> tags and not the explicit role so this will
                # be treated the same as an actual system message. We do this because
                # of the nesting of the system prompt in the user message.
                continue
            elif message.role == "user":
                content = (
                    [{"type": "text", "content": self.template["user"][0]}]
                    + system_message
                    + message.content
                    + [{"type": "text", "content": self.template["user"][1]}]
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


class MistralChatTemplate(PromptTemplateInterface):
    """
    Formats according to `Mistral's instruct model
    <https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1#instruction-format>`_.

    It is identical to :class:`~torchtune.data.Llama2ChatTemplate`, except it does not support system
    prompts.

    Note:
        This template is only recommended for Mistral's Instruct-v0.1 and Instruct-v0.2 models.
        Instruct-v0.3 adds additional tags for tool calls, which is not yet supported by this
        template.

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

    def __call__(
        self,
        messages: List[Message],
    ) -> List[Message]:
        """
        Format user and system messages with appropriate tags.

        Args:
            messages (List[Message]): a single conversation, structured as a list
                of `Message` objects

        Returns:
            The formatted list of messages

        Raises:
            ValueError: If system prompts are provided
        """
        formatted_dialogue = []
        for message in messages:
            if message.role == "system":
                raise ValueError(
                    "System prompts are not supported in MistralChatTemplate"
                )
            else:
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


class ChatMLTemplate(PromptTemplateInterface):
    """
    OpenAI's `Chat Markup Language
    <https://github.com/MicrosoftDocs/azure-docs/blob/772c14eeabfa0c0c561d5c2d34ef19341f528b7b/articles/ai-services/openai/how-to/chat-markup-language.md>`_
    used by their chat models.

    It is the default chat template used by Hugging Face models.

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

    def __call__(
        self,
        messages: List[Message],
    ) -> List[Message]:
        """
        Format user, assistant, and system messages with appropriate tags.

        Args:
            messages (List[Message]): a single conversation, structured as a list
                of `Message` objects

        Returns:
            The formatted list of messages
        """
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
    PromptTemplate,
    template={
        "user": ("Correct this to standard English: ", "\n---\nCorrected: "),
    },
)
GrammarErrorCorrectionTemplate.__doc__ = """
A prompt template for grammar error correction tasks::

    Correct this to standard English: {user_message}
    ---
    Corrected: {assistant_message}

Please see :class:`~torchtune.data.PromptTemplate` for full API arguments.
"""
SummarizeTemplate = partial(
    PromptTemplate,
    template={
        "user": ("Summarize this dialogue:\n", "\n---\nSummary:\n"),
    },
)
SummarizeTemplate.__doc__ = """
A prompt template for summarization tasks::

    Summarize this dialogue:
    {user_message}
    ---
    Summary:
    {assistant_message}

Please see :class:`~torchtune.data.PromptTemplate` for full API arguments.
"""
