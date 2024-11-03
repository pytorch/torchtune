# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from functools import partial
from typing import Dict, List, Protocol, Tuple, Union

from torchtune.config._utils import _get_component_from_path

from torchtune.data._messages import Message, Role

_TemplateType = Union[str, Dict[Role, Tuple[str, str]]]


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
        inference: bool = False,
    ) -> List[Message]:
        """
        Format each role's message(s) according to the prompt template

        Args:
            messages (List[Message]): a single conversation, structured as a list
                of :class:`~torchtune.data.Message` objects
            inference (bool): Whether the template is being used for inference or not.

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

    def __call__(
        self, messages: List[Message], inference: bool = False
    ) -> List[Message]:
        """
        Format each role's message(s) according to the prompt template by prepending
        and appending the defined tags.

        Args:
            messages (List[Message]): list of messages to apply the template to
            inference (bool): Whether the template is being used for inference or not.

        Returns:
            List[Message]: The formatted list of messages
        """
        formatted_dialogue = []
        for message in messages:
            content = message.content
            if message.role in self.template:
                prepend_tag = self.template[message.role][0]
                append_tag = self.template[message.role][1]
                content = message.content

                if isinstance(prepend_tag, str) and len(prepend_tag) > 0:
                    content = [{"type": "text", "content": prepend_tag}] + content

                if isinstance(append_tag, str) and len(append_tag) > 0:
                    content = content + [{"type": "text", "content": append_tag}]
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
        "assistant": ("<|im_start|>assistant\n", "<|im_end|>\n"),
        "ipython": ("", ""),
    }

    def __call__(
        self,
        messages: List[Message],
        inference: bool = False,
    ) -> List[Message]:
        """
        Format user, assistant, and system messages with appropriate tags.

        Args:
            messages (List[Message]): a single conversation, structured as a list
                of `Message` objects
            inference (bool): Whether the template is being used for inference or not.

        Returns:
            The formatted list of messages
        """
        formatted_dialogue = []
        for index, message in enumerate(messages):
            prepend_tag = self.template[message.role][0]
            append_tag = self.template[message.role][1]
            # If empty assistant message at the end, we are expecting the model
            # to generate the response continuing from the assistant prepend tag,
            # so do not add the append tag.
            if (
                message.role == "assistant"
                and index == len(messages) - 1
                and len(message.text_content) == 0
            ):
                content = message.content
                if isinstance(prepend_tag, str) and len(prepend_tag) > 0:
                    content = [
                        {"type": "text", "content": prepend_tag}
                    ] + message.content
            else:
                content = message.content

                if isinstance(prepend_tag, str) and len(prepend_tag) > 0:
                    content = [{"type": "text", "content": prepend_tag}] + content

                if isinstance(append_tag, str) and len(append_tag) > 0:
                    content = content + [{"type": "text", "content": append_tag}]

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
QuestionAnswerTemplate = partial(
    PromptTemplate,
    template={
        "user": ("Question: ", "\n\nAnswer: "),
    },
)
QuestionAnswerTemplate.__doc__ = """
A prompt template for question answering tasks::

    Question: {user_message}

    Answer: {assistant_message}

Please see :class:`~torchtune.data.PromptTemplate` for full API arguments.
"""


def _get_prompt_template(
    prompt_template: _TemplateType,
) -> PromptTemplateInterface:
    """
    Retrieve prompt template from import dotpath or create a custom one with provided
    template dictionary.

    Args:
        prompt_template (_TemplateType): optional specified prompt template.
            If a string, it is assumed to be the dotpath of a :class:`~torchtune.data.PromptTemplateInterface`
            class. If a dictionary, it is assumed to be a custom prompt template mapping role to the
            prepend/append tags.

    Returns:
        PromptTemplateInterface: the specified prompt template

    Raises:
        ValueError: If a string or dictionary is not passed in
    """
    if isinstance(prompt_template, str):
        return _get_component_from_path(prompt_template)()
    elif isinstance(prompt_template, dict):
        return PromptTemplate(prompt_template)
    else:
        raise ValueError(
            f"Prompt template must be a dotpath string or dictionary with custom template, got {type(prompt_template)}"
        )
