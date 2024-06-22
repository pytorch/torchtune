# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import Any, List, Mapping, Optional

from torchtune.data._types import Message
from torchtune.modules.transforms import Transform


class Llama2ChatTemplate(Transform):
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

    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    def __init__(
        self,
        system: Optional[str] = None,
        user: Optional[str] = None,
        assistant: Optional[str] = None,
    ):
        self.system = system or f"{self.B_SYS}{{content}}{self.E_SYS}"
        self.user = (
            user or f"{self.B_INST} {{system_message}}{{content}} {self.E_INST} "
        )
        self.assistant = assistant or ""

    def __call__(self, *, messages: List[Message], **kwargs) -> Mapping[str, Any]:
        """
        Format user and system messages with appropriate tags.

        Args:
            sample (List[Message]): a single conversation, structured as a list
                of `Message` objects

        Returns:
            The formatted list of messages
        """
        system_message = ""
        formatted_dialogue = []
        for message in messages:
            content = ""
            if message.role == "system":
                content = self.system.format(content=message.content)
                system_message = content
                # Incorporate the system message in the user message - Llama2 only
                # looks for the <<SYS>> tags and not the explicit role so this will
                # be treated the same as an actual system message. We do this because
                # of the nesting of the system prompt in the user message.
                continue
            elif message.role == "user":
                content = self.user.format(
                    system_message=system_message, content=message.content
                )
            elif message.role == "assistant":
                # No special formatting needed for assistant message
                content = message.content
            formatted_dialogue.append(
                Message(role=message.role, content=content, masked=message.masked),
            )
        return kwargs.update({"messages": formatted_dialogue})


class MistralChatTemplate(Transform):
    """
    Formats according to `Mistral's instruct model <https://docs.mistral.ai/models/>`_.

    It is identical to :class:`~torchtune.data.Llama2ChatTemplate`, except it does not support system
    prompts.

    .. code-block:: text

        "[INST] I am going to Paris, what should I see? [/INST] Paris, the capital
        of France, is known for its stunning architecture..."

    """

    B_INST, E_INST = "[INST]", "[/INST]"

    def __init__(
        self,
        user: Optional[str] = None,
        assistant: Optional[str] = None,
    ):
        self.user = user or f"{self.B_INST} {{content}} {self.E_INST} "
        self.assistant = assistant or ""

    def __call__(self, *, messages: List[Message], **kwargs) -> Mapping[str, Any]:
        """
        Format user messages with appropriate tags.

        Args:
            sample (List[Message]): a single conversation, structured as a list
                of :class:`~torchtune.data.Message` objects

        Returns:
            The formatted list of messages

        Raises:
            ValueError: If system prompts are provided
        """
        formatted_dialogue = []
        for message in messages:
            content = ""
            if message.role == "system":
                raise ValueError(
                    "System prompts are not supported in MistralChatTemplate"
                )
            elif message.role == "user":
                content = self.user.format(
                    content=message.content,
                )
            elif message.role == "assistant":
                # No special formatting needed for assistant message
                content = message.content
            formatted_dialogue.append(
                Message(role=message.role, content=content, masked=message.masked),
            )
        return kwargs.update({"messages": formatted_dialogue})


class ChatMLTemplate(Transform):
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

    IM_START, IM_END = "<|im_start|>", "<|im_end|>"

    def __init__(
        self,
        system: Optional[str] = None,
        user: Optional[str] = None,
        assistant: Optional[str] = None,
    ):
        self.system = system or f"{self.IM_START}system\n{{content}}{self.IM_END}\n"
        self.user = user or f"{self.IM_START}user\n{{content}}{self.IM_END}\n"
        self.assistant = (
            assistant or f"{self.IM_START}assistant\n{{content}}{self.IM_END}"
        )

    def __call__(self, *, messages: List[Message], **kwargs) -> Mapping[str, Any]:
        """
        Format user and system messages with appropriate tags.

        Args:
            sample (List[Message]): a single conversation, structured as a list
                of `Message` objects

        Returns:
            The formatted list of messages
        """
        formatted_dialogue = []
        for message in messages:
            content = ""
            if message.role == "system":
                content = self.system.format(content=message.content)
            elif message.role == "user":
                content = self.user.format(
                    content=message.content,
                )
            elif message.role == "assistant":
                content = self.assistant.format(
                    content=message.content,
                )
            formatted_dialogue.append(
                Message(role=message.role, content=content, masked=message.masked),
            )
        return kwargs.update({"messages": formatted_dialogue})


class AlpacaInstructTemplate(Transform):
    """
    Prompt template for Alpaca-style datasets. Template prompt changes slightly depending
    on if there's an instruction + input or just an instruction. This does not use the base
    PromptTemplate interface because it requires formatting separate columns from the input
    into the template and thus has a different format signature.
    """

    template = {
        "prompt_input": (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
        ),
        "prompt_no_input": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:\n"
        ),
    }

    def __call__(
        self, *, instruction: str, output: str, input: Optional[str] = None, **kwargs
    ) -> Mapping[str, Any]:
        """
        Generate prompt from instruction and input.

        Args:
            sample (Mapping[str, Any]): a single data sample with instruction and optional input
            column_map (Optional[Dict[str, str]]): a mapping from the expected
                placeholder names in the template to the column names in the sample.
                If None, assume these are identical.

        Returns:
            The formatted list of messages
        """

        if input is not None:
            prompt = self.template["prompt_input"].format(
                instruction=instruction,
                input=input,
            )
        else:
            prompt = self.template["prompt_no_input"].format(instruction=instruction)

        messages = [
            Message(role="user", content=prompt),
            Message(role="assistant", content=output),
        ]

        return kwargs.update({"messages": messages})


class QuickTemplate(Transform):
    def __init__(
        self,
        template: str,
    ):
        self.template = template

    def __call__(self, *, input: str, output: str, **kwargs) -> Mapping[str, Any]:
        """
        Generate prompt from sentence that needs grammar correction.

        Args:
            sample (List[Message]): a single conversation, structured as a list
                of `Message` objects

        Returns:
            The formatted list of messages
        """
        messages = []
        messages.append(
            Message(
                role="user",
                content=self.template.format(content=input),
            ),
        )
        messages.append(
            Message(role="assistant", content=output),
        )
        return kwargs.update({"messages": messages})


GrammarErrorCorrectionTemplate = partial(
    QuickTemplate,
    template="Correct this to standard English: {content}\n---\nCorrected: ",
)
SummarizeTemplate = partial(
    QuickTemplate, template="Summarize this dialogue:\n{content}\n---\nSummary:\n"
)
QuestionAnswerTemplate = partial(
    QuickTemplate, template="Question: {content}\n\nAnswer: "
)
