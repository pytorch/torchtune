# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import List

from torchtune.data import Message, PromptTemplateInterface


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
