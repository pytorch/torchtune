# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import List

from torchtune.data import Message, PromptTemplateInterface


class MistralSmallTemplate(PromptTemplateInterface):
    """
    Formats according to Mistral Small's chat template format.
    This template supports both system and user messages, with special tags for each.

    .. code-block:: text

        [INST] [SYSTEM_PROMPT]You are Mistral Small 3, a Large Language Model (LLM) created by Mistral AI[/SYSTEM_PROMPT]
        Hello, how are you?[/INST]
        I'm doing well, thank you for asking!

    """

    template = {
        "system": ("[SYSTEM_PROMPT]", "[/SYSTEM_PROMPT]\n"),
        "user": ("[INST] ", "[/INST]"),
        "assistant": ("", ""),
        "ipython": ("", ""),
    }

    def __call__(
        self,
        messages: List[Message],
    ) -> List[Message]:
        """
        Format messages with appropriate tags based on their role.

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
                continue
            elif message.role == "user":
                content = (
                    [{"type": "text", "content": self.template["user"][0]}]
                    + system_message
                    + message.content
                    + [{"type": "text", "content": self.template["user"][1]}]
                )
            elif message.role == "assistant":
                content = message.content + [{"type": "text", "content": "\n"}]
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
