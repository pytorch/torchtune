# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import List

from torchtune.data import Message, PromptTemplateInterface


class MistralChatTemplate(PromptTemplateInterface):
    """
    Formats according to Mistral's `instruct model
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
