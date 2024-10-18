# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import List, Optional

from torchtune.data import Message, PromptTemplateInterface


DEFAULT_SYS_PROMPT = (
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
)
TOOL_INSTRUCTION_START = "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>"  # noqa: B950
TOOL_INSTRUCTION_END = '</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'  # noqa: B950


class Qwen25ChatTemplate(PromptTemplateInterface):
    """Qwen2.5's chat template.

    Defined in the Jinja template in
    https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/raw/main/tokenizer_config.json

    Attributes:
        template: a dictionary mapping role to the
            prepend and append tags

    Args:
        add_default_sys_prompt_if_missing (bool): Add Qwen's default system prompt if the messages do not have one.
            Default: False
        tools (Optional[List[str]]): Optional JSON schemas for tools that the LLM can access.
            If provided, the tool schemas will be added to the system prompt according to Qwen's tool instruction format.
            See https://huggingface.co/blog/unified-tool-use for more information about the JSON tool schema.
            Default: None
    """

    template = {
        "system": ("<|im_start|>system\n", "<|im_end|>\n"),
        "user": ("<|im_start|>user\n", "<|im_end|>\n"),
        "assistant": ("<|im_start|>assistant\n", "<|im_end|>\n"),
    }

    def __init__(
        self,
        add_default_sys_prompt_if_missing: bool = False,
        tools: Optional[List[str]] = None,
    ):
        self._add_default_sys_prompt_if_missing = add_default_sys_prompt_if_missing
        self._tools = tools

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
        # Add a system prompt if missing
        if self._add_default_sys_prompt_if_missing and (
            not messages or messages[0].role != "system"
        ):
            messages.insert(
                0, Message(role="system", content=DEFAULT_SYS_PROMPT, masked=True)
            )

        # Add tool instructions to the system prompt
        if self._tools and messages[0].role == "system":
            tool_instruction = [TOOL_INSTRUCTION_START]
            for tool in self._tools:
                tool_instruction.append(tool)
            tool_instruction.append(TOOL_INSTRUCTION_END)
            tool_instruction = "\n".join(tool_instruction)
            assert messages[0].role == "system"
            messages[0].content.append({"type": "text", "content": tool_instruction})

        # Add start/end tags to messages
        formatted_dialogue = []
        for i, message in enumerate(messages):
            if message.role == "ipython":
                if i == 0 or messages[i - 1].role != "ipython":
                    prepend_tag = f"{self.template['user'][0]}<tool_response>\n"
                else:
                    prepend_tag = "\n<tool_response>\n"

                if i == len(messages) - 1 or messages[i + 1]["role"] != "ipython":
                    append_tag = f"\n</tool_response>{self.template['user'][1]}"
                else:
                    append_tag = "\n</tool_response>"
            else:
                prepend_tag, append_tag = self.template[message.role]

            content = [{"type": "text", "content": prepend_tag}] + message.content
            # If empty assistant message at the end, we are expecting the model
            # to generate the response continuing from the assistant prepend tag,
            # so do not add the append tag.
            if message.role != "assistant" or i != len(messages) - 1:
                content += [{"type": "text", "content": append_tag}]

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
