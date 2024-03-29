# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping, Optional


class PromptTemplate(ABC):
    """
    Interface for prompt templates. Each template should include the template
    prompt with placeholders for the data inputs.
    """

    template = ""

    @abstractmethod
    def format(
        self, sample: Mapping[str, Any], column_map: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Format the prompt template with the given arguments.

        Args:
            sample (Mapping[str, Any]): a single data sample with various fields
            column_map (Optional[Dict[str, str]]): a mapping from the expected
                placeholder names in the template to the column names in the sample.
                If None, assume these are identical. Note: if the sample output is not named
                as "output" in the dataset, you always need to map it to "output" in column_map.

        Returns:
            The formatted prompt
        """
        pass


class AlpacaInstructTemplate(PromptTemplate):
    """
    Prompt template for the Alpaca dataset. Template prompt changes slightly depending
    on if there's an instruction + input or just an instruction.
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

    def format(
        self, sample: Mapping[str, Any], column_map: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Generate prompt from instruction and input.

        Args:
            sample (Mapping[str, Any]): a single data sample with instruction
            column_map (Optional[Dict[str, str]]): a mapping from the expected
                placeholder names in the template to the column names in the sample.
                If None, assume these are identical.

        Returns:
            The formatted prompt
        """
        column_map = column_map or {}
        key_input = column_map.get("input", "input")
        key_instruction = column_map.get("instruction", "instruction")

        if key_input in sample and sample[key_input]:
            prompt = self.template["prompt_input"].format(
                instruction=sample[key_instruction], input=sample[key_input]
            )
        else:
            prompt = self.template["prompt_no_input"].format(
                instruction=sample[key_instruction]
            )
        return prompt


class GrammarErrorCorrectionTemplate(PromptTemplate):
    """
    Prompt template for the Grammar dataset.
    """

    template = "Correct this to standard English: {sentence}\n---\nCorrected: "

    def format(
        self, sample: Mapping[str, Any], column_map: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Generate prompt from sentence.

        Args:
            sample (Mapping[str, Any]): a single data sample with sentence
            column_map (Optional[Dict[str, str]]): a mapping from the expected
                placeholder names in the template to the column names in the sample.
                If None, assume these are identical.

        Returns:
            The formatted prompt
        """
        column_map = column_map or {}
        key_sentence = column_map.get("sentence", "sentence")

        prompt = self.template.format(sentence=sample[key_sentence])
        return prompt


class SummarizeTemplate(PromptTemplate):
    """
    Prompt template to format datasets for summarization tasks.
    """

    template = "Summarize this dialogue:\n{dialogue}\n---\nSummary:\n"

    def format(
        self, sample: Mapping[str, Any], column_map: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Generate prompt from dialogue.

        Args:
            sample (Mapping[str, Any]): a single data sample with dialog
            column_map (Optional[Dict[str, str]]): a mapping from the expected
                placeholder names in the template to the column names in the sample.
                If None, assume these are identical.

        Returns:
            The formatted prompt
        """
        column_map = column_map or {}
        key_dialogue = column_map.get("dialogue", "dialogue")

        prompt = self.template.format(dialogue=sample[key_dialogue])
        return prompt


class Llama2ChatTemplate(PromptTemplate):
    """
    Prompt template that formats human and system prompts with appropriate tags
    used in LLaMA2 pre-training. Taken from Meta's official LLaMA inference
    repository at https://github.com/meta-llama/llama/blob/main/llama/generation.py.
    The response is tokenized outside of this template.

    Example:
        "[INST] <<SYS>>
        You are a helpful, respectful and honest assistant.
        <</SYS>>

        I am going to Paris, what should I see? [/INST] Paris, the capital of France, is known for its stunning architecture..."
    """

    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    template = {
        "system": f"{B_INST} {B_SYS}{{system}}{E_SYS}{{user}} {E_INST} ",
        "no_system": f"{B_INST} {{user}} {E_INST} ",
    }

    def format(
        self, sample: Mapping[str, Any], column_map: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Generate prompt from a user message and optional system prompt.

        Args:
            sample (Mapping[str, Any]): a single data sample, expects role keys "system" (optional)
                and "user" in the sample.
            column_map (Optional[Dict[str, str]]): a mapping from the expected
                role names in the template to the actual role names in the sample.
                If None, assume these are "system" and "user".

        Returns:
            The formatted prompt
        """
        column_map = column_map or {}
        key_system = column_map.get("system", "system")
        key_user = column_map.get("user", "user")

        if key_system in sample:
            return self.template["system"].format(
                system=sample[key_system], user=sample[key_user]
            )
        else:
            return self.template["no_system"].format(user=sample[key_user])


class MistralChatTemplate(PromptTemplate):
    """
    Prompt template that formats according to Mistral's instruct model:
    https://docs.mistral.ai/models/

    It is identical to `Llama2ChatTemplate`, except it does not support system
    prompts.

    Example:
        "[INST] I am going to Paris, what should I see? [/INST] Paris, the capital
        of France, is known for its stunning architecture..."
    """

    B_INST, E_INST = "[INST]", "[/INST]"
    template = f"{B_INST} {{user}} {E_INST} "

    def format(
        self, sample: Mapping[str, Any], column_map: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Generate prompt from a user message

        Args:
            sample (Mapping[str, Any]): a single data sample, expects only "user" in the sample.
            column_map (Optional[Dict[str, str]]): a mapping from the expected
                role names in the template to the actual role names in the sample.
                If None, assume these are "user".

        Returns:
            The formatted prompt

        Raises:
            ValueError: if the sample contains a "system" key
        """
        if "system" in sample:
            raise ValueError("System prompts are not supported in MistralChatTemplate")

        column_map = column_map or {}
        key_user = column_map.get("user", "user")

        return self.template.format(user=sample[key_user])


class ChatMLTemplate(PromptTemplate):
    """
    OpenAI's Chat Markup Language used by their chat models:
    https://github.com/MicrosoftDocs/azure-docs/blob/main/articles/ai-services/openai/includes/chat-markup-language.md
    It is the default template used by Hugging Face models.

    Example:
        <|im_start|>system
        Provide some context and/or instructions to the model.<|im_end|>
        <|im_start|>user
        The user’s message goes here<|im_end|>
        <|im_start|>assistant
        The assistant’s response goes here<|im_end|>
    """

    IM_START, IM_END = "<|im_start|>", "<|im_end|>"
    template = {
        "system": f"{IM_START}system\n{{system}}{IM_END}\n{IM_START}user\n{{user}}{IM_END}\n{IM_START}assistant\n",
        "no_system": f"{IM_START}user\n{{user}}{IM_END}\n{IM_START}assistant\n",
    }

    def format(
        self, sample: Mapping[str, Any], column_map: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Generate prompt from a user message and optional system prompt.

        Args:
            sample (Mapping[str, Any]): a single data sample, expects role keys "system" (optional)
                and "user" in the sample.
            column_map (Optional[Dict[str, str]]): a mapping from the expected
                role names in the template to the actual role names in the sample.
                If None, assume these are "system" and "user".

        Returns:
            The formatted prompt
        """
        column_map = column_map or {}
        key_system = column_map.get("system", "system")
        key_user = column_map.get("user", "user")

        if key_system in sample:
            return self.template["system"].format(
                system=sample[key_system], user=sample[key_user]
            )
        else:
            return self.template["no_system"].format(user=sample[key_user])
