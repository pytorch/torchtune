# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping, Optional


class _Llama2ChatFormatConstants:
    """
    Contains constants that are used in Llama2 Chat Format.
    """

    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"


class PromptTemplate(ABC):
    """
    Interface for prompt templates. Each template should include the system
    prompt with placeholders for the data inputs.
    """

    system = ""

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
                If None, assume these are identical.

        Returns:
            The formatted prompt
        """
        pass


class AlpacaInstructTemplate(PromptTemplate):
    """
    Prompt template for the Alpaca dataset. System prompt changes slightly depending
    on if there's an instruction + input or just an instruction.
    """

    system = {
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
        if column_map is not None:
            key_input = column_map["input"]
            key_instruction = column_map["instruction"]
        else:
            key_input = "input"
            key_instruction = "instruction"

        if key_input in sample and sample[key_input]:
            prompt = self.system["prompt_input"].format(
                instruction=sample[key_instruction], input=sample[key_input]
            )
        else:
            prompt = self.system["prompt_no_input"].format(
                instruction=sample[key_instruction]
            )
        return prompt


class GrammarErrorCorrectionTemplate(PromptTemplate):
    """
    Prompt template for the Grammar dataset.

    """

    system = "Correct this to standard English: {sentence}\n---\nCorrected: "

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
        if column_map is not None and "sentence" in column_map:
            key_sentence = column_map["sentence"]
        else:
            key_sentence = "sentence"

        prompt = self.system.format(sentence=sample[key_sentence])
        return prompt


class SummarizeTemplate(PromptTemplate):
    """
    Prompt template for the Summarize dataset.
    """

    system = "Summarize this dialog:\n{dialogue}\n---\nSummary:\n"

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
        if column_map is not None and "dialogue" in column_map:
            key_dialogue = column_map["dialogue"]
        else:
            key_dialogue = "dialogue"

        prompt = self.system.format(dialogue=sample[key_dialogue])
        return prompt


class LLaMA2ChatTemplate(PromptTemplate):
    """
    Prompt template for the Llama2 Chat dataset.
    This template supports only back-and-forth conversation per sample (as it is sufficient for SlimOrca dataset).
    """

    system = {
        "from_system": ("{B_INST} {B_SYS}{system_reply}{E_SYS}{human_reply} {E_INST}"),
        "from_others": ("{B_INST} {human_reply} {E_INST}"),
    }

    def format(
        self, sample: Mapping[str, Any], column_map: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Generate prompt adhering to Llama2 Chat Format.

        Args:
            sample (Mapping[str, Any]): a single data sample with conversations
            column_map (Optional[Dict[str, str]]): a mapping from the expected
                placeholder names in the template to the column names in the sample.
                If None, assume these are identical.

        Returns:
            The formatted prompt
        """
        if column_map is not None:
            key_conversations = column_map["conversations"]
            key_from = column_map["from"] if "from" in column_map else "from"
            key_value = column_map["value"] if "value" in column_map else "value"
        else:
            key_conversations = "conversations"
            key_from = "from"
            key_value = "value"

        agent_text_dict = {}
        for conversation in sample[key_conversations]:
            agent = conversation[key_from]
            text = conversation[key_value]
            agent_text_dict[agent] = text

        if "system" in agent_text_dict:
            prompt = self.system["from_system"].format(
                B_INST=_Llama2ChatFormatConstants.B_INST,
                B_SYS=_Llama2ChatFormatConstants.B_SYS,
                system_reply=agent_text_dict["system"],
                E_SYS=_Llama2ChatFormatConstants.E_SYS,
                human_reply=agent_text_dict["human"],
                E_INST=_Llama2ChatFormatConstants.E_INST,
            )
        else:
            prompt = self.system["from_others"].format(
                B_INST=_Llama2ChatFormatConstants.B_INST,
                human_reply=agent_text_dict["human"],
                E_INST=_Llama2ChatFormatConstants.E_INST,
            )

        return prompt
