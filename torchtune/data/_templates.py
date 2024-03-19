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
                If None, assume these are identical.

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
        if column_map is not None:
            key_input = column_map["input"]
            key_instruction = column_map["instruction"]
        else:
            key_input = "input"
            key_instruction = "instruction"

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
        if column_map is not None and "sentence" in column_map:
            key_sentence = column_map["sentence"]
        else:
            key_sentence = "sentence"

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
        if column_map is not None and "dialogue" in column_map:
            key_dialogue = column_map["dialogue"]
        else:
            key_dialogue = "dialogue"

        prompt = self.template.format(dialogue=sample[key_dialogue])
        return prompt
