# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping, Optional


class InstructTemplate(ABC):
    """
    Interface for instruction templates. Each template should include the template
    prompt with placeholders for the data inputs.
    """

    template = ""

    @classmethod
    @abstractmethod
    def format(
        cls, sample: Mapping[str, Any], column_map: Optional[Dict[str, str]] = None
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


class AlpacaInstructTemplate(InstructTemplate):
    """
    Prompt template for Alpaca-style datasets. Template prompt changes slightly depending
    on if there's an instruction + input or just an instruction.


    Template:
    .. code-block:: text
        Below is an instruction that describes a task, paired with an input that provides further context.
        Write a response that appropriately completes the request.


        ### Instruction:

        <YOUR INSTRUCTION HERE>


        ### Input:

        <YOUR INPUT HERE>


        ### Response:



    Template **without** 'input':
    .. code-block:: text
        Below is an instruction that describes a task. Write a response that appropriately completes the request.


        ### Instruction:

        <YOUR INSTRUCITON HERE>


        ### Response:


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

    @classmethod
    def format(
        cls, sample: Mapping[str, Any], column_map: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Generate prompt from instruction and input.

        Args:
            sample (Mapping[str, Any]): a single data sample with instruction
            column_map (Optional[Dict[str, str]]): a mapping from the expected
                placeholder names in the template to the column names in the sample.
                If None, assume these are identical.

        Examples:
            >>> # Simple instruction
            >>> AlpacaInstructTemplate.format(sample={"instruction": "Write a poem"})

            >>> # Instruction with input
            >>> AlpacaInstructTemplate.format(sample={"instruction": "Write a poem", "input": "The poem should be 5 lines long"})

            >>> # Instruction with column map where the 'instruction' key is actually named 'prompt' in the given sample
            >>> AlpacaInstructTemplate.format(sample={"prompt": "Write me a poem"}, column_map={"instruction": "prompt"})

        Returns:
            The formatted prompt
        """
        column_map = column_map or {}
        key_input = column_map.get("input", "input")
        key_instruction = column_map.get("instruction", "instruction")

        if key_input in sample and sample[key_input]:
            prompt = cls.template["prompt_input"].format(
                instruction=sample[key_instruction], input=sample[key_input]
            )
        else:
            prompt = cls.template["prompt_no_input"].format(
                instruction=sample[key_instruction]
            )
        return prompt


class GrammarErrorCorrectionTemplate(InstructTemplate):
    """
    Prompt template for grammar correction datasets.

    Template:
    .. code-block:: text
        Correct this to standard English: <YOUR SENTENCE HERE>
        ---
        Corrected:
    """

    template = "Correct this to standard English: {sentence}\n---\nCorrected: "

    @classmethod
    def format(
        cls, sample: Mapping[str, Any], column_map: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Generate prompt from sentence.

        Args:
            sample (Mapping[str, Any]): a single data sample with sentence
            column_map (Optional[Dict[str, str]]): a mapping from the expected
                placeholder names in the template to the column names in the sample.
                If None, assume these are identical.

        Examples:
            >>> # Simple sentence
            >>> GrammarErrorCorrectionTemplate.format(sample={"sentence": "The quik brown fox jumps the lazy dog"})

            >>> # Sentence with column map where the 'sentence' key is actually named 'input' in the given sample
            >>> GrammarErrorCorrectionTemplate.format(
            ...     sample={"input": "The quik brown fox jumps the lazy dog"},
            ...     column_map={"sentence": "input"}
            ... )

        Returns:
            The formatted prompt
        """
        column_map = column_map or {}
        key_sentence = column_map.get("sentence", "sentence")

        prompt = cls.template.format(sentence=sample[key_sentence])
        return prompt


class SummarizeTemplate(InstructTemplate):
    """
    Prompt template to format datasets for summarization tasks.

    Template:
    .. code-block:: text
        Summarize this dialogue:
        <YOUR DIALOGUE HERE>
        ---
        Summary:

    """

    template = "Summarize this dialogue:\n{dialogue}\n---\nSummary:\n"

    @classmethod
    def format(
        cls, sample: Mapping[str, Any], column_map: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Generate prompt from dialogue.

        Args:
            sample (Mapping[str, Any]): a single data sample with dialog
            column_map (Optional[Dict[str, str]]): a mapping from the expected
                placeholder names in the template to the column names in the sample.
                If None, assume these are identical.

        Examples:
            >>> # Simple dialogue
            >>> SummarizeTemplate.format(sample={"dialogue": "Hello, how are you? Did you know the capital of France is Paris?"})

            >>> # Dialogue with column map where the 'dialogue' key is actually named 'prompt' in the given sample
            >>> SummarizeTemplate.format(
            ...     sample={"prompt": "Hello, how are you? Did you know the capital of France is Paris?"},
            ...     column_map={"dialogue": "prompt"}
            ... )

        Returns:
            The formatted prompt
        """
        column_map = column_map or {}
        key_dialogue = column_map.get("dialogue", "dialogue")

        prompt = cls.template.format(dialogue=sample[key_dialogue])
        return prompt


class StackExchangedPairedTemplate(InstructTemplate):
    """
    Prompt template for preference datasets similar to StackExchangedPaired.

    Template:
    .. code-block:: text
        Question: <YOUR QUESTION HERE>

        Answer:
    """

    template = "Question: {question}\n\nAnswer: "

    @classmethod
    def format(
        cls, sample: Mapping[str, Any], column_map: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Generate prompt from instruction and input.

        Args:
            sample (Mapping[str, Any]): a single data sample with instruction
            column_map (Optional[Dict[str, str]]): a mapping from the expected
                placeholder names in the template to the column names in the sample.
                If None, assume these are identical.

        Examples:
            >>> # Simple question
            >>> StackExchangedPairedTemplate.format(sample={"question": "What is the capital of France?"})

            >>> # Question with column map where the 'question' key is actually named 'prompt' in the given sample
            >>> StackExchangedPairedTemplate.format(
            ...     sample={"prompt": "What is the capital of France?"},
            ...     column_map={"question": "prompt"}
            ... )

        Returns:
            The formatted prompt
        """
        column_map = column_map or {}
        key_prompt = column_map.get("prompt", "prompt")
        prompt = cls.template.format(question=sample[key_prompt])

        return prompt
