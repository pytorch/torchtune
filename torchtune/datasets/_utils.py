# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from typing import List, Tuple

from torchtune.datasets._common import CROSS_ENTROPY_IGNORE_IDX
from torchtune.modules import Tokenizer


def tokenize_prompt_and_response(
    tokenizer: Tokenizer,
    prompt: str,
    response: str,
    train_on_input: bool = False,
) -> Tuple[List[int], List[int]]:
    """
    Tokenize a prompt and response pair.

    Args:
        tokenizer (Tokenizer): The tokenizer to use.
        prompt (str): The prompt to tokenize.
        response (str): The response to tokenize.
        train_on_input (bool): Whether to train on prompt or mask it out. Default
            is False.

    Returns:
        The tokenized prompt and response.
    """
    prompt_with_response = prompt + response
    encoded_prompt = tokenizer.encode(text=prompt, add_bos=True, add_eos=False)
    encoded_prompt_with_response = tokenizer.encode(
        text=prompt_with_response, add_bos=True, add_eos=True
    )
    labels = copy.deepcopy(encoded_prompt_with_response)

    if not train_on_input:
        labels[: len(encoded_prompt)] = [CROSS_ENTROPY_IGNORE_IDX] * len(encoded_prompt)

    assert len(encoded_prompt_with_response) == len(labels)

    return encoded_prompt_with_response, labels


def truncate_if_necessary(
    tokenizer: Tokenizer,
    prompt_tokens: List[int],
    label_tokens: List[int],
    max_seq_len: int,
) -> Tuple[List[int], List[int]]:
    """
    Truncate a prompt and response pair if longer than max sequence length.

    Args:
        tokenizer (Tokenizer): The tokenizer to use.
        prompt_tokens (List[int]): The prompt tokens.
        label_tokens (List[int]): The label tokens.
        max_seq_len (int): The maximum sequence length.

    Returns:
        The truncated prompt and response.
    """
    # Truncate to max token length - 2 (so that there is at least one label token)
    prompt_tokens_truncated = prompt_tokens[: max_seq_len - 2]

    # Calculate space left for label tokens
    label_tokens_length = max_seq_len - len(prompt_tokens_truncated)

    # Truncate label tokens
    label_tokens_truncated = label_tokens[: label_tokens_length - 1]
    if label_tokens_truncated[-1] != tokenizer.eos_id:
        label_tokens_truncated.append(tokenizer.eos_id)

    return prompt_tokens_truncated, label_tokens_truncated


def _get_template(template: str) -> PromptTemplate:
    """
    Get the prompt template class from the template string.

    String should either be the PromptTemplate class name directly, or a raw
    string with 1 or more placeholders. If none of these apply, then raise an
    error.

    Args:
        template (str): class name of template, or string with placeholders

    Returns:
        PromptTemplate: the prompt template class or the same verified string

    Raises:
        ValueError: if the template is not a PromptTemplate class or a proper
            template string
    """
    path = "torchtune.data." + template
    try:
        template_class = _get_component_from_path(path)
        return template_class()
    except InstantiationError:
        # Verify that string can be used as a template, should have variable
        # placeholders
        pattern = r"\{.+?\}"
        if not re.search(pattern, template):
            raise ValueError(
                f"Invalid template '{template}': "
                + "Must be a PromptTemplate class or a string with placeholders."
            ) from None
        return template
