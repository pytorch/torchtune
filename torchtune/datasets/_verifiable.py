from typing import Any, Callable, Dict, List, Mapping, Optional

import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset

from torchtune.data import CROSS_ENTROPY_IGNORE_IDX
from torchtune.modules.transforms import Transform
from torchtune.modules.transforms.tokenizers import ModelTokenizer
from torchtune.data import Message


class PromptToMessage(Transform):
    """
    Message transform class that converts a single sample with "prompt" and other fields,
    (or equivalent fields specified in column_map) to a user message.

        |  prompt         |  output          |
        |-----------------|------------------|
        | "user prompt"   | "other data" |

    Args:
        train_on_input (bool): Whether the model is trained on the user prompt or not.
            Default is False.
        column_map (Optional[Dict[str, str]]): a mapping to change the expected "prompt"
            column names to the actual column names in the dataset. Keys should
            be "prompt" and any other columns you want to preserve. Default is None,
            keeping the default "prompt" column.
        new_system_prompt (Optional[str]): if specified, prepend a system message. This can
            serve as instructions to guide the model response. Default is None.

    Raises:
        ValueError:
            If ``column_map`` is provided and "prompt" not in ``column_map``.
    """

    def __init__(
        self,
        train_on_input: bool = False,
        column_map: Optional[Dict[str, str]] = None,
        new_system_prompt: Optional[str] = None
    ):
        self.train_on_input = train_on_input
        self.new_system_prompt = new_system_prompt

        self.column_map = column_map

        if self.column_map is not None:
            if "prompt" not in self.column_map:
                raise ValueError(
                    f"Expected a key of 'prompt' in column_map but found {self.column_map.keys()}."
                )
        else:
            self.column_map = {"prompt": "prompt"}

    def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        # Build the user content from the prompt.
        content = [{"type": "text", "content": sample[self.column_map["prompt"]]}]

        # Construct messages array.
        messages = [
            Message(
                role="user",
                content=content,
                masked=not self.train_on_input,
                eot=True,
            ),
        ]
        if self.new_system_prompt is not None:
            # Prepend system prompt if specified.
            messages = [
                Message(
                    role="system", content=self.new_system_prompt, masked=True, eot=True
                )
            ] + messages

        # Gather extra columns (besides "prompt") from column_map.
        extra_columns = {}
        for key, col_name in self.column_map.items():
            if key != "prompt":
                extra_columns[key] = sample[col_name]

        # Return messages plus any extra columns.
        return {
            "messages": messages,
            **extra_columns,
        }


class VerifiableDataset(Dataset):
    """
    Class for fine-tuning with verifiable rewards. This class supplies a prompt,
    plus additional columns to be passed through to the verifier.

    1. Dataset-specific transform.
    2. Tokenization with optional prompt template if configured.

    All datasets are formatted into a list of :class:`~torchtune.data.Message`

    Args:
        source (str): path to dataset repository on Hugging Face or local path.
        message_transform (Transform): callable that transforms each sample into a dict with "messages"
            and any extra columns that should be preserved.
        tokenizer (ModelTokenizer): Tokenizer used by the model.
        filter_fn (Optional[Callable]): optional callable used to filter the dataset prior to pre-processing.
        packed (bool): Whether or not to pack the dataset to ``max_seq_len`` prior to training.
            Not supported in this class.
        **load_dataset_kwargs (Dict[str, Any]): additional keyword arguments to pass to ``load_dataset``.
    """

    def __init__(
        self,
        *,
        source: str,
        message_transform: Transform,
        tokenizer: ModelTokenizer,
        filter_fn: Optional[Callable] = None,
        packed: bool = False,
        split: str = "train",
        **load_dataset_kwargs: Dict[str, Any],
    ) -> None:
        if packed:
            raise ValueError(
                "Packed is currently not supported for verifiable datasets."
            )

        self._tokenizer = tokenizer
        self._message_transform = message_transform
        self._data = load_dataset(source, split=split, **load_dataset_kwargs)

        if filter_fn is not None:
            self._data = self._data.filter(filter_fn)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self._data[index]
        return self._prepare_sample(sample)

    def _prepare_sample(self, sample: Mapping[str, Any]) -> Dict[str, Any]:
        # Transform the sample to get messages plus any extra columns.
        transformed_sample = self._message_transform(sample)
        messages = transformed_sample.pop("messages")

        # Tokenize messages.
        input_ids, masks = self._tokenizer.tokenize_messages(messages)
        labels = list(
            np.where(masks, CROSS_ENTROPY_IGNORE_IDX, input_ids)
        )

        # Return tokens, labels, plus all extra columns from transform.
        tokenized_dict = {
            "tokens": input_ids,
            "labels": labels,
            **transformed_sample,
        }
        return tokenized_dict


def verifiable_dataset(
    tokenizer: ModelTokenizer,
    *,
    source: str,
    column_map: Optional[Dict[str, str]] = None,
    train_on_input: bool = False,
    new_system_prompt: Optional[str] = None,
    filter_fn: Optional[Callable] = None,
    split: str = "train",
    **load_dataset_kwargs: Dict[str, Any],
) -> VerifiableDataset:
    """
    Configures a custom dataset for use with verifiable rewards.

    This function requires the dataset to have a "prompt" column. Additional columns specified
    in ``column_map`` will be preserved and returned in each sample.

    Args:
        tokenizer (ModelTokenizer): Tokenizer used by the model.
        source (str): Path to dataset repository on Hugging Face or local path.
        column_map (Optional[Dict[str, str]]): Mapping of required/extra columns to dataset columns.
            Must include "prompt" if that column is different from "prompt" in your dataset.
        train_on_input (bool): If True, includes the prompt tokens in the training labels.
            If False, the prompt tokens are masked.
        new_system_prompt (Optional[str]): If set, prepends a system message to each sample.
        filter_fn (Optional[Callable]): Optional filter function.
        split (str): Which split to use. Defaults to "train".
        **load_dataset_kwargs (Dict[str, Any]): Additional arguments for load_dataset.

    Returns:
        VerifiableDataset: Configured dataset that yields dicts with tokens, labels, and any extra columns.
    """
    message_transform = PromptToMessage(
        train_on_input=train_on_input,
        column_map=column_map,
        new_system_prompt=new_system_prompt,
    )

    return VerifiableDataset(
        source=source,
        message_transform=message_transform,
        tokenizer=tokenizer,
        filter_fn=filter_fn,
        split=split,
        **load_dataset_kwargs,
    )
