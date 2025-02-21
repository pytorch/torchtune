from typing import Any, Callable, Dict, List, Mapping, Optional
import logging
from torchtune.data import Message
from torchtune.datasets._preference import PreferenceDataset, Trajectory_DPO_Dataset, Trajectory_CE_Dataset
from torchtune.modules.tokenizers import ModelTokenizer
from torchtune.modules.transforms import Transform
import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset

from torchtune.data import CROSS_ENTROPY_IGNORE_IDX

from torchtune.modules.tokenizers import ModelTokenizer
from torchtune.modules.transforms import Transform
import random, json
import os

class PrimeMessages(Transform):
    """
    Data loader class for preparing data for cross-entropy loss. Processes only chosen
    (positive) trajectories as rejected trajectories are not required.
    
    Args:
        train_on_input (bool): If True, includes input in training data. Default is False.
        column_map (Optional[Dict[str, str]]): Mapping of column names to internal keys.
    """
    def __init__(self, train_on_input: bool = False, column_map: Optional[Dict[str, str]] = None):
        self.train_on_input = train_on_input
        self._column_map = column_map or {
            "trajectory": "trajectory",
        }

    def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        """
        Transform a sample into the format required for training with cross-entropy loss.

        Args:
            sample (Mapping[str, Any]): A sample containing input and output trajectories.

        Returns:
            Mapping[str, Any]: Transformed sample with only chosen (positive) conversations.
        """
        # Extract keys from the sample
        input_trajectories = sample["positive_trajactories_input"]
        output_trajectories = sample["positive_trajactories_output"]

        trajectories = []

        # Iterate over each conversation for chosen responses
        for inp, out in zip(input_trajectories, output_trajectories):
            user_message = [
                Message(role="user", content=inp, masked=not self.train_on_input),
                Message(role="assistant", content=out),
            ]
            trajectories.append(user_message)

        self._column_map["trajectory"]=trajectories
        return self._column_map

class prime_dataloader(Dataset):

    def __init__(
        self,
        *,
        source: str,
        message_transform: Transform,
        tokenizer: ModelTokenizer,
        filter_fn: Optional[Callable] = None,
        **load_dataset_kwargs: Dict[str, Any],
    ) -> None:
        self._tokenizer = tokenizer
        self._message_transform = message_transform
        path=source
        

        if load_dataset_kwargs["split"]=="train":
            path=os.path.join(source, "train.json")
        elif load_dataset_kwargs["split"]=="validation":
            path=os.path.join(source, "valid.json")
        with open(path, "r") as f:
            self._data = json.load(f)
        # self._data = load_dataset(source, **load_dataset_kwargs)
        if filter_fn is not None:
            self._data = self._data.filter(filter_fn)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self._data[index]
        return self._prepare_sample(sample)
    
    def _prepare_sample(self, sample: Mapping[str, Any]) -> Dict[str, Any]:
    
        transformed_sample=self._message_transform(sample)
        
        chosen_input_ids, chosen_masks = zip(*[self._tokenizer.tokenize_messages(msg) for msg in transformed_sample["trajectory"]])
    
        chosen_labels=[list(np.where(chosen_mask, CROSS_ENTROPY_IGNORE_IDX, chosen_input_id)) for chosen_mask, chosen_input_id in zip(chosen_masks, chosen_input_ids)]
        
        tokenized_dict = dict(
            chosen_input_ids=chosen_input_ids,
            chosen_labels=chosen_labels,
            reward=sample["reward"]
            
        )

        return tokenized_dict


def prime_dataset(
    tokenizer: ModelTokenizer,
    *,
    source: str,
    column_map: Optional[Dict[str, str]] = None,
    train_on_input: bool = False,
    filter_fn: Optional[Callable] = None,
    split: str = "train",
    **load_dataset_kwargs: Dict[str, Any],
) -> PreferenceDataset:
    column_map = column_map or {
            "trajectory": "trajectory",
        }

    message_transform = PrimeMessages(
        train_on_input=train_on_input, column_map=column_map
    )

    return prime_dataloader(
        source=source,
        message_transform=message_transform,
        tokenizer=tokenizer,
        filter_fn=filter_fn,
        split=split,
        **load_dataset_kwargs,
    )