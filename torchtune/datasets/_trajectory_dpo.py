import logging
from typing import Any, Callable, Dict, List, Mapping, Optional

import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset

from torchtune.data import ChosenRejectedToMessages, CROSS_ENTROPY_IGNORE_IDX

from torchtune.modules.tokenizers import ModelTokenizer
from torchtune.modules.transforms import Transform
import random, json
import os

from typing import Any, Callable, Dict, Mapping, Optional

from torchtune.data import Message
from torchtune.modules.tokenizers import ModelTokenizer
from torchtune.modules.transforms import Transform


class trajectoryMessages(Transform):
    def __init__(self, train_on_input: bool = False, column_map: Optional[Dict[str, str]] = None):
        self.train_on_input = train_on_input
        self._column_map = column_map or {
            "chosen": "chosen",
            "rejected": "rejected",
        }

    def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        chosen_input_trajectories=sample["positive_trajactories_input"]
        chosen_output_trajectories=sample["positive_trajactories_output"]

        rejected_input_trajectories=sample["negative_trajectories_input"]
        rejected_output_trajectories=sample["negative_trajectories_output"]



        chosen_conversations = []
        rejected_conversations = []

        # Iterate over each conversation for chosen responses
        for inp, out in zip(chosen_input_trajectories, chosen_output_trajectories):
            user_message = [
                    Message(role="user", content=inp, masked=not self.train_on_input),
                    Message(role="assistant", content=out)
                ]
            chosen_conversations.append(user_message)

        # Iterate over each conversation for rejected responses
        for inp, out in zip(rejected_input_trajectories, rejected_output_trajectories):
            user_message = [
                    Message(role="user", content=inp, masked=not self.train_on_input),
                    Message(role="assistant", content=out)
                ]
            rejected_conversations.append(user_message)
        self._column_map["chosen"]=chosen_conversations
        self._column_map["rejected"]=rejected_conversations

        return self._column_map


class Trajectory_DPO_Dataloader(Dataset):

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
        
        chosen_input_ids, chosen_masks = zip(*[self._tokenizer.tokenize_messages(msg) for msg in transformed_sample["chosen"]])
        rejected_input_ids, rejected_masks = zip(*[self._tokenizer.tokenize_messages(msg) for msg in transformed_sample["rejected"]])

        chosen_labels=[list(np.where(chosen_mask, CROSS_ENTROPY_IGNORE_IDX, chosen_input_id)) for chosen_mask, chosen_input_id in zip(chosen_masks, chosen_input_ids)]
        rejected_labels=[list(np.where(rejected_mask, CROSS_ENTROPY_IGNORE_IDX, rejected_input_id)) for rejected_mask, rejected_input_id in zip(rejected_masks, rejected_input_ids)]
        
        

        tokenized_dict = dict(
            chosen_input_ids=chosen_input_ids,
            chosen_labels=chosen_labels,
            rejected_input_ids=rejected_input_ids,
            rejected_labels=rejected_labels,
        )

        return tokenized_dict
    



def trajectory_dataset(
    tokenizer: ModelTokenizer,
    *,
    source: str,
    column_map: Optional[Dict[str, str]] = None,
    train_on_input: bool = False,
    filter_fn: Optional[Callable] = None,
    split: str = "train",
    **load_dataset_kwargs: Dict[str, Any],
):
    column_map = column_map or {
        "chosen": "chosen",
        "rejected": "rejected",
    }

    message_transform = trajectoryMessages(
        train_on_input=train_on_input, column_map=column_map
    )

    return Trajectory_DPO_Dataloader(
        source=source,
        message_transform=message_transform,
        tokenizer=tokenizer,
        filter_fn=filter_fn,
        split=split,
        **load_dataset_kwargs,
    )