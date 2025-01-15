from typing import Any, Callable, Dict, Mapping, Optional

from torchtune.data import Message
from torchtune.datasets._preference import PreferenceDataset, Trajectory_DPO_Dataset, Trajectory_CE_Dataset
from torchtune.modules.tokenizers import ModelTokenizer
from torchtune.modules.transforms import Transform

class MultiConversationMessages(Transform):
    def __init__(self, train_on_input: bool = False, column_map: Optional[Dict[str, str]] = None):
        self.train_on_input = train_on_input
        self._column_map = column_map or {
            "prompt": "prompt",
            "chosen": "chosen",
            "rejected": "rejected",
        }

    def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        # Map columns based on the column_map
        
        chosen_key = self._column_map["chosen"]
        rejected_key = self._column_map["rejected"]

        
        # chosen_responses = sample.get(chosen_key, [])
        # rejected_responses = sample.get(rejected_key, [])
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

        return {"chosen": chosen_conversations, "rejected": rejected_conversations}
    


class CE_MultiConversationMessages(Transform):
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
            "prompt": "prompt",
            "chosen": "chosen",
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
        chosen_input_trajectories = sample["positive_trajactories_input"]
        chosen_output_trajectories = sample["positive_trajactories_output"]

        chosen_conversations = []

        # Iterate over each conversation for chosen responses
        for inp, out in zip(chosen_input_trajectories, chosen_output_trajectories):
            user_message = [
                Message(role="user", content=inp, masked=not self.train_on_input),
                Message(role="assistant", content=out),
            ]
            chosen_conversations.append(user_message)

        return {"chosen": chosen_conversations}
    

def CE_multi_conversation_dataset(
    tokenizer: ModelTokenizer,
    *,
    source: str = "/home/toolkit/scratch/Agents/Checkpoints/data/positive_test.json",
    column_map: Optional[Dict[str, str]] = None,
    train_on_input: bool = False,
    filter_fn: Optional[Callable] = None,
    split: str = "train",
    **load_dataset_kwargs: Dict[str, Any],
) -> PreferenceDataset:
    column_map = column_map or {
        "prompt": "prompt",
        "chosen": "chosen",
    }

    message_transform = CE_MultiConversationMessages(
        train_on_input=train_on_input, column_map=column_map
    )

    return Trajectory_CE_Dataset(
        source=source,
        message_transform=message_transform,
        tokenizer=tokenizer,
        filter_fn=filter_fn,
        split=split,
        data_dir="data/rl",
        **load_dataset_kwargs,
    )


def multi_conversation_dataset(
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
        "prompt": "prompt",
        "chosen": "chosen",
        "rejected": "rejected",
    }

    message_transform = MultiConversationMessages(
        train_on_input=train_on_input, column_map=column_map
    )

    return Trajectory_DPO_Dataset(
        source=source,
        message_transform=message_transform,
        tokenizer=tokenizer,
        filter_fn=filter_fn,
        split=split,
        data_dir="data/rl",
        **load_dataset_kwargs,
    )