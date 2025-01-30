import os
import json

from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Union

from torchtune.datasets._verifiable import VerifiableDataset, PromptToMessage
from torchtune.data import  Message
from torchtune.modules.transforms.tokenizers import ModelTokenizer


def calc_gsm8k_dataset(tokenizer: ModelTokenizer,
    source: str = "MU-NLPC/Calc-gsm8k",
    column_map: Optional[Dict[str, str]] = {"prompt": "question", "result": "result"},
    train_on_input: bool = False,
    new_system_prompt: Optional[str] = None,
    split: str = "train",
    **load_dataset_kwargs: Dict[str, Any],):
    """
    Builds the Calc-GSM8k dataset using TorchTune's PreferenceDataset utilities.

    Args:
        tokenizer (ModelTokenizer): The tokenizer to preprocess the data.

    Returns:
        Dataset: A TorchTune-compatible dataset (VerifiableDataset).
    """
    ds = VerifiableDataset(
        source=source,
        tokenizer=tokenizer,
        message_transform=PromptToMessage(
            train_on_input=train_on_input,
            column_map=column_map,
            new_system_prompt=new_system_prompt,
        ),
        split=split,
        **load_dataset_kwargs,
    )
    
    return ds
