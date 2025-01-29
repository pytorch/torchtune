import os
import json

from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Union

from datasets._verifiable import VerifiableDataset, PromptToMessage
from torchtune.data import  Message
from torchtune.modules.transforms.tokenizers import ModelTokenizer


def calc_gsm8k_dataset(tokenizer: ModelTokenizer):
    """
    Builds the Calc-GSM8k dataset using TorchTune's PreferenceDataset utilities.

    Args:
        tokenizer (ModelTokenizer): The tokenizer to preprocess the data.

    Returns:
        Dataset: A TorchTune-compatible dataset (VerifiableDataset).
    """
    ds = VerifiableDataset(
        source="MU-NLPC/Calc-gsm8k",
        tokenizer=tokenizer,
        message_transform=PromptToMessage(
            train_on_input=False,
            column_map={"prompt": "question"},
        ),
        split="train",
    )
    
    return ds
