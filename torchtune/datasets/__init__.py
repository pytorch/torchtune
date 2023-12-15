# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch.utils.data import Dataset

from typing import Tuple

from .alpaca import InstructionTuningDataset
from datasets import load_dataset

def get_dataset(name: str, **kwargs) -> Dataset:
    if name == "alpaca":
        tokenizer = kwargs["tokenizer"]
        split = kwargs["split"]

        def row_to_input_and_label(sample: str) -> Tuple[str, str]:
            response_tag = "\n\n### Response:\n"
            split_text = sample["text"].split(response_tag)
            return (split_text[0] + response_tag, split_text[1])

        return InstructionTuningDataset(
            load_dataset("tatsu-lab/alpaca", split=split),
            tokenizer, row_to_input_and_label)
    else:
        raise ValueError(f"Unknown dataset: {name}")
