# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Literal, Mapping, Optional, Union

from torchtune.data import StringToIntLabelTransform
from torchtune.datasets._classification import ClassificationDataset
from torchtune.modules.tokenizers import ModelTokenizer

from torchtune.modules.transforms import Transform


class LMSYSModerationLabelTransform(Transform):
    def __init__(self, label_map):
        
        self.label_map = label_map

    def __call__(self, labels):
        


def lmsys_chat_dataset(
    tokenizer: ModelTokenizer,
    *,
    source: str = "stanfordnlp/imdb",
    column_map: Optional[Dict[str, str]] = None,
    split: str = "train",
):
    return ClassificationDataset(
        source=source,
        model_transform=tokenizer,
        label_transform=label_transform,
        column_map=column_map,
        split=split,
    )
