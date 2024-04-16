# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

from torchtune.data import RAFTInstructTemplate
from torchtune.datasets._instruct_deeplake import InstructDatasetDeepLakeRAFT
from torchtune.modules import Tokenizer
import os

try:
    dataset_source_env = os.getenv("dataset_activeloop")
except:
    print("dataset_activeloop not found in environment variables")


def raft_dataset(
    tokenizer: Tokenizer,
    source: str = dataset_source_env,
    train_on_input: bool = True,
    max_seq_len: int = 512,
) -> InstructDatasetDeepLakeRAFT:

    return InstructDatasetDeepLakeRAFT(
        tokenizer=tokenizer,
        source=source,
        template=RAFTInstructTemplate,
        train_on_input=train_on_input,
        max_seq_len=max_seq_len,
        split="train",
    )
