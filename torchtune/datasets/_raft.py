# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtune.data import RAFTInstructTemplate
from torchtune.datasets._instruct_raft import InstructDatasetDeepLakeRAFT
from torchtune.modules.tokenizers import Tokenizer

activeloop_dataset = "hub://manufe/raft_format_dataset_biomedical"  # Replace with your ActiveLoop dataset


def raft_dataset(
    tokenizer: Tokenizer,
    source: str = activeloop_dataset,
    max_seq_len: int = 4096,
) -> InstructDatasetDeepLakeRAFT:
    """
    Load a dataset in RAFT format from ActiveLoop's DeepLake platform.

    This function initializes and returns a dataset in RAFT (Retrieval Augmented Fine Tuning) format
    from ActiveLoop's DeepLake platform. It provides support for training models on RAFT-formatted data
    using the specified tokenizer and other configuration parameters.

    Args:
        tokenizer (Tokenizer): Tokenizer used to encode data. Tokenizer must implement an `encode` and `decode` method.
        source (str): path string of dataset, anything supported by Deep Lake Datasets.
        max_seq_len (int): Maximum number of tokens in the returned input and label token id lists. Default is 512.

    Returns:
        InstructDatasetDeepLakeRAFT: A dataset configured with the provided source data, tokenizer,
        and template for RAFT format.

    Note:
        The RAFT format and its usage in this function are based on the paper "RAFT: Adapting Language Model
        to Domain Specific RAG" by Zhang et al.
        For more information, refer to the paper at: https://arxiv.org/html/2403.10131v1

    Example:
        >>> raft_ds = raft_dataset(tokenizer=tokenizer)
        >>> for batch in DataLoader(raft_ds, batch_size=8):
        >>>     print(f"Batch size: {len(batch)}")
        >>> Batch size: 8
    """
    return InstructDatasetDeepLakeRAFT(
        tokenizer=tokenizer,
        source=source,
        template=RAFTInstructTemplate,
        max_seq_len=max_seq_len,
    )
