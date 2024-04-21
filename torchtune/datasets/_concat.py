# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict

from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset
from torchtune.modules.tokenizers import Tokenizer
from tqdm import tqdm


class ConcatDataset(Dataset):
    """
    Class that enables continued pretraining by packing a single column of text data into equally sized samples.

    The class loads, tokenizes, and packs examples on initialization - no tokenization is done during training.

    Streaming datasets are supported. To stream a dataset, pass `'streaming': True` in the `load_dataset_kwargs`.
    If streaming, the total number of samples will be unknown until the end of the dataset, and the progress bar
    will be indeterminate.

    The general flow is:

    On initialization:
    optionally shuffle sample -> load sample -> tokenize and add to buffer ->
        when buffer is long enough, add to self.samples.

    During training:
    return self.samples[idx] as input and label.

    Args:
        tokenizer (Tokenizer): Tokenizer used to encode data. Tokenize must implement an `encode` and `decode` method.
        source (str): path string of dataset, anything supported by Hugging Face's `load_dataset`,
            or if `is_local` is true, a local path to load with `load_from_disk`.
        text_column (str): column name of the dataset to pack into samples.
        max_seq_len (int): Maximum number of tokens in the returned input and label token id lists.
            Default is None, disabling truncation. We recommend setting this to the highest you can fit in memory
            and is supported by the model. For example, llama2-7B supports up to 4096 for sequence length.
        max_rows (int): maximum number of samples to pack. Default is None, which will pack as many samples as possible.
        is_local (bool): whether the source is a local path. Default is False.
        shuffle_before_packing (bool): whether to shuffle the dataset before packing. Default is False.
        seed (int): seed for shuffling. Default is 29.
        train_split_name (str): name of the split to use. Default is None, which will use the full dataset.
        **load_dataset_kwargs (Dict[str, Any]): additional keyword arguments to pass to `load_dataset`.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        source: str,
        text_column: str,
        max_seq_len: int,
        max_rows: int = None,
        is_local: bool = False,
        shuffle_before_packing: bool = False,
        seed: int = 29,
        train_split_name: str = None,
        **load_dataset_kwargs: Dict[str, Any],
    ):

        assert tokenizer is not None, "tokenizer must be provided"
        assert source is not None, "source must be provided"
        assert text_column is not None, "text_column must be provided"
        assert max_seq_len is not None, "max_seq_len must be provided"

        if is_local:
            dataset = load_from_disk(source, **load_dataset_kwargs)
        else:
            dataset = load_dataset(source, **load_dataset_kwargs)

        if train_split_name is not None:
            dataset = dataset[train_split_name]

        if shuffle_before_packing:
            dataset = dataset.shuffle(seed=seed)

        # where final samples will be held
        self.samples = []

        # buffer to hold samples until they are long enough to be added to self.samples
        buffer = []

        def tokenize(sample):
            tokenized = tokenizer.encode(
                sample[text_column].strip(),
                add_bos=True,
                add_eos=True,
            )
            return {"input_ids": tokenized}

        pbar_total = None
        if max_rows is not None:
            pbar_total = max_rows
        # trying to get length of streaming dataset will throw an error
        elif (
            "streaming" in load_dataset_kwargs
            and load_dataset_kwargs["streaming"] is not True
        ):
            pbar_total = len(dataset)

        # create max_rows number of packed samples. If max_rows is None, create as many as possible
        pbar = tqdm(
            dataset, total=pbar_total, desc="Packing dataset", dynamic_ncols=True
        )
        for sample in dataset:
            if max_rows is not None:
                if len(self.samples) >= max_rows:
                    break

            sample = tokenize(sample)

            buffer = buffer + sample["input_ids"]
            while len(buffer) >= max_seq_len + 1:
                if max_rows is not None:
                    if len(self.samples) >= max_rows:
                        break

                self.samples.append(buffer[:max_seq_len])
                buffer = buffer[max_seq_len:]

                if max_rows is not None:
                    pbar.update(1)

            if max_rows is None:
                pbar.update(1)

    def __getitem__(self, idx):
        return self.samples[idx], self.samples[idx]

    def __len__(self):
        return len(self.samples)
