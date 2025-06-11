# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Generic, Iterable, Iterator, Optional, TypeVar

import torch
from torch.nn.attention.flex_attention import (
    create_block_mask as create_block_mask_flex,
)
from torch.utils.data import IterableDataset
from torchdata.stateful_dataloader import Stateful
from torchtune.data._common import CROSS_ENTROPY_IGNORE_IDX
from torchtune.utils._import_guard import _SUPPORTS_FLEX_ATTENTION


logger = logging.getLogger(__name__)

SampleType = TypeVar("SampleType")
PackType = dict[str, torch.Tensor]


class PackingStrategy(ABC, Generic[SampleType]):
    """
    Strategy to be used in IterablePackedDataset and with FlexAttention.
    """

    def __init__(self, padding_idx: int, ignore_idx: int = CROSS_ENTROPY_IGNORE_IDX):
        if not _SUPPORTS_FLEX_ATTENTION:
            raise RuntimeError(
                "The IterablePackedDataset and its strategies require Flex Attention support, "
                "which is not available in the current environment."
            )
        self.padding_idx = padding_idx
        self.ignore_idx = ignore_idx

    @abstractmethod
    def create_empty_pack(self) -> dict[str, list[Any]]:
        """
        Creates an empty pack.

        Returns:
            dict[str, list[Any]]: An empty dictionary with lists as values.

        Example:
            self.create_empty_pack()
            >>> {"tokens": [], "labels": [], "document_ids": [], "input_pos": []}
        """
        pass

    @abstractmethod
    def get_sample_size(self, sample: SampleType) -> int:
        """
        Returns the size of a sample.

        Args:
            sample (SampleType): The sample to get the size of.

        Returns:
            int: The size of the sample.

        Example:
            # for a sample with 100 tokens
            self.get_sample_size(sample)
            >>> 100
        """
        pass

    @abstractmethod
    def add_sample_to_pack(
        self, pack: dict[str, list[Any]], sample: SampleType, next_doc_id: int
    ) -> int:
        """
        Adds a sample to the pack dictionary in-place.

        Args:
            pack (dict[str, list[Any]]): The dictionary representing the pack, to be modified in-place.
            sample (SampleType): The sample to add.
            next_doc_id (int): The starting document ID to use for this sample.

        Returns:
            int: The number of new documents that were added to the pack.

        Example:
            pack = {"tokens": [1, 2], "labels": [3, 4], "document_ids": [0, 0], "input_pos": [0, 1]}
            sample = {"tokens": [5, 6], "labels": [7, 8], "document_ids": [1, 1], "input_pos": [0, 1]}
            added_docs = self.add_sample_to_pack(pack, sample, next_doc_id=1)
            print(pack)
            >>> {"tokens": [1, 2, 5, 6],
                "labels": [3, 4, 7, 8],
                "document_ids": [0, 0, 1, 1],
                "input_pos": [0, 1, 0, 1]}
            print(added_docs)
            >>> 1
        """
        pass

    @abstractmethod
    def finalize_pack(
        self, pack: dict[str, list[Any]], target_tokens_per_pack: int, next_doc_id: int
    ) -> PackType:
        """
        Finalizes a pack, primarily by padding it to the target length.

        Args:
            pack (dict[str, list[Any]]): The pack data.
            target_tokens_per_pack (int): The target length to pad to.
            next_doc_id (int): The document ID to use for the padding tokens.

        Returns:
            PackType: The finalized pack.

        Example:
            pack = {"tokens": [1, 2], "labels": [3, 4], "document_ids": [0, 0], "input_pos": [0, 1]}
            target_tokens_per_pack = 4
            next_doc_id = 1
            self.padding_idx = 999
            self.ignore_idx = -100

            self.finalize_pack(pack, target_tokens_per_pack, next_doc_id)
            >>> {"tokens": [1, 2, 999, 999],
            "labels": [3, 4, -100, -100],
            "document_ids": [0, 0, 1, 1],
            "input_pos": [0, 1, 0, 1]}
        """
        pass

    @abstractmethod
    def _mask_mod(
        self,
        b: int,
        h: int,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
        doc_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        The core logic for the block attention mask, to be passed to
        `torch.nn.attention.flex_attention.create_block_mask`.

        This method is implemented by each strategy to define the specific
        attention pattern (e.g., standard causal, DPO, etc.).

        Args:
            b (int): Batch index.
            h (int): Head index.
            q_idx (torch.Tensor): Query indices.
            kv_idx (torch.Tensor): Key/value indices.
            doc_ids (torch.Tensor): The complete document ID tensor for the batch,
                of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: A boolean tensor indicating which query/key pairs are allowed to attend.
        """
        pass

    def create_block_mask(self, batch_document_ids, device):
        """
        Creates a block-causal attention mask using FlexAttention.
        """
        batch_size, seq_len = batch_document_ids.shape
        doc_ids = batch_document_ids.to(device)

        # This wrapper is needed so we can unit-test the core `mask_mod` logic
        # while still conforming to the function signature required by `create_block_mask_flex`.
        def _mask_mod_for_flex(b, h, q_idx, kv_idx):
            return self._mask_mod(b, h, q_idx, kv_idx, doc_ids)

        return create_block_mask_flex(
            _mask_mod_for_flex, batch_size, None, seq_len, seq_len, device=device
        )


class IterablePackedDataset(IterableDataset[PackType], Stateful, Generic[SampleType]):
    """
    IterablePackedDataset takes any IterableDataset and a PackingStrategy, packs documents until
    the 'target_tokens_per_pack' is reached and yields a dictionary of tensors.

    Args:
        dataset (IterableDataset[SampleType]): The IterableDataset to pack.
        strategy (PackingStrategy[SampleType]): The PackingStrategy to use for packing.
        target_tokens_per_pack (int): The target number of tokens per pack.
        buffer_size (int): The size of the buffer to use for packing.
    """

    def __init__(
        self,
        dataset: IterableDataset[SampleType],
        strategy: PackingStrategy[SampleType],
        target_tokens_per_pack: int,
        buffer_size: int = 50,
    ):
        self.dataset = dataset
        self.strategy = strategy
        self.target_tokens_per_pack = target_tokens_per_pack
        self.buffer_size = buffer_size

        self._reset_packer_state()

    def _reset_packer_state(self) -> None:
        """Resets the packer's internal state for a new or resumed iteration."""
        # buffer: deque of (sample, size) tuples that have not been added to a pack yet
        if not hasattr(self, "_buffer"):
            self._buffer: deque[tuple[SampleType, int]] = deque()
        else:
            self._buffer.clear()

        # current_pack: the current pack being built
        self._current_pack: Optional[dict[str, list]] = None

        # current_pack_size: the number of tokens in the current pack
        self._current_pack_size: int = 0

        # iterator: the iterator over the dataset
        self._iterator: Optional[Iterator[SampleType]] = None

        # current_doc_id_in_pack: the document ID to use for the next sample
        self._current_doc_id_in_pack: int = 0

        # exhausted: whether the dataset is exhausted
        self._exhausted: bool = False

        # resuming: whether the packer is resuming from a checkpoint
        self._resuming: bool = False

    def _fill_buffer(self, iterator: Iterator[SampleType]) -> None:
        """
        Fills the buffer with samples from the dataset.
        The buffer is a deque of (sample, size) tuples that have not been added to a pack yet.
        """
        # Fill buffer until it's full or the dataset is exhausted
        while len(self._buffer) < self.buffer_size and not self._exhausted:
            try:
                sample = next(iterator)
                sample_size = self.strategy.get_sample_size(sample)

                # Drop samples that are too large
                if sample_size > self.target_tokens_per_pack:
                    logger.warning(
                        f"Dropping sample with size {sample_size} > target_tokens_per_pack {self.target_tokens_per_pack}."
                    )
                else:
                    self._buffer.append((sample, sample_size))
            except StopIteration:
                self._exhausted = True

    def _find_next_fitting_sample(self, remaining_size: int) -> Optional[int]:
        """
        Find the first sample in the buffer that fits in the remaining space.

        Args:
            remaining_size (int): The remaining space in the current pack.

        Returns:
            Optional[int]: The index of the sample in the buffer, or None if no sample fits.

        Example:
            self._buffer = deque([(sample1, 200), (sample2, 100), (sample3, 48), (sample4, 200)])

            # First iteration:
            selected_sample_idx = self._find_next_fitting_sample(remaining_size=150) # returns 1
            del self._buffer[selected_sample_idx]

            # Second iteration:
            selected_sample_idx = self._find_next_fitting_sample(remaining_size=50) # returns 1
            del self._buffer[selected_sample_idx]

            # Third iteration:
            selected_sample_idx = self._find_next_fitting_sample(remaining_size=2) # returns None
        """
        # Find the first sample in the buffer that fits in the remaining space
        for i, (_, sample_size) in enumerate(self._buffer):
            if sample_size <= remaining_size:
                return i
        return None

    def _build_one_pack(self, iterator: Iterator[SampleType]) -> Optional[PackType]:
        """
        Builds a pack of samples from the buffer.

        Args:
            iterator (Iterator[SampleType]): The iterator over the dataset.

        Returns:
            Optional[PackType]: The pack of samples, or None if the dataset is exhausted.
        """
        # Start a new pack if necessary
        if self._current_pack is None:
            self._current_pack = self.strategy.create_empty_pack()
            self._current_pack_size = 0
            self._current_doc_id_in_pack = 0

        # Fill the current pack until it's full
        while self._current_pack_size < self.target_tokens_per_pack:
            self._fill_buffer(iterator)
            remaining_size = self.target_tokens_per_pack - self._current_pack_size
            selected_sample_idx = self._find_next_fitting_sample(remaining_size)

            # If a fitting sample is found, del from buffer and add to the pack
            if selected_sample_idx is not None:
                sample, sample_size = self._buffer[selected_sample_idx]
                del self._buffer[selected_sample_idx]
                docs_consumed = self.strategy.add_sample_to_pack(
                    self._current_pack, sample, self._current_doc_id_in_pack
                )
                self._current_doc_id_in_pack += docs_consumed
                self._current_pack_size += sample_size
            else:
                # No fitting sample found, so break to finalize the pack
                break

        # If the pack has any content, finalize and return it
        if self._current_pack_size > 0:
            final_pack = self.strategy.finalize_pack(
                self._current_pack,
                self.target_tokens_per_pack,
                self._current_doc_id_in_pack,
            )
            self._current_pack = None
            self._current_pack_size = 0
            return final_pack

        if self._exhausted and not self._buffer:
            return None

        return None

    def __iter__(self) -> Iterator[PackType]:
        if not isinstance(self.dataset, Iterable):
            raise TypeError("Dataset is not iterable.")

        if not self._resuming:
            self._reset_packer_state()
            self._iterator = iter(self.dataset)

        # If resuming, the iterator must be recreated from the loaded state
        if self._iterator is None:
            self._iterator = iter(self.dataset)

        self._resuming = False  # Consume the resume flag

        # Main packing loop
        while True:

            # Stop if the source is exhausted and there's no data left to pack
            if self._exhausted and not self._buffer and self._current_pack_size == 0:
                break

            pack = self._build_one_pack(self._iterator)
            if pack:
                yield pack

            # If build_one_pack returns None but we are not done, continue loop
            # to attempt building another pack (e.g. after buffer is refilled).
            elif self._exhausted and not self._buffer:
                break

    def state_dict(self) -> dict[str, Any]:
        """
        Get the state of the packer. It relies on the input dataset to save the progress of iteration.
        It does NOT save the internal buffer or any partially built pack.
        """
        state = {}
        if isinstance(self.dataset, Stateful):
            state["dataset_state"] = self.dataset.state_dict()
        else:
            raise ValueError("Dataset is not stateful.")

        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """
        Load the state of the packer. This restores the state of the underlying dataset.
        The buffer and any partially-built pack are discarded.
        """
        if isinstance(self.dataset, Stateful) and "dataset_state" in state_dict:
            self.dataset.load_state_dict(state_dict["dataset_state"])
        else:
            raise ValueError("Dataset is not stateful.")

        self._reset_packer_state()
        self._resuming = True


class TextPackingStrategy(PackingStrategy[dict[str, list[int]]]):
    """
    Strategy for packing standard text samples for causal language modeling. It is designed
    to be used with the IterablePackedDataset.
    - Each sample is treated as a separate document.
    - `input_pos` restarts from 0 for each sample.
    - `document_ids` assigns a unique ID to each sample for masking.
    """

    def __init__(self, padding_idx: int, ignore_idx: int = CROSS_ENTROPY_IGNORE_IDX):
        super().__init__(padding_idx=padding_idx, ignore_idx=ignore_idx)

    def create_empty_pack(self) -> dict[str, list[int]]:
        return {
            "tokens": [],
            "labels": [],
            "document_ids": [],
            "input_pos": [],
        }

    def get_sample_size(self, sample: dict[str, list[int]]) -> int:
        return len(sample["tokens"])

    def add_sample_to_pack(
        self, pack: dict[str, list[int]], sample: dict[str, list[int]], next_doc_id: int
    ) -> int:
        seq_len = len(sample["tokens"])

        # Append sample data to the pack
        pack["tokens"].extend(sample["tokens"])
        pack["labels"].extend(sample["labels"])
        pack["document_ids"].extend([next_doc_id] * seq_len)
        pack["input_pos"].extend(range(seq_len))  # input_pos restarts for each doc

        # Increment doc ID for the next sample
        return 1

    def finalize_pack(
        self, pack: dict[str, list[int]], target_tokens_per_pack: int, next_doc_id: int
    ) -> PackType:
        current_size = len(pack["tokens"])
        num_padding = target_tokens_per_pack - current_size

        if num_padding > 0:
            pack["tokens"].extend([self.padding_idx] * num_padding)
            pack["labels"].extend([self.ignore_idx] * num_padding)
            pack["input_pos"].extend([0] * num_padding)
            pack["document_ids"].extend([next_doc_id] * num_padding)

        return {
            "tokens": torch.tensor(pack["tokens"], dtype=torch.long),
            "labels": torch.tensor(pack["labels"], dtype=torch.long),
            "document_ids": torch.tensor(pack["document_ids"], dtype=torch.long),
            "input_pos": torch.tensor(pack["input_pos"], dtype=torch.long),
        }

    def _mask_mod(
        self,
        b: int,
        h: int,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
        doc_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Standard block-causal mask logic. Tokens can only attend to other
        tokens within the same document, respecting causality.
        """
        causal_mask = q_idx >= kv_idx
        document_mask = doc_ids[b, q_idx] == doc_ids[b, kv_idx]
        return causal_mask & document_mask
