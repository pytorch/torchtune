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
from torchdata.stateful_dataloader import Stateful
from torchtune.data._common import CROSS_ENTROPY_IGNORE_IDX
from torchtune.data._metrics import AggregationType, Metric

from torchtune.datasets import TuneIterableDataset
from torchtune.utils._import_guard import _SUPPORTS_FLEX_ATTENTION

logger = logging.getLogger(__name__)


SampleType = TypeVar("SampleType")
PackType = dict[str, torch.Tensor | list[Metric]]


class Packer(ABC, Generic[SampleType]):
    """
    An abstract base class that defines the logic for packing samples into a
    fixed-size sequence. It is used by `IterablePackedDataset` to handle
    different data formats (e.g., standard text, DPO pairs).

    A `Packer` is responsible for:
    1. Defining how to extract the token count from a raw sample.
    2. Specifying how a raw sample is deconstructed into tensors and added
       to a "pack".
    3. Finalizing a pack by padding it to the target sequence length.
    4. Generating the appropriate attention mask for the packed format.

    This modular design allows `IterablePackedDataset` to remain agnostic to
    the data format and packing strategy.

    Args:
            padding_idx (int): The index of the padding token.
            ignore_idx (int): The index to use for labels that should be
                ignored in the loss calculation (e.g., padding tokens).

    Example:
        >>> packer = TextPacker(padding_idx=0, ignore_idx=-100)
        >>> pack = packer.create_empty_pack()
        >>> sample = {"tokens": torch.tensor([1, 2, 3]), "labels": torch.tensor([4, 5, 6])}
        >>> packer.add_sample_to_pack(pack, sample, next_doc_id=0)
        >>> final_pack = packer.finalize_pack(pack, target_tokens_per_pack=5, next_doc_id=1)
        >>> mask = packer.create_block_mask(final_pack["document_ids"].unsqueeze(0), device="cpu")
    
    Raises:
        RuntimeError: If FlexAttention is not supported in the current environment.
    """

    def __init__(self, padding_idx: int, ignore_idx: int = CROSS_ENTROPY_IGNORE_IDX):
        if not _SUPPORTS_FLEX_ATTENTION:
            raise RuntimeError(
                "The IterablePackedDataset and its packers require Flex Attention support, "
                "which is not available in the current environment."
            )
        self.padding_idx = padding_idx
        self.ignore_idx = ignore_idx

    @abstractmethod
    def set_dataset_name(self, dataset_name: str) -> None:
        """
        Sets the dataset name on the packer.

        Args:
            dataset_name (str): The name of the dataset.
        """
        pass

    @abstractmethod
    def create_empty_pack(self) -> dict[str, list[Any]]:
        """
        Creates an empty pack structure for accumulating samples.
        
        Returns:
            dict[str, list[Any]]: An empty structure that can accumulate sample data
                and be converted to tensors by finalize_pack().
        
        Example:
            >>> packer.create_empty_pack()
            {"tokens": [], "labels": []}
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
        Adds a sample to the pack dictionary in-place by appending tensors to lists.

        Args:
            pack (dict[str, list[Any]]): The dictionary representing the pack, to be modified in-place.
            sample (SampleType): The sample to add.
            next_doc_id (int): The starting document ID to use for this sample.

        Returns:
            int: The number of new documents that were added to the pack.

        Example:
            >>> packer = TextPacker(padding_idx=0, ignore_idx=-100)
            >>> pack = {"tokens": [torch.tensor([1, 2])], 
            ...         "labels": [torch.tensor([3, 4])], 
            ...         "document_ids": [torch.tensor([0, 0])], 
            ...         "input_pos": [torch.tensor([0, 1])], 
            ...         "metrics": []}
            >>> sample = {"tokens": torch.tensor([5, 6]), 
            ...         "labels": torch.tensor([7, 8])}
            >>> added_docs = packer.add_sample_to_pack(pack, sample, next_doc_id=1)
            >>> print(pack)
            {"tokens": [torch.tensor([1, 2]), torch.tensor([5, 6])],
             "labels": [torch.tensor([3, 4]), torch.tensor([7, 8])], 
             "document_ids": [torch.tensor([0, 0]), torch.tensor([1, 1])], 
             "input_pos": [torch.tensor([0, 1]), torch.tensor([0, 1])], "metrics": []}
            >>> print(added_docs)
            1
        """
        pass

    @abstractmethod
    def finalize_pack(
        self, pack: dict[str, list[Any]], target_tokens_per_pack: int, next_doc_id: int
    ) -> PackType:
        """
        Finalizes a pack by padding to target length and concatenating tensor lists.

        Args:
            pack (dict[str, list[Any]]): The pack data containing lists of tensors.
            target_tokens_per_pack (int): The target length to pad to.
            next_doc_id (int): The document ID to use for the padding tokens.

        Returns:
            PackType: The finalized pack with concatenated tensors.

        Example:
            >>> packer = TextPacker(padding_idx=999, ignore_idx=-100)
            >>> pack = {"tokens": [torch.tensor([1, 2])], 
            ...         "labels": [torch.tensor([3, 4])], 
            ...         "document_ids": [torch.tensor([0, 0])], 
            ...         "input_pos": [torch.tensor([0, 1])], "metrics": []}
            >>> target_tokens_per_pack = 4
            >>> next_doc_id = 1
            >>> result = packer.finalize_pack(pack, target_tokens_per_pack, next_doc_id)
            >>> print(result)
            {"tokens": torch.tensor([1, 2, 999, 999]),
             "labels": torch.tensor([3, 4, -100, -100]), 
             "document_ids": torch.tensor([0, 0, 1, 1]), 
             "input_pos": torch.tensor([0, 1, 0, 0]), "metrics": [...]}
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

        This method is implemented by each packer to define the specific
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

    def create_block_mask(
        self, batch_document_ids: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """
        Creates a block-causal attention mask for packed sequences using FlexAttention.

        The mask ensures tokens only attend to appropriate positions based on the
        packer's specific attention pattern (e.g., within same document for TextPacker,
        cross-attention for DPOPacker).

        Args:
            batch_document_ids (torch.Tensor): Document IDs of shape (batch_size, seq_len)
            device (torch.device): Device to create the mask on

        Returns:
            torch.Tensor: Block mask for FlexAttention
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


class IterablePackedDataset(
    TuneIterableDataset[PackType], Stateful, Generic[SampleType]
):
    """
    Wraps a `TuneIterableDataset` to combine multiple samples into a single,
    fixed-size "pack". This is highly efficient for training as it minimizes
    padding and ensures consistent batch shapes.

    The packing process works as follows:
    1. It fetches samples from the underlying `dataset` and stores them in
       an internal `buffer`.
    2. It uses a "best-fit" approach to select samples from the buffer that
       can fill a pack up to `target_tokens_per_pack`.
    3. The `packer` handles the logic for deconstructing samples, creating
       metadata (like document IDs and attention masks), and padding the
       final pack.

    This dataset is stateful and supports checkpointing (relies on child dataset to be stateful),
    allowing training to be resumed seamlessly.

    Args:
        dataset (TuneIterableDataset[SampleType]): The `TuneIterableDataset` to pack.
        packer (Packer[SampleType]): The `Packer` that defines the packing
            strategy for the dataset format (e.g. `TextPacker`).
        target_tokens_per_pack (int): The target number of tokens for each pack.
        buffer_size (int): The number of samples to buffer for finding the
            best fit. A larger buffer may improve packing efficiency at the
            cost of memory. Buffer samples are discarded if resuming from a checkpoint.
            Default is 100.
        dataset_name (str): The name of the dataset, used for metrics.
    """

    def __init__(
        self,
        dataset: TuneIterableDataset[SampleType],
        packer: Packer[SampleType],
        target_tokens_per_pack: int,
        buffer_size: int = 100,
        dataset_name: str = "IterablePackedDataset",
    ):
        self.dataset = dataset
        self.packer = packer
        self.target_tokens_per_pack = target_tokens_per_pack
        self.buffer_size = buffer_size
        self._dataset_name = dataset_name

        # Set dataset name on the packer
        self.packer.set_dataset_name(dataset_name)

        self._reset_packer_state()

    @property
    def dataset_name(self) -> str:
        """Returns the dataset name, used for metrics tracking."""
        return self._dataset_name

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
                sample_size = self.packer.get_sample_size(sample)

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
            self._current_pack = self.packer.create_empty_pack()
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
                docs_consumed = self.packer.add_sample_to_pack(
                    self._current_pack, sample, self._current_doc_id_in_pack
                )
                self._current_doc_id_in_pack += docs_consumed
                self._current_pack_size += sample_size
            else:
                # No fitting sample found, so break to finalize the pack
                break

        # If the pack has any content, finalize and return it
        if self._current_pack_size > 0:
            final_pack = self.packer.finalize_pack(
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
            raise TypeError("Dataset is not an iterable")

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


class TextPacker(Packer[dict[str, torch.Tensor]]):
    """
    Packer for packing standard text samples for causal language modeling. It is designed
    to be used with the IterablePackedDataset.
    - Each sample is treated as a separate document.
    - `input_pos` restarts from 0 for each sample.
    - `document_ids` assigns a unique ID to each sample for masking.

    Args:
        padding_idx (int): The index of the padding token.
        ignore_idx (int): The index for ignored labels.
    """

    def __init__(self, padding_idx: int, ignore_idx: int = CROSS_ENTROPY_IGNORE_IDX):
        super().__init__(padding_idx, ignore_idx)
        self.dataset_name = "packed_dataset"  # Default name

    def set_dataset_name(self, dataset_name: str) -> None:
        """
        Sets the dataset name on the packer. This is used for logging metrics.
        """
        self.dataset_name = dataset_name

    def create_empty_pack(self) -> dict[str, list]:
        """Creates an empty pack with lists that will hold tensors."""
        return {
            "tokens": [],
            "labels": [],
            "document_ids": [],
            "input_pos": [],
            "metrics": [],
        }

    def get_sample_size(self, sample: dict[str, torch.Tensor]) -> int:
        """Returns the number of tokens in the sample."""
        return sample["tokens"].numel()

    def add_sample_to_pack(
        self, pack: dict[str, list], sample: dict[str, torch.Tensor], next_doc_id: int
    ) -> int:
        """Adds a tensor sample to the pack by appending tensors to lists."""
        seq_len = sample["tokens"].numel()

        # Append tensors directly to pack lists
        pack["tokens"].append(sample["tokens"])
        pack["labels"].append(sample["labels"])
        
        # Generate metadata as tensors
        pack["document_ids"].append(
            torch.full((seq_len,), next_doc_id, dtype=torch.long, device="cpu")
        )
        # input_pos restarts from 0 for each document
        pack["input_pos"].append(torch.arange(seq_len, dtype=torch.long, device="cpu"))

        # Handle metrics if they exist in the sample
        if "metrics" in sample:
            pack["metrics"].extend(sample["metrics"])

        # return number of documents added
        return 1

    def finalize_pack(
        self, pack: dict[str, list], target_tokens_per_pack: int, next_doc_id: int
    ) -> PackType:
        """Finalizes pack by padding and concatenating tensor lists efficiently."""
        # Calculate current size from tensor list
        current_size = sum(t.numel() for t in pack["tokens"]) if pack["tokens"] else 0
        num_padding = target_tokens_per_pack - current_size

        # Add padding tensors if needed
        if num_padding > 0:
            pack["tokens"].append(
                torch.full((num_padding,), self.padding_idx, dtype=torch.long)
            )
            pack["labels"].append(
                torch.full((num_padding,), self.ignore_idx, dtype=torch.long)
            )
            pack["document_ids"].append(
                torch.full((num_padding,), next_doc_id, dtype=torch.long)
            )
            pack["input_pos"].append(
                torch.zeros(num_padding, dtype=torch.long)
            )

        # Add padding percentage metric
        if target_tokens_per_pack > 0:
            padding_pct = round(num_padding * 100 / target_tokens_per_pack, 2)
            padding_metric = Metric(
                dataset_name=self.dataset_name,
                name="pct_of_tokens_padded",
                value=padding_pct,
                agg_type=AggregationType.MEAN,
            )
            pack["metrics"].append(padding_metric)

        # Concatenate all tensor lists efficiently
        result = {
            "tokens": torch.cat(pack["tokens"]) if pack["tokens"] else torch.empty(0, dtype=torch.long),
            "labels": torch.cat(pack["labels"]) if pack["labels"] else torch.empty(0, dtype=torch.long),
            "document_ids": torch.cat(pack["document_ids"]) if pack["document_ids"] else torch.empty(0, dtype=torch.long),
            "input_pos": torch.cat(pack["input_pos"]) if pack["input_pos"] else torch.empty(0, dtype=torch.long),
            "metrics": pack["metrics"],
        }

        return result

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


class DPOPacker(Packer[dict[str, torch.Tensor]]):
    """
    Packer for Direct Preference Optimization (DPO). It packs a DPO sample
    as three logical documents: a shared prompt, a chosen response, and a rejected response.
    It encodes the attention mask with shared prompt, so that both responses can attend to the same prompt,
    without repetition, but not to each other.

    ASSUMPTION: The input DPO sample dict contains pre-tokenized tensors:
    - "prompt_ids"
    - "chosen_response_only_ids"
    - "chosen_response_only_labels"
    - "rejected_response_only_ids"
    - "rejected_response_only_labels"

     Args:
        padding_idx (int): The index of the padding token.
        ignore_idx (int): The index for ignored labels.
    """

    def __init__(self, padding_idx: int, ignore_idx: int = CROSS_ENTROPY_IGNORE_IDX):
        super().__init__(padding_idx, ignore_idx)
        self.dataset_name = "packed_dataset"  # Default name

    def set_dataset_name(self, dataset_name: str) -> None:
        """
        Sets the dataset name on the packer. This is used for logging metrics.
        """
        self.dataset_name = dataset_name

    def create_empty_pack(self) -> dict[str, list]:
        """Creates an empty pack with lists that will hold tensors."""
        return {
            "tokens": [],
            "labels": [],
            "document_ids": [],
            "input_pos": [],
            "chosen_response_mask": [],
            "rejected_response_mask": [],
            "metrics": [],
        }

    def get_sample_size(self, sample: dict[str, torch.Tensor]) -> int:
        """Returns total size of DPO sample: prompt + both responses."""
        return (
            sample["prompt_ids"].numel()
            + sample["chosen_response_only_ids"].numel()
            + sample["rejected_response_only_ids"].numel()
        )

    def add_sample_to_pack(
        self, pack: dict[str, list], sample: dict[str, torch.Tensor], next_doc_id: int
    ) -> int:
        """
        Adds a DPO sample to a pack. Each DPO sample consists of three parts
        (prompt, chosen, rejected), and each part is assigned its own document ID.
        """
        prompt_len = sample["prompt_ids"].numel()
        chosen_len = sample["chosen_response_only_ids"].numel()
        rejected_len = sample["rejected_response_only_ids"].numel()

        # 1. Concatenate tokens: [prompt, chosen_response, rejected_response]
        tokens = torch.cat(
            [
                sample["prompt_ids"],
                sample["chosen_response_only_ids"],
                sample["rejected_response_only_ids"],
            ]
        )

        # 2. Create labels: [ignore_idx for prompt, chosen_labels, rejected_labels]
        labels = torch.cat(
            [
                torch.full(
                    (prompt_len,), self.ignore_idx, dtype=torch.long),
                sample["chosen_response_only_labels"],
                sample["rejected_response_only_labels"],
            ]
        )

        # 3. Create document IDs: prompt(next_doc_id), chosen(next_doc_id+1), rejected(next_doc_id+2)
        document_ids = torch.cat(
            [
                torch.full(
                    (prompt_len,), next_doc_id, dtype=torch.long),
                torch.full(
                    (chosen_len,), next_doc_id + 1, dtype=torch.long),
                torch.full(
                    (rejected_len,), next_doc_id + 2, dtype=torch.long),
            ]
        )

        # 4. Create input positions (restarts from 0 for each DPO sample)
        total_len = tokens.numel()
        input_pos = torch.arange(total_len, dtype=torch.long, device="cpu")

        # 5. Create response masks
        chosen_response_mask = torch.cat(
            [
                torch.zeros(prompt_len, dtype=torch.bool, device="cpu"),
                torch.ones(chosen_len, dtype=torch.bool, device="cpu"),
                torch.zeros(rejected_len, dtype=torch.bool, device="cpu"),
            ]
        )
        rejected_response_mask = torch.cat(
            [
                torch.zeros(prompt_len, dtype=torch.bool, device="cpu"),
                torch.zeros(chosen_len, dtype=torch.bool, device="cpu"),
                torch.ones(rejected_len, dtype=torch.bool, device="cpu"),
            ]
        )

        # Append all complete tensors to the pack
        pack["tokens"].append(tokens)
        pack["labels"].append(labels)
        pack["document_ids"].append(document_ids)
        pack["input_pos"].append(input_pos)
        pack["chosen_response_mask"].append(chosen_response_mask)
        pack["rejected_response_mask"].append(rejected_response_mask)

        # Handle metrics if they exist in the sample
        if "metrics" in sample:
            pack["metrics"].extend(sample["metrics"])

        # Each DPO sample consists of 3 documents (prompt, chosen, rejected)
        return 3

    def finalize_pack(
        self, pack: dict[str, list], target_tokens_per_pack: int, next_doc_id: int
    ) -> PackType:
        """Finalizes pack by padding and concatenating tensor lists efficiently."""
        # Calculate current size from tensor list
        current_size = sum(t.numel() for t in pack["tokens"]) if pack["tokens"] else 0
        num_padding = target_tokens_per_pack - current_size

        # Add padding tensors if needed
        if num_padding > 0:
            pack["tokens"].append(
                torch.full(
                    (num_padding,), self.padding_idx, dtype=torch.long, device="cpu"
                )
            )
            pack["labels"].append(
                torch.full(
                    (num_padding,), self.ignore_idx, dtype=torch.long, device="cpu"
                )
            )
            pack["document_ids"].append(
                torch.full((num_padding,), next_doc_id, dtype=torch.long, device="cpu")
            )
            pack["input_pos"].append(
                torch.zeros(num_padding, dtype=torch.long, device="cpu")
            )
            pack["chosen_response_mask"].append(
                torch.zeros(num_padding, dtype=torch.bool, device="cpu")
            )
            pack["rejected_response_mask"].append(
                torch.zeros(num_padding, dtype=torch.bool, device="cpu")
            )

        # Add padding percentage metric
        if target_tokens_per_pack > 0:
            padding_pct = round(num_padding * 100 / target_tokens_per_pack, 2)
            padding_metric = Metric(
                dataset_name=self.dataset_name,
                name="pct_of_tokens_padded",
                value=padding_pct,
                agg_type=AggregationType.MEAN,
            )
            pack["metrics"].append(padding_metric)

        # Concatenate all tensor lists
        result = {
            "tokens": torch.cat(pack["tokens"]) if pack["tokens"] else torch.empty(0, dtype=torch.long),
            "labels": torch.cat(pack["labels"]) if pack["labels"] else torch.empty(0, dtype=torch.long),
            "document_ids": torch.cat(pack["document_ids"]) if pack["document_ids"] else torch.empty(0, dtype=torch.long),
            "input_pos": torch.cat(pack["input_pos"]) if pack["input_pos"] else torch.empty(0, dtype=torch.long),
            "chosen_response_mask": torch.cat(pack["chosen_response_mask"]) if pack["chosen_response_mask"] else torch.empty(0, dtype=torch.bool),
            "rejected_response_mask": torch.cat(pack["rejected_response_mask"]) if pack["rejected_response_mask"] else torch.empty(0, dtype=torch.bool),
            "metrics": pack["metrics"],
        }

        return result

    def _mask_mod(
        self,
        b: int,
        h: int,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
        doc_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mask logic for DPO.
        - Causal self-attention within each document (prompt, chosen, rejected)
        - Cross-attention: response tokens can attend to their prompt (shared for both responses)
        """
        # (batch_size, seq_len)
        q_doc = doc_ids[b, q_idx]
        kv_doc = doc_ids[b, kv_idx]

        # 1. Document-level Causal self-attention
        is_same_doc = q_doc == kv_doc
        is_causal = is_same_doc & (q_idx >= kv_idx)

        # 2. Cross-attention from response to prompt
        # For a given query token, find the document ID of its corresponding prompt.
        # Since each DPO sample consists of 3 documents (prompt, chosen, rejected),
        # this maps q_doc to the base ID of its group (e.g., 4 -> 3, 5 -> 3).
        q_prompt_doc_id = (q_doc // 3) * 3
        kv_is_part_of_q_prompt = kv_doc == q_prompt_doc_id
        q_is_response = (q_doc % 3) > 0
        is_cross_attention = q_is_response & kv_is_part_of_q_prompt

        return is_causal | is_cross_attention
