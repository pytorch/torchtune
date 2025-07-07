# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from functools import partial
from typing import Any, Iterator, Optional

import pytest
import torch
from torch.utils.data import IterableDataset
from torchdata.stateful_dataloader import Stateful, StatefulDataLoader

from torchtune.data._collate import collate_packed
from torchtune.datasets._iterable_base import DatasetInfo
from torchtune.datasets._iterable_packed import (
    DPOPacker,
    IterablePackedDataset,
    Packer,
    PackType,
    TextPacker,
)
from torchtune.data.metrics import MetricsAggregator
from torchtune.utils._import_guard import _SUPPORTS_FLEX_ATTENTION
from .test_iterable_utils import generate_ckpt

# --- Test Fixtures ---


@pytest.fixture
def device():
    return "cuda"


class DummyTextDataset(IterableDataset):
    """Dummy dataset that returns tensor-based samples."""

    def __init__(self, sample_sizes):
        self._sample_sizes = sample_sizes
        self._counter = 0

    @property
    def info(self) -> DatasetInfo:
        """Returns dataset information."""
        return DatasetInfo(name="DummyTextDataset", weight=1.0, children=())

    def __iter__(self):
        # Reset counter for each new iteration
        self._counter = 0
        for size in self._sample_sizes:
            yield {
                "tokens": torch.full((size,), self._counter, dtype=torch.long),
                "labels": torch.full((size,), self._counter, dtype=torch.long),
            }
            self._counter += 1


class StatefulDummyTextDataset(IterableDataset, Stateful):
    """
    A dummy text dataset that is also stateful, allowing its iteration
    progress to be saved and loaded. Returns tensor-based samples.
    """

    def __init__(self, sample_sizes: list[int]):
        self.sample_sizes = sample_sizes
        self._state_to_load: Optional[dict[str, Any]] = None
        # The state is the index of the *next* sample to be processed.
        self._active_iterator_state: dict[str, Any] = {"sample_idx": 0}

    @property
    def info(self) -> DatasetInfo:
        """Returns dataset information."""
        return DatasetInfo(name="StatefulDummyTextDataset", weight=1.0, children=())

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        # This base generator yields all samples from the beginning.
        def _base_iterator():
            for i, size in enumerate(self.sample_sizes):
                self._active_iterator_state = {"sample_idx": i}
                yield {
                    "tokens": torch.full(
                        (size,), i, dtype=torch.long
                    ),  # Use sample index as token value
                    "labels": torch.full((size,), i, dtype=torch.long),
                }
            # After iterating, the next sample index is out of bounds
            self._active_iterator_state = {"sample_idx": len(self.sample_sizes)}

        iterator = _base_iterator()

        # If resuming, fast-forward the iterator to the correct position.
        if self._state_to_load:
            start_idx = self._state_to_load.get("sample_idx", 0)
            logging.info(
                f"StatefulDummyTextDataset.__iter__(): Resuming. Fast-forwarding iterator to index {start_idx}."
            )
            self._state_to_load = None
            # Fast-forward the iterator to the sample index from the checkpoint.
            for _ in range(start_idx):
                next(
                    iterator, None
                )  # Consume and discard samples until the desired start point.

        yield from iterator

    def state_dict(self) -> dict[str, Any]:
        logging.info(
            f"StatefulDummyTextDataset.state_dict(): current state is {self._active_iterator_state}"
        )
        return self._active_iterator_state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        logging.info(
            f"StatefulDummyTextDataset.load_state_dict(): state to load is {state_dict}"
        )
        self._state_to_load = state_dict


class DummyDPODataset(IterableDataset):
    """Dummy DPO dataset that returns tensor-based samples."""

    def __init__(self, samples):
        self._samples = samples

    @property
    def info(self) -> DatasetInfo:
        """Returns dataset information."""
        return DatasetInfo(name="DummyDPODataset", weight=1.0, children=())

    def __iter__(self):
        yield from self._samples


# --- Test Utilities ---


def create_dense_mask_from_mask_mod(
    strategy: Packer, doc_ids: torch.Tensor
) -> torch.Tensor:
    """
    Helper utility to generate a dense boolean attention mask from a
    strategy's _mask_mod implementation for testing purposes.
    """
    batch_size, seq_len = doc_ids.shape
    device = doc_ids.device
    dense_mask = torch.zeros(
        batch_size, seq_len, seq_len, dtype=torch.bool, device=device
    )
    for b in range(batch_size):
        for q_idx in range(seq_len):
            q_tensor = torch.tensor(q_idx, device=device)
            for kv_idx in range(seq_len):
                kv_tensor = torch.tensor(kv_idx, device=device)
                # h (head index) is not used in current implementations
                dense_mask[b, q_idx, kv_idx] = strategy._mask_mod(
                    b, 0, q_tensor, kv_tensor, doc_ids
                )
    return dense_mask


@pytest.fixture
def dpo_packer():
    packer = DPOPacker(padding_idx=999, ignore_idx=-100)
    packer.set_dataset_name("TestDPODataset")
    return packer


# --- Test Classes ---


@pytest.fixture
def text_packer():
    packer = TextPacker(padding_idx=999, ignore_idx=-100)
    packer.set_dataset_name("TestTextDataset")
    return packer


@pytest.mark.skipif(not _SUPPORTS_FLEX_ATTENTION, reason="Flex attention not supported")
class TestTextPacker:
    """Test TextPacker methods, attention masks, and integration workflow"""

    def test_create_empty_pack(self, text_packer):
        """Test empty pack creation for TextPacker"""
        pack = text_packer.create_empty_pack()
        expected = {
            "tokens": [],
            "labels": [],
            "document_ids": [],
            "input_pos": [],
            "metrics": [],
        }
        assert pack == expected

    def test_get_sample_size(self, text_packer):
        """Test sample size calculation for multiple TextPacker samples"""
        samples = [
            {"tokens": torch.tensor([1, 2, 3]), "labels": torch.tensor([4, 5, 6])},
            {"tokens": torch.tensor([7]), "labels": torch.tensor([8])},
            {
                "tokens": torch.tensor([9, 10, 11, 12]),
                "labels": torch.tensor([13, 14, 15, 16]),
            },
        ]

        expected_sizes = [3, 1, 4]
        for sample, expected_size in zip(samples, expected_sizes):
            assert text_packer.get_sample_size(sample) == expected_size

    def test_add_multiple_samples_to_pack(self, text_packer):
        """Test adding multiple samples to same pack"""
        pack = text_packer.create_empty_pack()

        samples = [
            {"tokens": torch.tensor([1, 2]), "labels": torch.tensor([3, 4])},
            {"tokens": torch.tensor([5, 6, 7]), "labels": torch.tensor([8, 9, 10])},
            {"tokens": torch.tensor([11]), "labels": torch.tensor([12])},
        ]

        # Add all samples
        for i, sample in enumerate(samples):
            docs_consumed = text_packer.add_sample_to_pack(pack, sample, next_doc_id=i)
            assert docs_consumed == 1

        # Verify pack contents
        assert len(pack["tokens"]) == 3
        torch.testing.assert_close(pack["tokens"][0], torch.tensor([1, 2]))
        torch.testing.assert_close(pack["tokens"][1], torch.tensor([5, 6, 7]))
        torch.testing.assert_close(pack["tokens"][2], torch.tensor([11]))
        torch.testing.assert_close(pack["document_ids"][0], torch.tensor([0, 0]))
        torch.testing.assert_close(pack["document_ids"][1], torch.tensor([1, 1, 1]))
        torch.testing.assert_close(pack["document_ids"][2], torch.tensor([2]))

    def test_finalize_pack_multiple_samples(self, text_packer):
        """Test pack finalization with multiple samples and padding"""
        pack = {
            "tokens": [torch.tensor([1, 2]), torch.tensor([3, 4, 5])],
            "labels": [torch.tensor([6, 7]), torch.tensor([8, 9, 10])],
            "document_ids": [torch.tensor([0, 0]), torch.tensor([1, 1, 1])],
            "input_pos": [torch.tensor([0, 1]), torch.tensor([0, 1, 2])],
            "metrics": [],
        }

        result = text_packer.finalize_pack(
            pack, target_tokens_per_pack=8, next_doc_id=2
        )

        expected_tokens = torch.tensor([1, 2, 3, 4, 5, 999, 999, 999])
        expected_labels = torch.tensor([6, 7, 8, 9, 10, -100, -100, -100])
        expected_doc_ids = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2])
        expected_input_pos = torch.tensor([0, 1, 0, 1, 2, 0, 0, 0])

        torch.testing.assert_close(result["tokens"], expected_tokens)
        torch.testing.assert_close(result["labels"], expected_labels)
        torch.testing.assert_close(result["document_ids"], expected_doc_ids)
        torch.testing.assert_close(result["input_pos"], expected_input_pos)

    def test_text_causal_mask(self, device):
        """
        Verify the correctness of the causal attention mask by manually constructing
        the expected mask for a batch containing multiple documents.
        """
        text_packer = TextPacker(padding_idx=0)

        # Batch contains two packs of different layouts.
        # Pack 1: docs [A(2), B(3), C(2)]
        doc_ids_1 = torch.tensor([0, 0, 1, 1, 1, 2, 2], device=device)
        # Pack 2: docs [D(4), E(1), F(2)]
        doc_ids_2 = torch.tensor([0, 0, 0, 0, 1, 2, 2], device=device)
        batch_doc_ids = torch.stack([doc_ids_1, doc_ids_2])

        # Manually create the expected mask for the batch
        mask1 = torch.tensor(
            [
                # k_idx -> A  A  B  B  B  C  C
                [1, 0, 0, 0, 0, 0, 0],  # q=0 (A)
                [1, 1, 0, 0, 0, 0, 0],  # q=1 (A)
                [0, 0, 1, 0, 0, 0, 0],  # q=2 (B)
                [0, 0, 1, 1, 0, 0, 0],  # q=3 (B)
                [0, 0, 1, 1, 1, 0, 0],  # q=4 (B)
                [0, 0, 0, 0, 0, 1, 0],  # q=5 (C)
                [0, 0, 0, 0, 0, 1, 1],  # q=6 (C)
            ],
            dtype=torch.bool,
            device=device,
        )
        mask2 = torch.tensor(
            [
                # k_idx -> D  D  D  D  E  F  F
                [1, 0, 0, 0, 0, 0, 0],  # q=0 (D)
                [1, 1, 0, 0, 0, 0, 0],  # q=1 (D)
                [1, 1, 1, 0, 0, 0, 0],  # q=2 (D)
                [1, 1, 1, 1, 0, 0, 0],  # q=3 (D)
                [0, 0, 0, 0, 1, 0, 0],  # q=4 (E)
                [0, 0, 0, 0, 0, 1, 0],  # q=5 (F)
                [0, 0, 0, 0, 0, 1, 1],  # q=6 (F)
            ],
            dtype=torch.bool,
            device=device,
        )
        expected_mask = torch.stack([mask1, mask2])

        # Create mask using the strategy and verify
        actual_mask = create_dense_mask_from_mask_mod(text_packer, batch_doc_ids)
        torch.testing.assert_close(actual_mask, expected_mask)

    def test_text_packing_workflow_two_packs(self):
        """Test complete text workflow that creates exactly 2 packs with multiple samples"""
        # Design: Pack1=[3,2], Pack2=[4] to create 2 packs
        sample_sizes = [3, 2, 4]
        target_tokens = 6

        dataset = DummyTextDataset(sample_sizes)
        text_packer = TextPacker(padding_idx=999, ignore_idx=-100)
        packed_dataset = IterablePackedDataset(
            dataset=dataset, packer=text_packer, target_tokens_per_pack=target_tokens
        )

        packs = list(packed_dataset)
        assert len(packs) == 2

        # Pack 1: samples 0(size 3) + 1(size 2) + padding(1)
        pack1 = packs[0]
        assert pack1["tokens"].shape == (target_tokens,)
        assert (pack1["labels"] != -100).sum() == 5  # 3 + 2 real tokens
        expected_tokens_1 = torch.tensor([0, 0, 0, 1, 1, 999])
        expected_doc_ids_1 = torch.tensor([0, 0, 0, 1, 1, 2])
        expected_input_pos_1 = torch.tensor([0, 1, 2, 0, 1, 0])
        torch.testing.assert_close(pack1["tokens"], expected_tokens_1)
        torch.testing.assert_close(pack1["document_ids"], expected_doc_ids_1)
        torch.testing.assert_close(pack1["input_pos"], expected_input_pos_1)

        # Pack 2: sample 2(size 4) + padding(2) - single sample pack
        pack2 = packs[1]
        assert (pack2["labels"] != -100).sum() == 4  # 4 real tokens
        expected_tokens_2 = torch.tensor([2, 2, 2, 2, 999, 999])
        expected_doc_ids_2 = torch.tensor([0, 0, 0, 0, 1, 1])
        expected_input_pos_2 = torch.tensor([0, 1, 2, 3, 0, 0])
        torch.testing.assert_close(pack2["tokens"], expected_tokens_2)
        torch.testing.assert_close(pack2["document_ids"], expected_doc_ids_2)
        torch.testing.assert_close(pack2["input_pos"], expected_input_pos_2)


@pytest.mark.skipif(not _SUPPORTS_FLEX_ATTENTION, reason="Flex attention not supported")
class TestDPOPacker:
    """Test DPOPacker methods, attention masks, and integration workflow"""

    def test_create_empty_pack(self, dpo_packer):
        """Test empty pack creation for DPOPacker"""
        pack = dpo_packer.create_empty_pack()
        expected = {
            "tokens": [],
            "labels": [],
            "document_ids": [],
            "input_pos": [],
            "chosen_response_mask": [],
            "rejected_response_mask": [],
            "metrics": [],
        }
        assert pack == expected

    def test_get_sample_size(self, dpo_packer):
        """Test sample size calculation for multiple DPOPacker samples"""
        samples = [
            {
                "prompt_ids": torch.tensor([1, 2]),
                "chosen_response_only_ids": torch.tensor([3, 4]),
                "chosen_response_only_labels": torch.tensor([3, 4]),
                "rejected_response_only_ids": torch.tensor([5, 6]),
                "rejected_response_only_labels": torch.tensor([5, 6]),
            },
            {
                "prompt_ids": torch.tensor([7, 8, 9]),
                "chosen_response_only_ids": torch.tensor([10, 11]),
                "chosen_response_only_labels": torch.tensor([10, 11]),
                "rejected_response_only_ids": torch.tensor([12, 13, 14, 15]),
                "rejected_response_only_labels": torch.tensor([12, 13, 14, 15]),
            },
            {
                "prompt_ids": torch.tensor([16]),
                "chosen_response_only_ids": torch.tensor([17, 18, 19]),
                "chosen_response_only_labels": torch.tensor([17, 18, 19]),
                "rejected_response_only_ids": torch.tensor([20, 21]),
                "rejected_response_only_labels": torch.tensor([20, 21]),
            },
        ]

        expected_sizes = [6, 9, 6]  # [2+2+2, 3+2+4, 1+3+2]
        for sample, expected_size in zip(samples, expected_sizes):
            assert dpo_packer.get_sample_size(sample) == expected_size

    def test_add_multiple_samples_to_pack(self, dpo_packer):
        """Test adding multiple DPO samples to pack"""
        pack = dpo_packer.create_empty_pack()
        samples = [
            {
                "prompt_ids": torch.tensor([1, 2]),
                "chosen_response_only_ids": torch.tensor([3, 4]),
                "chosen_response_only_labels": torch.tensor([3, 4]),
                "rejected_response_only_ids": torch.tensor([5, 6]),
                "rejected_response_only_labels": torch.tensor([5, 6]),
            },
            {
                "prompt_ids": torch.tensor([7, 8]),
                "chosen_response_only_ids": torch.tensor([9]),
                "chosen_response_only_labels": torch.tensor([9]),
                "rejected_response_only_ids": torch.tensor([10, 11]),
                "rejected_response_only_labels": torch.tensor([10, 11]),
            },
        ]

        # Add all samples
        for i, sample in enumerate(samples):
            docs_consumed = dpo_packer.add_sample_to_pack(
                pack, sample, next_doc_id=i * 3
            )
            assert docs_consumed == 3  # prompt + chosen + rejected

        # Verify pack contents
        assert len(pack["tokens"]) == 2
        # First sample: [1,2,3,4,5,6]
        torch.testing.assert_close(pack["tokens"][0], torch.tensor([1, 2, 3, 4, 5, 6]))
        torch.testing.assert_close(
            pack["labels"][0], torch.tensor([-100, -100, 3, 4, 5, 6])
        )
        torch.testing.assert_close(
            pack["document_ids"][0], torch.tensor([0, 0, 1, 1, 2, 2])
        )
        torch.testing.assert_close(
            pack["chosen_response_mask"][0],
            torch.tensor([False, False, True, True, False, False]),
        )
        torch.testing.assert_close(
            pack["rejected_response_mask"][0],
            torch.tensor([False, False, False, False, True, True]),
        )

        # Second sample: [7,8,9,10,11]
        torch.testing.assert_close(pack["tokens"][1], torch.tensor([7, 8, 9, 10, 11]))
        torch.testing.assert_close(
            pack["labels"][1], torch.tensor([-100, -100, 9, 10, 11])
        )
        torch.testing.assert_close(
            pack["document_ids"][1], torch.tensor([3, 3, 4, 5, 5])
        )
        torch.testing.assert_close(
            pack["chosen_response_mask"][1],
            torch.tensor([False, False, True, False, False]),
        )
        torch.testing.assert_close(
            pack["rejected_response_mask"][1],
            torch.tensor([False, False, False, True, True]),
        )

    def test_finalize_pack_multiple_dpo_samples(self, dpo_packer):
        """Test DPO pack finalization with multiple samples and padding."""
        pack = dpo_packer.create_empty_pack()

        sample1 = {
            "prompt_ids": torch.tensor([1, 2]),
            "chosen_response_only_ids": torch.tensor([3, 4]),
            "chosen_response_only_labels": torch.tensor([3, 4]),
            "rejected_response_only_ids": torch.tensor([5, 6]),
            "rejected_response_only_labels": torch.tensor([5, 6]),
        }
        dpo_packer.add_sample_to_pack(pack, sample1, next_doc_id=0)  # docs 0, 1, 2

        sample2 = {
            "prompt_ids": torch.tensor([7]),
            "chosen_response_only_ids": torch.tensor([8]),
            "chosen_response_only_labels": torch.tensor([8]),
            "rejected_response_only_ids": torch.tensor([9, 10]),
            "rejected_response_only_labels": torch.tensor([9, 10]),
        }
        dpo_packer.add_sample_to_pack(pack, sample2, next_doc_id=3)  # docs 3, 4, 5

        # Total tokens = 6 (sample1) + 4 (sample2) = 10
        result = dpo_packer.finalize_pack(
            pack, target_tokens_per_pack=12, next_doc_id=6
        )

        expected_tokens = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 999, 999])
        expected_labels = torch.tensor(
            [-100, -100, 3, 4, 5, 6, -100, 8, 9, 10, -100, -100]
        )
        expected_doc_ids = torch.tensor([0, 0, 1, 1, 2, 2, 3, 4, 5, 5, 6, 6])
        expected_chosen_mask = torch.tensor(
            [
                False,
                False,
                True,
                True,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
            ]
        )
        expected_rejected_mask = torch.tensor(
            [
                False,
                False,
                False,
                False,
                True,
                True,
                False,
                False,
                True,
                True,
                False,
                False,
            ]
        )

        torch.testing.assert_close(result["tokens"], expected_tokens)
        torch.testing.assert_close(result["labels"], expected_labels)
        torch.testing.assert_close(result["document_ids"], expected_doc_ids)
        torch.testing.assert_close(result["chosen_response_mask"], expected_chosen_mask)
        torch.testing.assert_close(
            result["rejected_response_mask"], expected_rejected_mask
        )

    def test_dpo_specialized_mask(self, device):
        """
        Verify the correctness of the DPO attention mask by manually constructing
        the expected mask for a batch containing multiple, different samples.
        """
        dpo_packer = DPOPacker(padding_idx=0)

        # Batch contains two packs of different token layouts.
        # Pack 1: Two DPO samples. P(1), C(2), R(1) | P(1), C(1), R(1)
        # Doc IDs: P_A=0, C_A=1, R_A=2 | P_B=3, C_B=4, R_B=5
        doc_ids_1 = torch.tensor([0, 1, 1, 2, 3, 4, 5], device=device)

        # Pack 2: One DPO sample, then padding. P(2), C(2), R(1), Pad(2)
        # Doc IDs: P_A=0, C_A=1, R_A=2 | Padding=3
        doc_ids_2 = torch.tensor([0, 0, 1, 1, 2, 3, 3], device=device)
        batch_doc_ids = torch.stack([doc_ids_1, doc_ids_2])

        # --- Manually create the expected mask for Pack 1 ---
        mask1 = torch.tensor(
            [
                # k_idx -> P  C  C  R  P  C  R (k_idx)
                [1, 0, 0, 0, 0, 0, 0],  # q=0 (P_A) can see self
                [1, 1, 0, 0, 0, 0, 0],  # q=1 (C_A) can see P_A and self (causal)
                [1, 1, 1, 0, 0, 0, 0],  # q=2 (C_A) can see P_A and C_A (causal)
                [1, 0, 0, 1, 0, 0, 0],  # q=3 (R_A) can see P_A and self
                [0, 0, 0, 0, 1, 0, 0],  # q=4 (P_B) can see self
                [0, 0, 0, 0, 1, 1, 0],  # q=5 (C_B) can see P_B and self
                [0, 0, 0, 0, 1, 0, 1],  # q=6 (R_B) can see P_B and self
            ],
            dtype=torch.bool,
            device=device,
        )

        # --- Manually create the expected mask for Pack 2 ---
        mask2 = torch.tensor(
            [
                # q_idx,  P  P  C  C  R  Pad Pad(k_idx)
                [1, 0, 0, 0, 0, 0, 0],  # q=0 (P_A)
                [1, 1, 0, 0, 0, 0, 0],  # q=1 (P_A)
                [1, 1, 1, 0, 0, 0, 0],  # q=2 (C_A)
                [1, 1, 1, 1, 0, 0, 0],  # q=3 (C_A)
                [1, 1, 0, 0, 1, 0, 0],  # q=4 (R_A)
                [0, 0, 0, 0, 0, 1, 0],  # q=5 (Pad)
                [0, 0, 0, 0, 0, 1, 1],  # q=6 (Pad)
            ],
            dtype=torch.bool,
            device=device,
        )

        expected_mask = torch.stack([mask1, mask2])

        actual_mask = create_dense_mask_from_mask_mod(dpo_packer, batch_doc_ids)
        torch.testing.assert_close(actual_mask, expected_mask)

    def test_dpo_packing_workflow_two_packs(self):
        """Test complete DPO workflow that creates exactly 2 packs with multiple samples"""
        samples = [
            {  # Sample 0: total 4 tokens (1+1+2)
                "prompt_ids": torch.tensor([1]),
                "chosen_response_only_ids": torch.tensor([2]),
                "chosen_response_only_labels": torch.tensor([2]),
                "rejected_response_only_ids": torch.tensor([3, 4]),
                "rejected_response_only_labels": torch.tensor([3, 4]),
            },
            {  # Sample 1: total 5 tokens (2+1+2)
                "prompt_ids": torch.tensor([5, 6]),
                "chosen_response_only_ids": torch.tensor([7]),
                "chosen_response_only_labels": torch.tensor([7]),
                "rejected_response_only_ids": torch.tensor([8, 9]),
                "rejected_response_only_labels": torch.tensor([8, 9]),
            },
            {  # Sample 2: total 6 tokens (2+2+2)
                "prompt_ids": torch.tensor([10, 11]),
                "chosen_response_only_ids": torch.tensor([12, 13]),
                "chosen_response_only_labels": torch.tensor([12, 13]),
                "rejected_response_only_ids": torch.tensor([14, 15]),
                "rejected_response_only_labels": torch.tensor([14, 15]),
            },
        ]

        dataset = DummyDPODataset(samples)
        dpo_packer = DPOPacker(padding_idx=999, ignore_idx=-100)
        packed_dataset = IterablePackedDataset(
            dataset=dataset, packer=dpo_packer, target_tokens_per_pack=10
        )

        packs = list(packed_dataset)
        assert len(packs) == 2  # Pack1: samples 0+1 (4+5=9), Pack2: sample 2 (6)

        # Pack 1: samples 0+1 (9 tokens) + padding (1)
        pack1 = packs[0]
        assert pack1["tokens"].shape == (10,)
        assert "chosen_response_mask" in pack1
        assert "rejected_response_mask" in pack1
        non_padding_1 = (pack1["tokens"] != 999).sum()
        assert non_padding_1 == 9

        # Pack 2: sample 2 (6 tokens) + padding (4)
        pack2 = packs[1]
        non_padding_2 = (pack2["tokens"] != 999).sum()
        assert non_padding_2 == 6

        # Verify masks are mutually exclusive
        chosen_and_rejected_1 = (
            pack1["chosen_response_mask"] & pack1["rejected_response_mask"]
        )
        chosen_and_rejected_2 = (
            pack2["chosen_response_mask"] & pack2["rejected_response_mask"]
        )
        assert not chosen_and_rejected_1.any()
        assert not chosen_and_rejected_2.any()


@pytest.mark.skipif(not _SUPPORTS_FLEX_ATTENTION, reason="Flex attention not supported")
class TestCollatedPacked:
    """Test collate_packed function"""

    def test_collate_empty_batch(self):
        """Test collating an empty batch"""
        result = collate_packed(batch=[], mask_fn=lambda doc_ids, device: None, device="cpu")
        assert result == {}

    def test_collate_basic_batch(self):
        """Test basic collation functionality"""
        # Create mock packed samples
        batch = [
            {
                "tokens": torch.tensor([1, 2, 3]),
                "labels": torch.tensor([4, 5, 6]),
                "document_ids": torch.tensor([0, 0, 1]),
                "input_pos": torch.tensor([0, 1, 0]),
                "metrics": [
                    type('Metric', (), {'metric_name': 'test', 'value': 1.0})(),
                    type('Metric', (), {'metric_name': 'test2', 'value': 2.0})()
                ]
            },
            {
                "tokens": torch.tensor([7, 8]),
                "labels": torch.tensor([9, 10]),
                "document_ids": torch.tensor([2, 2]),
                "input_pos": torch.tensor([0, 1]),
                "metrics": [
                    type('Metric', (), {'metric_name': 'test3', 'value': 3.0})()
                ]
            }
        ]
        
        # Mock mask function
        def mock_mask_fn(doc_ids, device):
            batch_size, seq_len = doc_ids.shape
            return torch.ones(batch_size, seq_len, seq_len, dtype=torch.bool)
        
        result = collate_packed(batch, mock_mask_fn, "cpu")
        
        # Check tensor stacking
        expected_tokens = torch.stack([torch.tensor([1, 2, 3]), torch.tensor([7, 8])])
        expected_labels = torch.stack([torch.tensor([4, 5, 6]), torch.tensor([9, 10])])
        expected_doc_ids = torch.stack([torch.tensor([0, 0, 1]), torch.tensor([2, 2])])
        
        torch.testing.assert_close(result["tokens"], expected_tokens)
        torch.testing.assert_close(result["labels"], expected_labels)
        torch.testing.assert_close(result["document_ids"], expected_doc_ids)
        
        # Check metrics flattening
        assert len(result["metrics"]) == 3  # All metrics from both samples
        
        # Check mask creation
        assert "mask" in result
        assert result["mask"].shape == (2, 3, 3)  # batch_size=2, seq_len=3

    def test_collate_different_keys_error(self):
        """Test that different keys across samples raises ValueError"""
        batch = [
            {"tokens": torch.tensor([1, 2]), "labels": torch.tensor([3, 4])},
            {"tokens": torch.tensor([5, 6]), "other_key": torch.tensor([7, 8])}
        ]
        
        def mock_mask_fn(doc_ids, device):
            return torch.ones(1, 1, 1)
        
        with pytest.raises(ValueError, match="All samples must have the same keys"):
            collate_packed(batch, mock_mask_fn, "cpu")

    def test_collate_mixed_tensor_non_tensor(self):
        """Test collation with mixed tensor and non-tensor data"""
        batch = [
            {
                "tokens": torch.tensor([1, 2]),
                "document_ids": torch.tensor([0, 0]),
                "text_data": "sample1",
                "metrics": ["DummyMetric1"]
            },
            {
                "tokens": torch.tensor([3, 4]),
                "document_ids": torch.tensor([1, 1]),
                "text_data": "sample2",
                "metrics": ["DummyMetric2"]
            }
        ]
        
        def mock_mask_fn(doc_ids, device):
            return torch.ones(2, 2, 2)
        
        result = collate_packed(batch, mock_mask_fn, "cpu")
        
        # Tensors should be stacked
        expected_tokens = torch.stack([torch.tensor([1, 2]), torch.tensor([3, 4])])
        torch.testing.assert_close(result["tokens"], expected_tokens)
        
        # Non-tensors should be kept as lists
        assert result["text_data"] == ["sample1", "sample2"]
        
        # Metrics should be flattened
        assert result["metrics"] == ["DummyMetric1", "DummyMetric2"]


@pytest.mark.skipif(not _SUPPORTS_FLEX_ATTENTION, reason="Flex attention not supported")
class TestIterablePackedDataset:
    """Test IterablePackedDataset functionality - buffer efficiency, checkpointing, edge cases"""

    def test_buffer_efficiency(self):
        """Test buffer improves packing efficiency"""
        # Test case where buffer helps vs hurts - order matters for first-fit
        sample_sizes = [3, 4, 1, 2]  # Total 10 tokens
        target_tokens = 6

        # With large buffer: can see all samples and pick best fit [3,1,2], [4]
        dataset1 = DummyTextDataset(sample_sizes)
        packer1 = TextPacker(padding_idx=999, ignore_idx=-100)
        packed_dataset1 = IterablePackedDataset(
            dataset=dataset1,
            packer=packer1,
            target_tokens_per_pack=target_tokens,
            buffer_size=10,
        )
        packs_buffered = list(packed_dataset1)

        # With small buffer: greedy first-fit [3], [4,1], [2]
        dataset2 = DummyTextDataset(sample_sizes)
        packer2 = TextPacker(padding_idx=999, ignore_idx=-100)
        packed_dataset2 = IterablePackedDataset(
            dataset=dataset2,
            packer=packer2,
            target_tokens_per_pack=target_tokens,
            buffer_size=1,
        )
        packs_unbuffered = list(packed_dataset2)

        # Buffer should create fewer packs (more efficient)
        assert len(packs_buffered) < len(packs_unbuffered)
        assert len(packs_buffered) == 2  # [3,1,2], [4]
        assert len(packs_unbuffered) == 3  # [3], [4,1], [2]

        # Verify both preserve all tokens
        total_buffered = sum((p["labels"] != -100).sum().item() for p in packs_buffered)
        total_unbuffered = sum(
            (p["labels"] != -100).sum().item() for p in packs_unbuffered
        )
        assert total_buffered == total_unbuffered == sum(sample_sizes)

    def test_oversized_sample_dropping(self):
        """Test that oversized samples are dropped"""
        sample_sizes = [3, 10, 2, 8, 1]  # 10 and 8 are oversized for target=6
        target_tokens = 5

        dataset = DummyTextDataset(sample_sizes)
        packer = TextPacker(padding_idx=999, ignore_idx=-100)
        packed_dataset = IterablePackedDataset(
            dataset=dataset, packer=packer, target_tokens_per_pack=target_tokens
        )

        packs = list(packed_dataset)

        # Only samples 3, 2, 1 should be packed (oversized 10, 8 dropped)
        total_packed_tokens = sum((p["labels"] != -100).sum().item() for p in packs)
        expected_tokens = 3 + 2 + 1  # Only non-oversized samples
        assert total_packed_tokens == expected_tokens

        # Should create 2 packs: [3, 2], [1]
        assert len(packs) == 2

    def test_checkpoint_and_resume(self):
        """Test checkpointing and resumption functionality using StatefulDataLoader"""
        sample_sizes = [3, 2, 5, 4, 1, 6]  # Total 21 tokens
        target_tokens_per_pack = 6
        batch_size = 2

        # Setup dataset factory
        def create_loader_and_aggregator():
            dataset = StatefulDummyTextDataset(sample_sizes)
            packer = TextPacker(padding_idx=999, ignore_idx=-100)
            packed_dataset = IterablePackedDataset(
                dataset=dataset,
                packer=packer,
                target_tokens_per_pack=target_tokens_per_pack,
                buffer_size=0,  # No buffer for deterministic checkpointing
            )

            collate_fn = partial(
                collate_packed, mask_fn=packer.create_block_mask, device="cpu"
            )

            loader = StatefulDataLoader(
                packed_dataset, batch_size=batch_size, collate_fn=collate_fn
            )
            aggregator = MetricsAggregator()
            return loader, aggregator

        loader1, aggregator1 = create_loader_and_aggregator()
        loader2, aggregator2 = create_loader_and_aggregator()

        steps_before_checkpoint = 3
        steps_after_checkpoint = 3

        # Generate checkpoint and resume
        result = generate_ckpt(
            loader1,
            aggregator1,
            steps_before_checkpoint=steps_before_checkpoint,
            steps_after_checkpoint=steps_after_checkpoint,
            resume_dataloader=loader2,
            resume_aggregator=aggregator2,
        )

        # Verify that checkpointing and resumption are identical
        assert len(result["post_checkpoint_batches"]) == steps_after_checkpoint
        assert len(result["resumed_batches"]) == steps_after_checkpoint

        for orig_batch, resumed_batch in zip(
            result["post_checkpoint_batches"], result["resumed_batches"]
        ):
            assert orig_batch.keys() == resumed_batch.keys()
            for key in orig_batch:
                if isinstance(orig_batch[key], torch.Tensor):
                    torch.testing.assert_close(
                        orig_batch[key],
                        resumed_batch[key],
                        msg=f"Mismatch in batch key: {key}",
                    )
                else:
                    assert (
                        orig_batch[key] == resumed_batch[key]
                    ), f"Mismatch in batch key: {key}"

        assert (
            result["final_metrics"] == result["resumed_metrics"]
        ), "Final metrics should match"

    def test_multiple_iterations_same_dataset(self):
        """Test that multiple iterations over same packed dataset work correctly"""
        sample_sizes = [2, 3, 1]
        dataset = DummyTextDataset(sample_sizes)
        packer = TextPacker(padding_idx=999, ignore_idx=-100)
        packed_dataset = IterablePackedDataset(
            dataset=dataset, packer=packer, target_tokens_per_pack=4
        )
        # First iteration
        packs1 = list(packed_dataset)
        # Second iteration should produce same result
        packs2 = list(packed_dataset)

        assert len(packs1) == len(packs2)
        for p1, p2 in zip(packs1, packs2):
            torch.testing.assert_close(p1["tokens"], p2["tokens"])
            torch.testing.assert_close(p1["document_ids"], p2["document_ids"])

    @pytest.mark.parametrize(
        "sample_sizes,target_tokens,buffer_size,expected_packs,scenario",
        [
            ([3, 2, 4], 8, 10, 2, "basic_packing"),  # Pack1: [3,2]+pad, Pack2: [4]+pad
            ([4, 3], 8, 10, 1, "partial_final_pack"),  # Pack1: [4,3]+pad
            ([], 8, 10, 0, "empty_dataset"),
            ([5], 10, 10, 1, "single_sample"),
            ([5, 5, 5], 5, 10, 3, "exact_fit"),
            ([2, 3, 1], 5, 1, 2, "small_target_and_buffer"),  # Pack1: [2,3], Pack2: [1]
        ],
    )
    def test_scenarios(
        self, sample_sizes, target_tokens, buffer_size, expected_packs, scenario
    ):
        """Parametrized edge case testing"""
        dataset = DummyTextDataset(sample_sizes)
        packer = TextPacker(padding_idx=999, ignore_idx=-100)
        packed_dataset = IterablePackedDataset(
            dataset=dataset,
            packer=packer,
            target_tokens_per_pack=target_tokens,
            buffer_size=buffer_size,
        )

        packs = list(packed_dataset)
        assert len(packs) == expected_packs, f"Failed scenario: {scenario}"

        # Verify output format consistency for all scenarios
        for pack in packs:
            assert pack["tokens"].shape == (target_tokens,)
            assert pack["labels"].shape == (target_tokens,)
            assert pack["document_ids"].shape == (target_tokens,)
            assert pack["input_pos"].shape == (target_tokens,)

        # Verify no token loss
        if sample_sizes:  # Skip for empty dataset
            total_packed = sum((p["labels"] != -100).sum().item() for p in packs)
            assert total_packed == sum(sample_sizes)
