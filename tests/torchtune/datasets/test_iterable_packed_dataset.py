# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from functools import partial
from itertools import islice
from pathlib import Path
from typing import Any

import pytest
import torch
from torchdata.stateful_dataloader import StatefulDataLoader

from torchtune.data._collate import collate_packed
from torchtune.data.metrics import MetricsAggregator
from torchtune.datasets._iterable_packed import (
    DPOPacker,
    IterablePackedDataset,
    Packer,
    TextPacker,
)
from torchtune.datasets._sft import HfIterableDataset
from torchtune.utils._import_guard import _SUPPORTS_FLEX_ATTENTION

from .test_iterable_utils import generate_ckpt


@pytest.fixture
def device():
    return "cuda"


# --- Test Utilities ---


def create_test_json_file(path: Path, samples: list[dict[str, list[int]]]) -> None:
    """Creates a dummy JSON test data file."""
    with open(path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")


class ToTensorTransform:
    """Converts lists in a sample to tensors, as expected by the packer."""

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        output = {}
        for k, v in sample.items():
            if isinstance(v, list):
                output[k] = torch.tensor(v, dtype=torch.long)
            else:
                output[k] = v
        # TextPacker expects "tokens" and "labels".
        if "tokens" in output and "labels" not in output:
            output["labels"] = output["tokens"].clone()
        return output


@pytest.fixture
def dataset_factory(tmp_path):
    """Factory for creating HfIterableDataset instances for testing."""

    def _create_dataset(
        data: list[dict[str, list[int]]],
        **kwargs,
    ) -> HfIterableDataset:
        file_path = tmp_path / "data.json"
        create_test_json_file(file_path, data)
        return HfIterableDataset(
            path="json",
            data_files=str(file_path),
            split="train",
            shuffle_buffer_size=0,
            output_transform=ToTensorTransform(),
            num_shards_per_rank=1,
            **kwargs,
        )

    return _create_dataset


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

    def test_text_packing_workflow_two_packs(self, dataset_factory):
        """Test complete text workflow that creates exactly 2 packs with multiple samples"""
        # Design: Pack1=[3,2], Pack2=[4] to create 2 packs
        samples = [
            {"tokens": [0] * 3},
            {"tokens": [1] * 2},
            {"tokens": [2] * 4},
        ]
        target_tokens = 6

        dataset = dataset_factory(samples)
        text_packer = TextPacker(padding_idx=999, ignore_idx=-100)
        packed_dataset = IterablePackedDataset(
            dataset=dataset,
            packer=text_packer,
            target_tokens_per_pack=target_tokens,
            buffer_size=1,
        )

        packs = list(islice(packed_dataset, 2))
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

    def test_dpo_packing_workflow_two_packs(self, dataset_factory):
        """Test complete DPO workflow that creates exactly 2 packs with multiple samples"""
        samples = [
            {  # Sample 0: total 4 tokens (1+1+2)
                "prompt_ids": [1],
                "chosen_response_only_ids": [2],
                "chosen_response_only_labels": [2],
                "rejected_response_only_ids": [3, 4],
                "rejected_response_only_labels": [3, 4],
            },
            {  # Sample 1: total 5 tokens (2+1+2)
                "prompt_ids": [5, 6],
                "chosen_response_only_ids": [7],
                "chosen_response_only_labels": [7],
                "rejected_response_only_ids": [8, 9],
                "rejected_response_only_labels": [8, 9],
            },
            {  # Sample 2: total 6 tokens (2+2+2)
                "prompt_ids": [10, 11],
                "chosen_response_only_ids": [12, 13],
                "chosen_response_only_labels": [12, 13],
                "rejected_response_only_ids": [14, 15],
                "rejected_response_only_labels": [14, 15],
            },
        ]

        dataset = dataset_factory(samples)
        dpo_packer = DPOPacker(padding_idx=999, ignore_idx=-100)
        packed_dataset = IterablePackedDataset(
            dataset=dataset, packer=dpo_packer, target_tokens_per_pack=10, buffer_size=1
        )

        packs = list(islice(packed_dataset, 2))
        assert (
            len(packs) == 2
        )  # Pack1: samples 0+1 (4+5=9), Pack2: sample 2+0 from cycle (6+4=10)

        # Pack 1: samples 0+1 (9 tokens) + padding (1)
        pack1 = packs[0]
        assert pack1["tokens"].shape == (10,)
        assert "chosen_response_mask" in pack1
        assert "rejected_response_mask" in pack1
        non_padding_1 = (pack1["tokens"] != 999).sum()
        assert non_padding_1 == 9

        # Pack 2: sample 2 (6 tokens) + sample 0 from cycle (4 tokens) = 10 tokens (no padding)
        pack2 = packs[1]
        non_padding_2 = (pack2["tokens"] != 999).sum()
        assert non_padding_2 == 10

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
        result = collate_packed(
            batch=[], mask_fn=lambda doc_ids, device: None, device="cpu"
        )
        assert result == {}

    def test_collate_basic_batch(self):
        """Test basic collation functionality"""
        # Create mock packed samples with same tensor sizes (as expected from IterablePackedDataset)
        batch = [
            {
                "tokens": torch.tensor([1, 2, 3]),
                "labels": torch.tensor([4, 5, 6]),
                "document_ids": torch.tensor([0, 0, 1]),
                "input_pos": torch.tensor([0, 1, 0]),
                "metrics": [
                    type("Metric", (), {"metric_name": "test", "value": 1.0})(),
                    type("Metric", (), {"metric_name": "test2", "value": 2.0})(),
                ],
            },
            {
                "tokens": torch.tensor([7, 8, 999]),  # Padded to same size
                "labels": torch.tensor([9, 10, -100]),  # Padded to same size
                "document_ids": torch.tensor([2, 2, 3]),  # Padded to same size
                "input_pos": torch.tensor([0, 1, 0]),  # Padded to same size
                "metrics": [
                    type("Metric", (), {"metric_name": "test3", "value": 3.0})()
                ],
            },
        ]

        # Mock mask function
        def mock_mask_fn(doc_ids, device):
            batch_size, seq_len = doc_ids.shape
            return torch.ones(batch_size, seq_len, seq_len, dtype=torch.bool)

        result = collate_packed(batch, mock_mask_fn, "cpu")

        # Check tensor stacking
        expected_tokens = torch.stack(
            [torch.tensor([1, 2, 3]), torch.tensor([7, 8, 999])]
        )
        expected_labels = torch.stack(
            [torch.tensor([4, 5, 6]), torch.tensor([9, 10, -100])]
        )
        expected_doc_ids = torch.stack(
            [torch.tensor([0, 0, 1]), torch.tensor([2, 2, 3])]
        )

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
            {"tokens": torch.tensor([5, 6]), "other_key": torch.tensor([7, 8])},
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
                "metrics": ["DummyMetric1"],
            },
            {
                "tokens": torch.tensor([3, 4]),
                "document_ids": torch.tensor([1, 1]),
                "text_data": "sample2",
                "metrics": ["DummyMetric2"],
            },
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
    def test_buffer(self, dataset_factory):
        """Test buffer behaves as expected, i.e. when next sentence doesn't fit, goes over
        the buffer and see if something fits"""
        samples = [
            {"tokens": [0] * 3},  # Sample 0: size 3
            {"tokens": [1] * 4},  # Sample 1: size 4
            {"tokens": [2] * 1},  # Sample 2: size 1
            {"tokens": [3] * 2},  # Sample 3: size 2
        ]
        target_tokens = 6

        # Test 1: Large buffer - can see all samples and optimize packing
        # Expected: [0,0,0,2,3,3] (sizes 3+1+2=6), [1,1,1,1,999,999] (size 4+2 padding)
        dataset1 = dataset_factory(samples)
        packer1 = TextPacker(padding_idx=999, ignore_idx=-100)
        packed_dataset1 = IterablePackedDataset(
            dataset=dataset1,
            packer=packer1,
            target_tokens_per_pack=target_tokens,
            buffer_size=10,
        )
        packs_buffered = list(islice(packed_dataset1, 2))
        assert len(packs_buffered) == 2

        # Pack 1: samples 0+2+3 (3+1+2=6) - perfect fit
        pack1 = packs_buffered[0]
        expected_tokens_1 = torch.tensor([0, 0, 0, 2, 3, 3])
        expected_doc_ids_1 = torch.tensor([0, 0, 0, 1, 2, 2])
        torch.testing.assert_close(pack1["tokens"], expected_tokens_1)
        torch.testing.assert_close(pack1["document_ids"], expected_doc_ids_1)

        # Pack 2: sample 1 (4) + sample 2 (1) + sample 2 again from cycle (1) = 6 tokens exactly
        pack2 = packs_buffered[1]
        expected_tokens_2 = torch.tensor([1, 1, 1, 1, 2, 2])
        expected_doc_ids_2 = torch.tensor([0, 0, 0, 0, 1, 2])
        torch.testing.assert_close(pack2["tokens"], expected_tokens_2)
        torch.testing.assert_close(pack2["document_ids"], expected_doc_ids_2)

        # Test 2: Small buffer - greedy first-fit packing with infinite dataset cycling
        # Expected: [0,0,0,999,999,999] (size 3+3 padding), [1,1,1,1,2,999] (size 4+1+1 padding),
        # [3,3,0,0,0,999] (size 2+3+1 padding)
        dataset2 = dataset_factory(samples)
        packer2 = TextPacker(padding_idx=999, ignore_idx=-100)
        packed_dataset2 = IterablePackedDataset(
            dataset=dataset2,
            packer=packer2,
            target_tokens_per_pack=target_tokens,
            buffer_size=1,
        )
        packs_unbuffered = list(islice(packed_dataset2, 3))
        assert len(packs_unbuffered) == 3

        # Pack 1: sample 0 (3) + padding (3)
        pack1_unbuf = packs_unbuffered[0]
        expected_tokens_1_unbuf = torch.tensor([0, 0, 0, 999, 999, 999])
        expected_doc_ids_1_unbuf = torch.tensor([0, 0, 0, 1, 1, 1])
        torch.testing.assert_close(pack1_unbuf["tokens"], expected_tokens_1_unbuf)
        torch.testing.assert_close(
            pack1_unbuf["document_ids"], expected_doc_ids_1_unbuf
        )

        # Pack 2: samples 1+2 (4+1=5) + padding (1)
        pack2_unbuf = packs_unbuffered[1]
        expected_tokens_2_unbuf = torch.tensor([1, 1, 1, 1, 2, 999])
        expected_doc_ids_2_unbuf = torch.tensor([0, 0, 0, 0, 1, 2])
        torch.testing.assert_close(pack2_unbuf["tokens"], expected_tokens_2_unbuf)
        torch.testing.assert_close(
            pack2_unbuf["document_ids"], expected_doc_ids_2_unbuf
        )

        # Pack 3: sample 3 (2) + sample 0 from cycle (3) + padding (1)
        pack3_unbuf = packs_unbuffered[2]
        expected_tokens_3_unbuf = torch.tensor([3, 3, 0, 0, 0, 999])
        expected_doc_ids_3_unbuf = torch.tensor([0, 0, 1, 1, 1, 2])
        torch.testing.assert_close(pack3_unbuf["tokens"], expected_tokens_3_unbuf)
        torch.testing.assert_close(
            pack3_unbuf["document_ids"], expected_doc_ids_3_unbuf
        )

    def test_buffer_size_validation(self, dataset_factory):
        """Test that buffer_size < 1 raises ValueError"""
        samples = [{"tokens": [0] * 3}]
        dataset = dataset_factory(samples)

        with pytest.raises(ValueError, match="Buffer size must be greater than 0"):
            IterablePackedDataset(
                dataset=dataset,
                packer=TextPacker(padding_idx=999, ignore_idx=-100),
                target_tokens_per_pack=6,
                buffer_size=0,
            )

    def test_info_property(self, dataset_factory):
        """Test that the info property works correctly and includes child dataset info"""
        samples = [{"tokens": [0] * 3}]
        dataset = dataset_factory(samples)
        packer = TextPacker(padding_idx=999, ignore_idx=-100)

        # Create packed dataset with custom name
        packed_dataset = IterablePackedDataset(
            dataset=dataset,
            packer=packer,
            target_tokens_per_pack=6,
            buffer_size=1,
            dataset_name="TestPackedDataset",
        )

        # Check info property
        info = packed_dataset.info
        assert info.name == "TestPackedDataset"
        assert info.weight == 1.0
        assert len(info.children) == 1

        # Check child dataset info is included
        child_info = info.children[0]
        assert (
            child_info.name == "json_train"
        )  # From HfIterableDataset auto-generated name
        assert child_info.weight == 1.0

    def test_oversized_sample_dropping(self, dataset_factory):
        """Test that oversized samples are dropped"""
        samples = [
            {"tokens": [0] * 3},  # Kept
            {"tokens": [1] * 10},  # Dropped
            {"tokens": [2] * 2},  # Kept
            {"tokens": [3] * 8},  # Dropped
            {"tokens": [4] * 1},  # Kept
        ]
        target_tokens = 5

        dataset = dataset_factory(samples)
        packer = TextPacker(padding_idx=999, ignore_idx=-100)
        packed_dataset = IterablePackedDataset(
            dataset=dataset,
            packer=packer,
            target_tokens_per_pack=target_tokens,
            buffer_size=1,
        )

        packs = list(islice(packed_dataset, 5))

        # Only samples 3, 2, 1 should be packed (oversized 10, 8 dropped)
        # Verify that only samples 0, 2, 4 are packed (samples 1, 3 were dropped)
        all_tokens = torch.cat([pack["tokens"] for pack in packs])
        all_tokens = set(all_tokens.tolist())

        # Check that expected tokens are present and dropped tokens are not
        expected_tokens = {0, 2, 4, 999}
        assert (
            all_tokens == expected_tokens
        ), f"Expected {expected_tokens}, got {all_tokens}"

    def test_checkpoint_and_resume(self, dataset_factory):
        """Test checkpointing and resumption functionality using StatefulDataLoader

        Note: This test verifies that the checkpoint/resume mechanism works correctly,
        but does NOT expect identical batches after resumption. The IterablePackedDataset
        explicitly does NOT save buffer or partial pack state, so packing may differ
        after resumption due to different buffer fill patterns. This is by design.
        """
        samples = [
            {"tokens": [0] * 3},
            {"tokens": [1] * 2},
            {"tokens": [2] * 5},
            {"tokens": [3] * 4},
            {"tokens": [4] * 1},
            {"tokens": [5] * 6},
        ]
        target_tokens_per_pack = 6
        batch_size = 1

        # Setup dataset factory
        def create_loader_and_aggregator():
            dataset = dataset_factory(samples)
            packer = TextPacker(padding_idx=999, ignore_idx=-100)
            packed_dataset = IterablePackedDataset(
                dataset=dataset,
                packer=packer,
                target_tokens_per_pack=target_tokens_per_pack,
                buffer_size=1,  # Small buffer for predictable behavior
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

        steps_before_checkpoint = 2
        steps_after_checkpoint = 2

        # Generate checkpoint and resume
        result = generate_ckpt(
            loader1,
            aggregator1,
            steps_before_checkpoint=steps_before_checkpoint,
            steps_after_checkpoint=steps_after_checkpoint,
            resume_dataloader=loader2,
            resume_aggregator=aggregator2,
        )

        # Verify that checkpointing and resumption work
        assert len(result["post_checkpoint_batches"]) == steps_after_checkpoint
        assert len(result["resumed_batches"]) == steps_after_checkpoint
