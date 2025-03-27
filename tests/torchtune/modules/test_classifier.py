# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from tests.test_utils import fixed_init_model
from torchtune.modules import classifier_model
from torchtune.training.seed import set_seed

NUM_LAYERS = 4
NUM_HEADS = 16
NUM_KV_HEADS = 8
VOCAB_SIZE = 32000
MAX_SEQ_LEN = 2048
INTERMEDIATE_DIM = 512


@pytest.fixture(autouse=True, scope="class")
def random():
    set_seed(16)


class TestClassifierModelBuilder:
    @pytest.mark.parametrize(
        "base_model_path",
        [
            "torchtune.models.llama3_2.llama3_2",
            "torchtune.models.llama3_1.llama3_1",
            "torchtune.models.qwen2.qwen2",
            "torchtune.models.phi3.phi3",
            "torchtune.models.mistral.mistral",
        ],
    )
    @pytest.mark.parametrize(
        "bsz, embed_dim, seq_len, n_classes, expected",
        [
            (2, 64, 64, 2, 22.6879),
            (1, 256, 256, 1, 110.2561),
        ],
    )
    def test_forward_classifier_model(
        self,
        base_model_path: str,
        bsz: int,
        embed_dim: int,
        seq_len: int,
        n_classes: int,
        expected: float,
    ):
        inputs = torch.randint(low=0, high=VOCAB_SIZE, size=(bsz, seq_len))
        model = classifier_model(
            num_classes=n_classes,
            base_model_path=base_model_path,
            vocab_size=VOCAB_SIZE,
            num_layers=n_classes,
            num_heads=NUM_HEADS,
            num_kv_heads=NUM_KV_HEADS,
            embed_dim=embed_dim,
            intermediate_dim=INTERMEDIATE_DIM,
            max_seq_len=MAX_SEQ_LEN,
        )
        fixed_init_model(model)
        actual = model(inputs)
        expected = torch.tensor(expected)
        assert actual.shape == (bsz, seq_len, n_classes)
        torch.testing.assert_close(actual.mean(), expected, atol=1e-4, rtol=1e-4)

    def test_classifier_builder_mm(self):
        with torch.device("meta"):
            model = classifier_model(
                num_classes=1,
                base_model_path="torchtune.models.llama3_2_vision.llama3_2_vision_11b",
            )
        assert model.decoder.output.weight.shape == (1, 4096)
