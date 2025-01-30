# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from tests.test_utils import DummyTokenizer

from torchtune.datasets.multimodal import multimodal_chat_dataset


class TestMultimodalChatDataset:
    @pytest.fixture
    def tokenizer(self):
        return DummyTokenizer()

    def test_dataset_fails_with_packed(self, tokenizer):
        with pytest.raises(
            ValueError, match="Multimodal datasets don't support packing yet."
        ):
            multimodal_chat_dataset(
                model_transform=tokenizer, source="json", packed=True
            )
