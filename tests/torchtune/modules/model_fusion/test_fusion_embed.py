# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch
from tests.test_utils import assert_expected, fixed_init_model
from torchtune.modules.model_fusion import FusionEmbedding
from torchtune.utils.seed import set_seed


@pytest.fixture(autouse=True)
def random():
    set_seed(1)


class TestFusionEmbedding:
    """
    Class for testing our FusionEmbedding.
    """

    @pytest.fixture
    def dim(self) -> int:
        return 2

    @pytest.fixture
    def vocab_size(self) -> int:
        return 10

    @pytest.fixture
    def additional_tokens(self) -> int:
        return 5

    @pytest.fixture
    def embed(self, dim, vocab_size, additional_tokens) -> FusionEmbedding:
        embeds = FusionEmbedding(
            vocab_size=vocab_size, additional_tokens=additional_tokens, embed_dim=dim
        )
        fixed_init_model(embeds.embedding, min_val=0, max_val=0.5)
        fixed_init_model(embeds.fusion_embedding, min_val=0.51, max_val=1)
        return embeds

    @torch.no_grad()
    def test_forward(self, embed, vocab_size, additional_tokens, dim):
        """
        Test that the forward pass of the FusionEmbedding works as expected.
        """
        tokens = torch.randint(0, vocab_size + additional_tokens, (2, 10))
        out = embed(tokens)

        assert out.shape == (2, 10, dim)
        assert_expected(out.mean(), torch.tensor(0.3409), atol=1e-3, rtol=1e-3)

        # Only new tokens, embeddings should be > 0.5
        tokens = torch.randint(vocab_size, vocab_size + additional_tokens, (2, 10))
        out = embed(tokens)

        assert out.min() > 0.5

        # Only old tokens, embeddings should be < 0.5
        tokens = torch.randint(0, vocab_size, (2, 10))
        out = embed(tokens)

        assert out.max() < 0.5

    def test_fusion_params(self, embed):
        """
        Test that the currect fusion params are returned.
        """
        fusion_params = set(embed.fusion_params())

        assert fusion_params == {"fusion_embedding.weight"}
