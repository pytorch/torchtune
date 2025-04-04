# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchtune.utils._tensor_utils import chunk


class TestChunkTensorUtil:
    def test_chunk_simple(self):
        """Tests the simplest usage of chunk function."""
        tensor = torch.rand((10, 20))
        chunks = chunk(tensor, 5)

        assert isinstance(chunks, tuple)
        assert len(chunks) == 5
        for el in chunks:
            assert isinstance(el, torch.Tensor)
            assert el.size() == torch.Size((2, 20))

    def test_chunk_dim(self):
        """Tests the non-default dim."""
        tensor = torch.rand((10, 50))
        chunks = chunk(tensor, 8, dim=1)
        chunk_lengths = [x.size(1) for x in chunks]

        assert chunk_lengths == [6, 6, 6, 6, 6, 6, 6, 8]

    def test_chunk_dim_last(self):
        """Tests the -1 dim."""
        tensor = torch.rand((10, 20, 50))
        chunks = chunk(tensor, 8, dim=-1)
        chunk_lengths = [x.size(2) for x in chunks]

        assert chunk_lengths == [6, 6, 6, 6, 6, 6, 6, 8]

    def test_chunk_torch_cornercase(self):
        """Tests the main reason to create the func - not exact chunks amount in torch."""
        tensor = torch.rand((49, 1))
        chunks = chunk(tensor, 8)
        chunk_lengths = [x.size(0) for x in chunks]

        assert chunk_lengths == [6, 6, 6, 6, 6, 6, 6, 7]
