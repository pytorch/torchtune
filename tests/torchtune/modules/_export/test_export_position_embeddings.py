# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile
import unittest

import torch
from torch._inductor.package import load_package, package_aoti
from torch.testing import assert_close
from torchtune.models.clip import (
    TiledTokenPositionalEmbedding as TuneTiledTokenPositionalEmbedding,
    TilePositionalEmbedding as TuneTilePositionalEmbedding,
)
from torchtune.modules._export._position_embeddings import (
    replace_tile_positional_embedding,
    replace_tiled_token_positional_embedding,
    TiledTokenPositionalEmbedding,
    TilePositionalEmbedding,
)
from torchtune.utils import torch_version_ge


class TilePositionalEmbeddingTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.tpe = TilePositionalEmbedding(4, 1280)
        self.ref_tpe = TuneTilePositionalEmbedding(4, 1280)
        self.x = torch.randn(1, 4, 1600, 1280)
        self.aspect_ratio = torch.tensor([[1, 1]])
        num_tiles_dim = torch.export.Dim("num_tiles", min=1, max=4)
        num_tokens = torch.export.Dim("num_tokens", min=1, max=1600)

        self.dynamic_shape = {
            0: 1,  # batch
            1: num_tiles_dim,  # num tiles
            2: num_tokens,  # num tokens
            3: 1280,  # embedding dim
        }

    def test_tile_positional_embedding_smoke(self):
        y = self.tpe(self.x, self.aspect_ratio)
        ref_y = self.ref_tpe(self.x, self.aspect_ratio)

        self.assertTrue(torch.allclose(y, ref_y))

    @unittest.skipUnless(
        torch_version_ge("2.6.0.dev20241117"), reason="Need recent fixes for export"
    )
    def test_tile_positional_embedding_export(self):
        tpe_ep = torch.export.export(
            self.tpe,
            (self.x, self.aspect_ratio),
            dynamic_shapes=(
                self.dynamic_shape,
                None,
            ),  # assuming aspect ratio is static
            strict=True,
        )

        y = tpe_ep.module()(self.x, self.aspect_ratio)
        ref_y = self.ref_tpe(self.x, self.aspect_ratio)

        self.assertTrue(torch.allclose(y, ref_y))

    @unittest.skipUnless(
        torch_version_ge("2.6.0.dev20241117"), reason="Need recent fixes for aoti"
    )
    def test_tile_positional_embedding_aoti(self):
        so = torch._export.aot_compile(
            self.tpe,
            args=(self.x, self.aspect_ratio),
            options={"aot_inductor.package": True},
            dynamic_shapes=(
                self.dynamic_shape,
                None,
            ),  # assuming aspect ratio is static
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = package_aoti(os.path.join(tmpdir, "tpe.pt2"), so)
            tpe_aoti = load_package(path)

            y = tpe_aoti(self.x, self.aspect_ratio)
            ref_y = self.ref_tpe(self.x, self.aspect_ratio)

            self.assertTrue(torch.allclose(y, ref_y))

    def test_replace_tile_positional_embedding(self):
        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.tpe = TuneTilePositionalEmbedding(4, 1280)

            def forward(self, x, aspect_ratio):
                return self.tpe(x, aspect_ratio)

        m = Module()
        m = replace_tile_positional_embedding(m)
        self.assertTrue(isinstance(m.tpe, TilePositionalEmbedding))


class TiledTokenPositionalEmbeddingTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.tpe = TiledTokenPositionalEmbedding(4, 1280, 40, 1)
        self.ref_tpe = TuneTiledTokenPositionalEmbedding(4, 1280, 40, 1)
        self.tpe.load_state_dict(self.ref_tpe.state_dict())
        self.x = torch.randn(1, 4, 1601, 1280)
        self.aspect_ratio = torch.tensor([[1, 2]])
        num_tiles_dim = torch.export.Dim("num_tiles", min=1, max=4)

        self.dynamic_shape = {
            0: 1,  # batch
            1: num_tiles_dim,  # num tiles
            2: 1601,  # num tokens
            3: 1280,  # embedding dim
        }

    def test_tiled_token_positional_embedding_smoke(self):
        y = self.tpe(self.x, self.aspect_ratio)
        ref_y = self.ref_tpe(self.x, self.aspect_ratio)

        assert_close(y, ref_y)

    @unittest.skipUnless(
        torch_version_ge("2.6.0.dev20241117"), reason="Need recent fixes for export"
    )
    def test_tiled_token_positional_embedding_export(self):
        tpe_ep = torch.export.export(
            self.tpe,
            (self.x, self.aspect_ratio),
            dynamic_shapes=(
                self.dynamic_shape,
                None,
            ),  # assuming aspect ratio is static
            strict=True,
        )

        y = tpe_ep.module()(self.x, self.aspect_ratio)
        ref_y = self.ref_tpe(self.x, self.aspect_ratio)

        assert_close(y, ref_y)

    @unittest.skipUnless(
        torch_version_ge("2.6.0.dev20241117"), reason="Need recent fixes for aoti"
    )
    def test_tiled_token_positional_embedding_aoti(self):
        tpe_ep = torch.export.export(
            self.tpe,
            (self.x, self.aspect_ratio),
            dynamic_shapes=(
                self.dynamic_shape,
                None,
            ),  # assuming aspect ratio is static
            strict=True,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = torch._inductor.aoti_compile_and_package(
                tpe_ep,
                package_path=os.path.join(tmpdir, "tpe.pt2"),
            )
            tpe_aoti = load_package(path)

            y = tpe_aoti(self.x, self.aspect_ratio)
            ref_y = self.ref_tpe(self.x, self.aspect_ratio)

            assert_close(y, ref_y)

    def test_replace_tiled_token_positional_embedding(self):
        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.tpe = TuneTiledTokenPositionalEmbedding(4, 1280, 40, 1)

            def forward(self, x, aspect_ratio):
                return self.tpe(x, aspect_ratio)

        m = Module()
        m = replace_tiled_token_positional_embedding(m)
        self.assertTrue(isinstance(m.tpe, TiledTokenPositionalEmbedding))
