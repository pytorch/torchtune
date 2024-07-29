# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch


class TileCrop(torch.nn.Module):
    """
    Divides a tensor into equally sized tiles. The tensor should be divisible by tile_size.
    """

    def __init__(self, tile_size: int):
        super().__init__()
        self.tile_size = tile_size

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image (torch.Tensor): Input image to crop into tiles of size tile_size.

        Returns:
            torch.Tensor: Tensor of shape [num_tiles, channel_size, tile_size, tile_size]

        Examples:
            >>> image = torch.rand(3, 200, 300)
            >>> tile_crop = TileCrop()
            >>> tiles = tile_crop(image, tile_size=50)
            >>> tiles.shape # 4x6 = 24 tiles
            torch.Size([24, 3, 50, 50])

            >>> image = torch.rand(3, 400, 600)
            >>> tiles = TileCrop(image, tile_size=200)
            >>> tiles.shape # 2x3 = 6 tiles
            torch.Size([6, 3, 200, 200])
        """
        channel_size, height, width = image.shape
        # assert sizes are divisible
        assert (
            height % self.tile_size == 0 and width % self.tile_size == 0
        ), f"Image size {height}x{width} is not divisible by tile size {self.tile_size}"

        # Reshape to split height and width into tile_size blocks
        tiles_height = height // self.tile_size
        tiles_width = width // self.tile_size

        reshaped = image.view(
            channel_size, tiles_height, self.tile_size, tiles_width, self.tile_size
        )

        # Transpose to bring tiles together
        # We want [tiles_height, tiles_width, channel_size, tile_size, tile_size]
        transposed = reshaped.permute(1, 3, 0, 2, 4)

        # Flatten the tiles
        tiles = transposed.contiguous().view(
            tiles_height * tiles_width, channel_size, self.tile_size, self.tile_size
        )
        return tiles
