# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn

# TODO (@Felipe): add load hooks + interpolation on positional encodings,
# so max_num_tiles can be variable and a trained model can be adapted to a
# new value.


class TokenPositionalEmbedding(nn.Module):
    """
    Token positional embedding for images, different for every token in an image.

    Notice that tile is different from patch (token). For details, please check the documentation of
    :class:`torchtune.modules.vision_transformer.VisionTransformer`.

    Args:
        embed_dim (int): The dimensionality of each token embedding.
        tile_size (int): The size of your image tiles, if the image was tile-cropped in advance. Otherwise,
            the size of the input image. In this case, the function will consider your image as a single tile.
        patch_size (int): The size of each patch. Used to divide the tiles into patches.
            E.g. for ``patch_size=40``, a tile of shape (400, 400) will have 10x10 grid of patches
            with shape (40, 40) each.
    """

    def __init__(self, embed_dim: int, tile_size: int, patch_size: int) -> None:
        super().__init__()
        patch_grid_size = tile_size // patch_size
        n_tokens_per_tile = patch_grid_size**2 + 1  # +1 for cls token
        scale = embed_dim**-0.5
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((n_tokens_per_tile, embed_dim))
        )

    def forward(self, x: torch.Tensor, *args: Tuple[Any]) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): torch.Tensor with shape (..., n_tokens_per_tile, embed_dim)
            *args (Tuple[Any]): Optional args.

        Returns:
            torch.Tensor: The input tensor with added positional embeddings.
        """
        return x + self.positional_embedding


class TiledTokenPositionalEmbedding(nn.Module):
    """

    Token positional embedding for tiled images. There are two positional embeddings in this module:

    * local_token_positional_embedding: same for every tile, different for every token. Equivalent \
        to :class:`torchtune.models.clip._position_embeddings.TokenPositionalEmbedding`, but gated.
    * global_token_positional_embedding: different for every tile, different for every token.

    Notice that tile is different from patch (token). For details, please check the documentation of
    :class:`torchtune.modules.vision_transformer.VisionTransformer`.

    Args:
        max_num_tiles (int): The maximum number of tiles an image can be divided into.
        embed_dim (int): The dimensionality of each token embedding.
        tile_size (int): The size of your image tiles, if the image was tile-cropped in advance. Otherwise,
            the size of the input image. In this case, the function will consider your image as a single tile.
        patch_size (int): The size of each patch. Used to divide the tiles into patches.
            E.g. for ``patch_size=40``, a tile of shape (400, 400) will have 10x10 grid of patches
            with shape (40, 40) each.
    """

    def __init__(
        self, max_num_tiles: int, embed_dim: int, tile_size: int, patch_size: int
    ) -> None:
        super().__init__()
        self.max_num_tiles = max_num_tiles

        patch_grid_size = tile_size // patch_size
        self.n_tokens_per_tile = patch_grid_size**2 + 1  # +1 for cls token
        scale = embed_dim**-0.5

        # different for every token, same for every tile
        self.local_token_positional_embedding = nn.Parameter(
            scale * torch.randn((self.n_tokens_per_tile, embed_dim))
        )

        # different for every token, different for every tile
        self.global_token_positional_embedding = nn.Parameter(
            scale
            * torch.randn(
                max_num_tiles,
                max_num_tiles,
                self.n_tokens_per_tile,
                embed_dim,
            )
        )

        self.gate = nn.Parameter(torch.zeros(1))

        self._register_load_state_dict_pre_hook(self._load_state_dict_hook)

    def _load_state_dict_hook(
        self,
        state_dict: Dict[str, Any],
        prefix: str,
        *args: Tuple[Any],
        **kwargs: Dict[str, Any],
    ) -> None:
        """
        Interpolates positional embeddings to accomodate different number of tiles
        and tokens per tile.

        For more info, check self._dynamic_resize function.

        Args:
            state_dict (Dict[str, Any]): The state dict to load.
            prefix (str): The prefix of the state dict.
            *args (Tuple[Any]): Additional positional arguments.
            **kwargs (Dict[str, Any]): Additional keyword arguments.

        Raises:
            ValueError: if after interpolation, the shape of the loaded local embedding
                 is not compatible with the current embedding.
            ValueError: if after interpolation, the shape of the loaded global embedding
                 is not compatible with the current embedding.
        """
        if hasattr(self, "gated_positional_embedding"):
            inpt_pos_embed = state_dict.get(prefix + "local_token_positional_embedding")
            (
                tgt_n_tokens_per_tile,
                tgt_embed_dim,
            ) = self.local_token_positional_embedding.shape
            inpt_n_tokens_per_tile, inpt_embed_dim = inpt_pos_embed.shape
            inpt_pos_embed = self._resize_local_position_embedding(
                inpt_pos_embed,
                tgt_n_tokens_per_tile=tgt_n_tokens_per_tile,
                inpt_n_tokens_per_tile=inpt_n_tokens_per_tile,
            )
            state_dict[prefix + "local_token_positional_embedding"] = inpt_pos_embed

            if inpt_pos_embed.shape != self.local_token_positional_embedding.shape:
                raise ValueError(
                    f"Loaded local positional embedding has shape {inpt_pos_embed.shape}, "
                    f"after interpolation. Expected shape {self.local_token_positional_embedding.shape}."
                )

        if hasattr(self, "gated_positional_embedding"):
            inpt_global_pos_embed = state_dict.get(
                prefix + "local_token_positional_embedding"
            )
            (
                tgt_max_num_tiles_x,
                tgt_max_num_tiles_y,
                tgt_n_tokens_per_tile,
                tgt_embed_dim,
            ) = self.global_token
            inpt_global_pos_embed = self._resize_global_position_embedding(
                global_pos_embed=inpt_global_pos_embed,
                tgt_max_num_tiles=tgt_max_num_tiles_x,
                tgt_patch_grid_size=math.sqrt(tgt_n_tokens_per_tile - 1),
            )
            state_dict[prefix + "gated_positional_embedding"] = inpt_global_pos_embed

            if (
                inpt_global_pos_embed.shape
                != self.global_token_positional_embedding.shape
            ):
                raise ValueError(
                    f"Loaded global positional embedding has shape {inpt_global_pos_embed.shape}, "
                    f"after interpolation. Expected shape {self.global_token_positional_embedding.shape}."
                )

    def _resize_local_position_embedding(
        self,
        inpt_pos_embed: torch.Tensor,
        tgt_n_tokens_per_tile: int,
        inpt_n_tokens_per_tile: int,
    ) -> torch.Tensor:
        """
        Resize position embedding for vision encoder.
        Original position embedding is [n_tiles * n_tiles + 1, dim]
        New position embedding will be [grid_size[0] * grid_size[1] + 1, dim]
        """
        # check __init__ for details
        inpt_patch_grid_size = int(math.sqrt(len(tgt_n_tokens_per_tile) - 1))
        inpt_patch_grid_size = int(math.sqrt(len(inpt_n_tokens_per_tile) - 1))

        # split tokens between cls and img tokens.
        # we don't want to interpolate cls token.
        cls_token, inpt_pos_emb = (
            inpt_pos_embed[0],  # cls token
            inpt_pos_embed[1:],  # image tokens
        )

        # local_pos_emb has shape [self.n_tokens_per_tile, embed_dim]
        # we reshape tokens_per_tile to be [inpt_patch_grid_size, inpt_patch_grid_size, -1]
        # since tokens_per_tile == inpt_patch_grid_size**2 + 1 (cls token)
        # we add 1 to the first dim because interpolate sees it as batch size.
        # Finally we permute from [1, inpt_patch_grid_size, inpt_patch_grid_size, embed_dim]
        # to [1, embed_dim, grig_patch_grid_size, inpt_patch_grid_size]
        # because only the last 2 dims get interpolated by F.interpolate
        # yielding the output shape [1, embed_dim, inpt_patch_grid_size, inpt_patch_grid_size]
        inpt_pos_emb = inpt_pos_emb.reshape(
            1, inpt_patch_grid_size, inpt_patch_grid_size, -1
        ).permute(0, 3, 1, 2)

        inpt_pos_emb = F.interpolate(
            inpt_pos_emb,
            size=[inpt_patch_grid_size, inpt_patch_grid_size],
            mode="bilinear",
            align_corners=True,
        )

        # reshape back to [1, inpt_n_tokens_per_tile, embed_dim]
        inpt_pos_emb = inpt_pos_emb.permute(0, 2, 3, 1).reshape(
            1, inpt_n_tokens_per_tile - 1, -1
        )

        # remove batch dim added previously
        inpt_pos_emb = inpt_pos_emb[0]

        inpt_pos_embed = torch.cat([cls_token, inpt_pos_emb], dim=0)
        return inpt_pos_embed

    def _resize_global_position_embedding(
        self,
        global_pos_embed: torch.Tensor,
        tgt_max_num_tiles: int,
        tgt_patch_grid_size: int,
    ) -> torch.Tensor:
        """
        Takes a global position embedding for vision encoder and resizes it to new size.
        Input: global position embedding of shape [x_old, y_old, old_grid_size[0] * old_grid_size[1] + 1, dim]
        Returns: global position embedding of shape [x_scale, y_scale, grid_size[0] * grid_size[1] + 1, dim]
        Here x_scale and y_scale are the number of tiles along x-axis and y-axis respectively.
        """
        # remove cls token to interpolate it separately
        pos_embed = global_pos_embed[:, :, 1:]
        cls_embed = global_pos_embed[:, :, 0].unsqueeze(2)

        (
            max_num_tiles_x,
            max_num_tiles_y,
            n_tokens_per_tile,
            embed_dim,
        ) = pos_embed.shape
        inpt_patch_grid_size = int(math.sqrt(n_tokens_per_tile))

        # tokens_per_tile == inpt_patch_grid_size**2
        # we reshape n_tokens_per_tile --> (inpt_patch_grid_size, inpt_patch_grid_size)
        pos_embed = pos_embed.view(
            max_num_tiles_x,
            max_num_tiles_y,
            inpt_patch_grid_size,
            inpt_patch_grid_size,
            embed_dim,
        )
        pos_embed = pos_embed.permute(0, 2, 1, 3, 4).contiguous()
        pos_embed = pos_embed.view(
            max_num_tiles_x * inpt_patch_grid_size,
            max_num_tiles_y * inpt_patch_grid_size,
            embed_dim,
        )
        pos_embed = pos_embed.unsqueeze(0)  # add batch dim for interpolation

        # interpolate
        tgt_size = (
            tgt_max_num_tiles * tgt_patch_grid_size,
            tgt_max_num_tiles * tgt_patch_grid_size,
        )
        pos_embed = pos_embed.permute(0, 3, 1, 2)
        pos_embed = F.interpolate(
            pos_embed,
            size=tgt_size,
            mode="bilinear",
            align_corners=True,
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1)[0]

        # move it back in place
        pos_embed = pos_embed.view(
            max_num_tiles_x,
            inpt_patch_grid_size,
            max_num_tiles_y,
            inpt_patch_grid_size,
            embed_dim,
        )
        pos_embed = pos_embed.permute(0, 2, 1, 3, 4).contiguous()
        pos_embed = pos_embed.view(
            max_num_tiles_x, max_num_tiles_y, n_tokens_per_tile, embed_dim
        )

        # interpolate cls token
        cls_embed = cls_embed.permute(2, 3, 0, 1)
        cls_embed_resized = F.interpolate(
            cls_embed,
            size=(tgt_max_num_tiles, tgt_max_num_tiles),
            mode="bilinear",
            align_corners=True,
        )
        cls_embed = cls_embed_resized.permute(2, 3, 0, 1)

        # add cls token back in
        global_pos_embed = torch.cat([cls_embed, pos_embed], dim=2)

        return global_pos_embed

    def forward(self, x: torch.Tensor, aspect_ratio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): torch.Tensor with shape
                (bsz * n_imgs, n_tiles, n_tokens_per_tile, embed_dim).
            aspect_ratio (torch.Tensor): torch.Tensor with shape (bsz * n_imgs, 2),
                where aspect_ratio[k] represents the aspect ratio of the k^th image
                of the batch before tile-cropping,  e.g. aspect_ratio[k] = (2,1).
        Returns:
            torch.Tensor: The input tensor with added positional embeddings.
        """
        bsz_and_n_imgs, n_tiles, n_tokens_per_tile, embed_dim = x.shape

        # apply local position embedding (same for every tile)
        x = x + (self.local_token_positional_embedding * (1 - self.gate.tanh()))

        # apply global positional embedding (different for every tile)
        x = x.view(bsz_and_n_imgs, n_tiles, n_tokens_per_tile, embed_dim)
        for batch_idx, (n_tiles_h, n_tiles_w) in enumerate(aspect_ratio):
            # When we batch images, all are padded to the same amount of tiles.
            # The aspect_ratio lets us know the non padded tiles for each image.
            # We only add positional encoding to those.
            n_non_padded_tiles = int(n_tiles_h * n_tiles_w)

            # We get only the positional encoding for non padded tiles,
            # i.e. n_tiles_h, n_tiles_w.
            pos_embed = self.global_token_positional_embedding[
                :n_tiles_h, :n_tiles_w, :, :
            ]

            # Add pos encoding to the non padded tiles.
            pos_embed = pos_embed.reshape(
                n_non_padded_tiles, self.n_tokens_per_tile, embed_dim
            )
            pos_embed = pos_embed * self.gate.tanh()
            x[batch_idx, :n_non_padded_tiles, :, :] += pos_embed

        return x


class TilePositionalEmbedding(nn.Module):
    """
    Positional embedding for tiles, different for every tile, same for every token within a tile.

    Notice that tile is different from patch (token). For details, please check the documentation of
    :class:`torchtune.modules.vision_transformer.VisionTransformer`.

    Args:
        max_num_tiles (int): The maximum number of tiles an image can be divided into.
        embed_dim (int): The dimensionality of each tile embedding.
    """

    def __init__(
        self,
        max_num_tiles: int,
        embed_dim: int,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        scale = embed_dim**-0.5
        self.embedding = nn.Parameter(
            scale * torch.randn(max_num_tiles, max_num_tiles, 1, embed_dim)
        )
        self.gate = nn.Parameter(torch.zeros(1))

        # Register load hook to interpolate positional embeddings
        self._register_load_state_dict_pre_hook(self._load_state_dict_hook)

    def _load_state_dict_hook(
        self,
        state_dict: Dict[str, Any],
        prefix: str,
        *args: Tuple[Any],
        **kwargs: Dict[str, Any],
    ):
        """
        Interpolates positional embeddings to accomodate different number of tiles.

        For more info, check self._dynamic_resize function.

        Args:
            state_dict (Dict[str, Any]): The state dict to load.
            prefix (str): The prefix of the state dict.
            *args (Tuple[Any]): Additional positional arguments.
            **kwargs (Dict[str, Any]): Additional keyword arguments.

        Raises:
            ValueError: if the shape of the loaded embedding is not compatible with the current embedding.
            ValueError: if max_num_tiles_x, max_num_tiles_y are not equal.
            ValueError: if after interpolation, the shape of the loaded embedding is not compatible with the current embedding.
        """

        if hasattr(self, "embedding"):
            embedding = state_dict.get(prefix + "embedding")

            # Error if shape is not compatible
            (
                tgt_max_num_tiles_x,
                tgt_max_num_tiles_y,
                tgt_num_tokens,
                tgt_emb,
            ) = self.embedding.shape
            (
                inpt_max_num_tiles_x,
                inpt_max_num_tiles_y,
                inpt_num_tokens,
                inpt_emb,
            ) = state_dict[prefix + "embedding"].shape
            if inpt_num_tokens != tgt_num_tokens or inpt_emb != tgt_emb:
                raise ValueError(
                    "Expected embedding shape to be (..., num_tokens, tgt_emb) to match"
                    f" but found shapes {self.embedding.shape} and {state_dict[prefix+'embedding'].shape}"
                )

            # Error if shape is not compatible
            if inpt_max_num_tiles_x != inpt_max_num_tiles_y:
                raise ValueError(
                    "Expected max_num_tiles_x, max_num_tiles_y to be equal but found, but found"
                    f"(max_num_tiles_x, max_num_tiles_y, 1, embed_dim) = {self.embedding.shape}"
                )

            # interpolate
            embedding_new = self._dynamic_resize(
                embedding, tgt_num_tiles=tgt_max_num_tiles_x
            )

            # Error if shape after interpolation is not the same
            if embedding_new.shape != self.embedding.shape:
                raise ValueError(
                    "Expected embedding shape and embedding_new.shape to match"
                    f" but found shapes {self.embedding.shape} and {embedding_new.shape}"
                )

            # assign
            state_dict[prefix + "embedding"] = embedding_new

    @staticmethod
    def _dynamic_resize(embedding: torch.Tensor, tgt_num_tiles: int) -> torch.Tensor:
        """
        Interpolates positional embeddings to accomodate different number of tiles.

        self.embedding is of shape [max_num_tiles, max_num_tiles, 1, embed_dim].

        This resize expects (..., 1, embed_dim) to remain unchanged, so it will only interpolate
        the first two dimensions of self.embedding, i.e. [max_num_tiles, max_num_tiles].

        Args:
            embedding (torch.Tensor): torch.Tensor with shape (max_num_tiles, max_num_tiles, 1, embed_dim
            tgt_num_tiles (int): The number of tiles to resize to.

        Returns:
            torch.Tensor: The resized embedding.

        Example:
            >>> import torch
            >>> embedding = torch.arange(2*2*2*2).reshape(2, 2, 2, 2).float()
            >>> resized_embed = _dynamic_resize(embedding, tgt_num_tiles=1)
            >>> print(resized_embed.shape)
            >>> torch.Size([1, 1, 2, 2])
        """
        embedding = embedding.permute(2, 3, 0, 1)

        embedding = F.interpolate(
            embedding,
            size=(tgt_num_tiles, tgt_num_tiles),
            mode="bilinear",
            align_corners=True,
        )
        # reshape the weights to the correct shape
        embedding = embedding.permute(2, 3, 0, 1)
        return embedding

    def forward(self, x: torch.Tensor, aspect_ratio: torch.Tensor) -> torch.Tensor:
        """
        args:
            x (torch.Tensor): torch.Tensor with shape (bsz * n_imgs, n_tiles, n_tokens_per_tile, embed_dim).
            aspect_ratio (torch.Tensor): torch.Tensor with shape (bsz * n_imgs, 2),
                representing the aspect ratio of the image before tile-cropping, e.g. (2,1).
        returns:
            torch.Tensor: The input tensor with added positional embeddings.
        """

        for batch_idx, (n_tiles_h, n_tiles_w) in enumerate(aspect_ratio):
            # When we batch images, all are padded to the same amount of tiles.
            # The aspect_ratio lets us know the non padded tiles for each image.
            # We only add positional encoding to those.
            n_non_padded_tiles = int(n_tiles_h * n_tiles_w)

            # We get only the positional encoding for non padded tiles,
            # i.e. n_tiles_h, n_tiles_w.
            pos_embed = self.embedding[:n_tiles_h, :n_tiles_w, :, :]

            # Add pos encoding to the non padded tiles.
            pos_embed = pos_embed.reshape(n_non_padded_tiles, 1, self.embed_dim)
            x[batch_idx, :n_non_padded_tiles, :, :] += pos_embed * self.gate.tanh()

        return x
