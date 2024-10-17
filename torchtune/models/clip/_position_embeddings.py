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
from torch.distributed._tensor import distribute_tensor, DTensor


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

    Token positional embedding for tiled images, different for every tile, different for every token.

    There are two positional embeddings in this module:

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

    @torch.no_grad()
    def _load_state_dict_hook(
        self,
        state_dict: Dict[str, Any],
        prefix: str,
        *args: Tuple[Any],
        **kwargs: Dict[str, Any],
    ) -> None:
        """
        Interpolates positional embeddings to accomodate different number of tiles
        and tokens per tile, in case the model was instantiated with different
        settings than the one you are loading the state dict from.

        For more info, please check self._resize_local_position_embedding and
        self._resize_global_position_embedding functions.

        Args:
            state_dict (Dict[str, Any]): The state dict to load.
            prefix (str): The prefix of the state dict.
            *args (Tuple[Any]): Additional positional arguments.
            **kwargs (Dict[str, Any]): Additional keyword arguments.

        Raises:
            ValueError: if loaded local or global embedding n_tokens_per_tile is not derived
                from a squared grid.
            ValueError: if after interpolation, the shape of the loaded local embedding
                is not compatible with the current embedding.
            ValueError: if after interpolation, the shape of the loaded global embedding
                is not compatible with the current embedding.
        """

        # process local_token_positional_embedding
        inpt_local_pos_embed = state_dict.get(
            prefix + "local_token_positional_embedding"
        )

        if inpt_local_pos_embed is not None:

            # We can only apply F.interpolate to vanilla tensors, not DTensors
            # If pos embeds are a DTensor, we gather the full tensor, apply
            # interpolate, and then reshard after
            if isinstance(inpt_local_pos_embed, DTensor):
                local_embed_is_sharded = True
                local_embed_device_mesh = inpt_local_pos_embed.device_mesh
                local_embed_placements = inpt_local_pos_embed.placements
                inpt_local_pos_embed = inpt_local_pos_embed.full_tensor()
            else:
                local_embed_is_sharded = False

            # sanity check
            inpt_n_tokens_per_tile, inpt_embed_dim = inpt_local_pos_embed.shape
            if math.sqrt(inpt_n_tokens_per_tile - 1) % 1 != 0:
                raise ValueError(
                    f"Loaded local positional embedding has shape {inpt_n_tokens_per_tile=}, "
                    f"which indicates a grid_size that is not squared. This is currently not supported."
                )

            # instantiated pos emb
            (
                tgt_n_tokens_per_tile,
                tgt_embed_dim,
            ) = self.local_token_positional_embedding.shape

            # resize ckpt to match instantiated shape
            inpt_local_pos_embed = self._resize_local_position_embedding(
                local_pos_embed=inpt_local_pos_embed,
                tgt_patch_grid_size=int(math.sqrt(tgt_n_tokens_per_tile - 1)),
            )

            if local_embed_is_sharded:
                inpt_local_pos_embed = distribute_tensor(
                    inpt_local_pos_embed,
                    device_mesh=local_embed_device_mesh,
                    placements=local_embed_placements,
                )

            # update state dict
            state_dict[
                prefix + "local_token_positional_embedding"
            ] = inpt_local_pos_embed
            if (
                inpt_local_pos_embed.shape
                != self.local_token_positional_embedding.shape
            ):
                raise ValueError(
                    f"Loaded local positional embedding has shape {inpt_local_pos_embed.shape}, "
                    f"after interpolation. Expected shape {self.local_token_positional_embedding.shape}."
                )

        # process global_token_positional_embedding
        inpt_global_pos_embed = state_dict.get(
            prefix + "global_token_positional_embedding"
        )

        if inpt_global_pos_embed is not None:

            # We can only apply F.interpolate to vanilla tensors, not DTensors
            # If pos embeds are a DTensor, we gather the full tensor, apply
            # interpolate, and then reshard after
            if isinstance(inpt_global_pos_embed, DTensor):
                global_embed_is_sharded = True
                global_embed_device_mesh = inpt_global_pos_embed.device_mesh
                global_embed_placements = inpt_global_pos_embed.placements
                inpt_global_pos_embed = inpt_global_pos_embed.full_tensor()
            else:
                global_embed_is_sharded = False

            _, _, inpt_n_tokens_per_tile, _ = inpt_global_pos_embed.shape

            # sanity check
            if math.sqrt(inpt_n_tokens_per_tile - 1) % 1 != 0:
                raise ValueError(
                    f"Loaded local positional embedding has shape {inpt_n_tokens_per_tile=}, "
                    f"which indicates a grid_size that is not squared. This is currently not supported."
                )

            # instantiated pos emb
            (
                tgt_max_num_tiles_x,
                tgt_max_num_tiles_y,  # not used, same as tgt_max_num_tiles_x
                tgt_n_tokens_per_tile,
                tgt_embed_dim,
            ) = self.global_token_positional_embedding.shape

            # resize ckpt to match instantiated shape
            inpt_global_pos_embed = self._resize_global_position_embedding(
                global_pos_embed=inpt_global_pos_embed,
                tgt_max_num_tiles=tgt_max_num_tiles_x,
                tgt_patch_grid_size=int(math.sqrt(tgt_n_tokens_per_tile - 1)),
            )

            if global_embed_is_sharded:
                inpt_global_pos_embed = distribute_tensor(
                    inpt_global_pos_embed,
                    device_mesh=global_embed_device_mesh,
                    placements=global_embed_placements,
                )

            # update state dict
            state_dict[
                prefix + "global_token_positional_embedding"
            ] = inpt_global_pos_embed
            if (
                inpt_global_pos_embed.shape
                != self.global_token_positional_embedding.shape
            ):
                raise ValueError(
                    f"Loaded global positional embedding has shape {inpt_global_pos_embed.shape}, "
                    f"after interpolation. Expected shape {self.global_token_positional_embedding.shape}."
                )

    @staticmethod
    def _resize_local_position_embedding(
        local_pos_embed: torch.Tensor, tgt_patch_grid_size: int
    ) -> torch.Tensor:
        """
        Interpolates the local position embedding for a vision encoder to accommodate
        a different number of tokens per tile. This is the only dimension that
        changes during interpolation.

        Args:
            local_pos_embed (torch.Tensor): The position embeddings tensor to be resized. It
                has shape [n_tokens_per_tile, emb_dim], where the first token is the CLS token
                and n_tokens_per_tile = patch_grid_size**2 + 1.
            tgt_patch_grid_size (int): The target size of each patch grid, i.e.,
                the square root of the number of tokens per tile, excluding the class token.

        Returns:
            torch.Tensor: The resized position embeddings tensor of shape
                [tgt_n_tokens_per_tile, dim], where tgt_n_tokens_per_tile = tgt_patch_grid_size**2 + 1.

        Example:
            >>> import torch
            >>> import math
            >>> local_pos_embed = torch.randn((10*10+1, 64))  # Example input tensor
            >>> tgt_patch_grid_size = 20  # Target number of tokens per tile
            >>> resized_pos_embed = _resize_local_position_embedding(local_pos_embed, tgt_patch_grid_size)
            >>> print(resized_pos_embed.shape)
            torch.Size([20*20+1, 64])
        """
        # inverse n_tokens_per_tile = patch_grid_size**2 + 1, where +1 is the cls token
        inpt_n_tokens_per_tile, inpt_embed_dim = local_pos_embed.shape
        inpt_patch_grid_size = int(math.sqrt(inpt_n_tokens_per_tile - 1))

        # split tokens between cls and img tokens.
        # we don't want to interpolate cls token.
        cls_token, local_pos_embed = (
            local_pos_embed[[0]],  # cls token
            local_pos_embed[1:],  # image tokens
        )

        # we reshape n_tokens_per_tile - 1 --> (inpt_patch_grid_size, inpt_patch_grid_size)
        # and permute to have inpt_patch_grid_size as the last two dimensions
        # we also add a batch dim to the tensor, since F.interpolate expects it
        local_pos_embed = local_pos_embed.reshape(
            1, inpt_patch_grid_size, inpt_patch_grid_size, -1
        ).permute(0, 3, 1, 2)

        local_pos_embed = F.interpolate(
            local_pos_embed,
            size=[tgt_patch_grid_size, tgt_patch_grid_size],
            mode="bilinear",
            align_corners=True,  # defaults from internal-llama-models
        )

        # reshape back to [1, tokens_per_tile, embed_dim]
        local_pos_embed = local_pos_embed.permute(0, 2, 3, 1).reshape(
            1, -1, inpt_embed_dim
        )

        # remove batch dim added previously
        local_pos_embed = local_pos_embed.squeeze(0)

        # add cls token back in
        local_pos_embed = torch.cat([cls_token, local_pos_embed], dim=0)

        return local_pos_embed

    # TODO: Switch to public method after 2.5 is stable
    @staticmethod
    def _resize_global_position_embedding(
        global_pos_embed: torch.Tensor,
        tgt_max_num_tiles: int,
        tgt_patch_grid_size: int,
    ) -> torch.Tensor:
        """
        Interpolates the global position embedding for a vision encoder to accommodate new grid dimensions.
        The embedding dimension is not changed during interpolation, only max_num_tiles and num_tokens_per_tile.

        Args:
            global_pos_embed (torch.Tensor): The input global position embeddings tensor of shape
                [max_num_tiles_x, max_num_tiles_y, num_tokens_per_tile, embed_dim],
                where num_tokens_per_tile = inpt_patch_grid_size * inpt_patch_grid_size + 1 (CLS token), and
                max_num_tiles_x == max_num_tiles_y.
            tgt_max_num_tiles (int): The target maximum number of tiles along one dimension (assumed square grid).
            tgt_patch_grid_size (int): The target size of each patch grid, i.e., the square root of the number of tokens
                per tile, excluding the class token.


        Returns:
            torch.Tensor: The resized global position embeddings tensor of shape
                [tgt_max_num_tiles, tgt_max_num_tiles, tgt_patch_grid_size * tgt_patch_grid_size + 1, embed_dim].

        Example:
            >>> import torch
            >>> global_pos_embed = torch.arange(3*3*(2*2+1)*4).reshape((3, 3, 2*2+1, 4))  # Example input tensor
            >>> tgt_max_num_tiles = 2  # Target maximum number of tiles
            >>> tgt_patch_grid_size = 3  # Target patch grid size
            >>> resized_global_pos_embed = (
            >>> _resize_global_position_embedding(global_pos_embed, tgt_max_num_tiles, tgt_patch_grid_size))
            >>> print(resized_global_pos_embed.shape)
            torch.Size([2, 2, 3*3+1, 4])
        """

        # remove cls token to interpolate it separately
        pos_embed = global_pos_embed[:, :, 1:, :]
        cls_embed = global_pos_embed[:, :, [0], :]

        (
            max_num_tiles_x,
            max_num_tiles_y,
            n_tokens_per_tile,
            embed_dim,
        ) = pos_embed.shape

        # tokens_per_tile == inpt_patch_grid_size**2
        # we reshape n_tokens_per_tile --> (inpt_patch_grid_size, inpt_patch_grid_size)
        inpt_patch_grid_size = int(math.sqrt(n_tokens_per_tile))
        pos_embed = pos_embed.reshape(
            max_num_tiles_x,
            max_num_tiles_y,
            inpt_patch_grid_size,
            inpt_patch_grid_size,
            embed_dim,
        )

        # combine max_num_tiles and patch_grid_size into one dimension
        pos_embed = pos_embed.permute(0, 2, 1, 3, 4).contiguous()
        pos_embed = pos_embed.reshape(
            max_num_tiles_x * inpt_patch_grid_size,
            max_num_tiles_y * inpt_patch_grid_size,
            embed_dim,
        )

        # add batch dim for interpolation
        pos_embed = pos_embed.unsqueeze(0)

        tgt_size = (
            int(tgt_max_num_tiles * tgt_patch_grid_size),
            int(tgt_max_num_tiles * tgt_patch_grid_size),
        )

        # move to the last two dim for interpolation
        pos_embed = pos_embed.permute(0, 3, 1, 2)
        pos_embed = F.interpolate(
            pos_embed,
            size=tgt_size,
            mode="bilinear",
            align_corners=True,  # defaults from internal-llama-models
        )

        # return to original shape and remove batch dim
        pos_embed = pos_embed.permute(0, 2, 3, 1).squeeze(0)

        # move it back in place
        pos_embed = pos_embed.view(
            tgt_max_num_tiles,
            tgt_patch_grid_size,
            tgt_max_num_tiles,
            tgt_patch_grid_size,
            embed_dim,
        )
        pos_embed = pos_embed.permute(0, 2, 1, 3, 4).contiguous()
        pos_embed = pos_embed.view(
            tgt_max_num_tiles,
            tgt_max_num_tiles,
            int(tgt_patch_grid_size**2),
            embed_dim,
        )

        # interpolate cls token
        cls_embed = cls_embed.permute(2, 3, 0, 1)
        cls_embed_resized = F.interpolate(
            cls_embed,
            size=(tgt_max_num_tiles, tgt_max_num_tiles),
            mode="bilinear",
            align_corners=True,  # defaults from internal-llama-models
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

    # TODO: Switch to public method after 2.5 is stable
    @torch.no_grad()
    def _load_state_dict_hook(
        self,
        state_dict: Dict[str, Any],
        prefix: str,
        *args: Tuple[Any],
        **kwargs: Dict[str, Any],
    ):
        """
        Interpolates positional embeddings to accomodate different number of tiles,
        in case the model was instantiated with different
        settings than the one you are loading the state dict from.

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

        embedding = state_dict.get(prefix + "embedding")

        if embedding is not None:

            # We can only apply F.interpolate to vanilla tensors, not DTensors
            # If pos embeds are a DTensor, we gather the full tensor, apply
            # interpolate, and then reshard after
            if isinstance(embedding, DTensor):
                embedding_is_sharded = True
                device_mesh = embedding.device_mesh
                placements = embedding.placements
                embedding = embedding.full_tensor()
            else:
                embedding_is_sharded = False

            # ckpt pos emb
            (
                tgt_max_num_tiles_x,
                tgt_max_num_tiles_y,
                tgt_num_tokens,
                tgt_emb,
            ) = self.embedding.shape

            # instantiated pos emb
            (
                inpt_max_num_tiles_x,
                inpt_max_num_tiles_y,
                inpt_num_tokens,
                inpt_emb,
            ) = state_dict[prefix + "embedding"].shape

            # sanity check
            if inpt_num_tokens != tgt_num_tokens or inpt_emb != tgt_emb:
                raise ValueError(
                    "Expected embedding shape to be (..., num_tokens, tgt_emb) to match"
                    f" but found shapes {self.embedding.shape} and {state_dict[prefix + 'embedding'].shape}"
                )

            if inpt_max_num_tiles_x != inpt_max_num_tiles_y:
                raise ValueError(
                    "Expected max_num_tiles_x, max_num_tiles_y to be equal but found, but found"
                    f"(max_num_tiles_x, max_num_tiles_y, 1, embed_dim) = {self.embedding.shape}"
                )

            # resize ckpt to match instantiated shape
            embedding_new = self._resize_position_embedding(
                embedding, tgt_max_num_tiles=tgt_max_num_tiles_x
            )

            if embedding_is_sharded:
                embedding_new = distribute_tensor(
                    embedding_new,
                    device_mesh=device_mesh,
                    placements=placements,
                )

            # update state dict
            state_dict[prefix + "embedding"] = embedding_new
            if embedding_new.shape != self.embedding.shape:
                raise ValueError(
                    "Expected embedding shape and embedding_new.shape to match"
                    f" but found shapes {self.embedding.shape} and {embedding_new.shape}"
                )

    @staticmethod
    def _resize_position_embedding(
        embedding: torch.Tensor, tgt_max_num_tiles: int
    ) -> torch.Tensor:
        """
        Interpolates positional embeddings to accomodate a different max_num_tiles. These
        are the only dimensions that changes during interpolation.

        Args:
            embedding (torch.Tensor): torch.Tensor with shape (max_num_tiles, max_num_tiles, 1, embed_dim
            tgt_max_num_tiles (int): The number of tiles to resize to.

        Returns:
            torch.Tensor: The resized embedding.

        Example:
            >>> import torch
            >>> # create dummy embedding
            >>> embedding = torch.arange(2*2*2*2).reshape(2, 2, 2, 2).float()
            >>> resized_embed = _dynamic_resize(embedding, tgt_max_num_tiles=1)
            >>> print(resized_embed.shape)
            >>> torch.Size([1, 1, 2, 2])
        """
        # set max_num_tiles to the last dimension
        embedding = embedding.permute(2, 3, 0, 1)

        embedding = F.interpolate(
            embedding,
            size=(tgt_max_num_tiles, tgt_max_num_tiles),
            mode="bilinear",
            align_corners=True,
        )
        # permute to the original shape
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
