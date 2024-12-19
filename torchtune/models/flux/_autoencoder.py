# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from torchtune.modules.attention import MultiHeadAttention

# ch = number of channels (size of the channel dimension)


class FluxAutoencoder(nn.Module):
    """
    The image autoencoder for Flux diffusion models.

    Args:
        resolution (int): The height/width of the square input image.
        ch_in (int): The number of channels of the input image.
        ch_out (int): The number of channels of the output image.
        ch_base (int): The base number of channels.
            This gets multiplied by `ch_mult` values to get the number of inner channels during downsampling/upsampling.
        ch_mults (List[int]): The channel multiple per downsample/upsample block.
            This gets multiplied by `ch_base` to get the number of inner channels during downsampling/upsampling.
        ch_z (int): The number of latent channels (dimension of the latent vector `z`).
        n_layers_per_resample_block (int): Number of resnet layers per downsample/upsample block.
        scale_factor (float): Constant for scaling `z`.
        shift_factor (float): Constant for shifting `z`.
    """

    def __init__(
        self,
        resolution: int,
        ch_in: int,
        ch_out: int,
        ch_base: int,
        ch_mults: List[int],
        ch_z: int,
        n_layers_per_resample_block: int,
        scale_factor: float,
        shift_factor: float,
    ):
        super().__init__()
        self.img_shape = (ch_in, resolution, resolution)

        channels = [ch_base * mult for mult in ch_mults]

        self.encoder = Encoder(
            ch_in=ch_in,
            ch_z=ch_z,
            channels=channels,
            n_layers_per_down_block=n_layers_per_resample_block,
            scale_factor=scale_factor,
            shift_factor=shift_factor,
        )

        self.decoder = Decoder(
            ch_out=ch_out,
            ch_z=ch_z,
            channels=list(reversed(channels)),
            # decoder gets one more layer per up block than the encoder's down blocks
            n_layers_per_up_block=n_layers_per_resample_block + 1,
            scale_factor=scale_factor,
            shift_factor=shift_factor,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): input image of shape [bsz, ch_in, img resolution, img resolution]

        Returns:
            Tensor: output image of the same shape
        """
        return self.decode(self.encode(x))

    def encode(self, x: Tensor) -> Tensor:
        """
        Encode images into their latent representations.

        Args:
            x (Tensor): input images (shape = [bsz, ch_in, img resolution, img resolution])

        Returns:
            Tensor: latent encodings (shape = [bsz, ch_z, latent resolution, latent resolution])
        """
        assert x.shape[1:] == self.img_shape
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        """
        Decode latent representations into images.

        Args:
            z (Tensor): latent encodings (shape = [bsz, ch_z, latent resolution, latent resolution])

        Returns:
            Tensor: output images (shape = [bsz, ch_in, img resolution, img resolution])
        """
        return self.decoder(z)


class Encoder(nn.Module):
    """
    The encoder half of the Flux diffusion model's image autoencoder.

    Args:
        ch_in (int): The number of channels of the input image.
        ch_z (int): The number of latent channels (dimension of the latent vector `z`).
        channels (List[int]): The number of output channels for each downsample block.
        n_layers_per_down_block (int): Number of resnet layers per upsample block.
        scale_factor (float): Constant for scaling `z`.
        shift_factor (float): Constant for shifting `z`.
    """

    def __init__(
        self,
        ch_in: int,
        ch_z: int,
        channels: List[int],
        n_layers_per_down_block: int,
        scale_factor: float,
        shift_factor: float,
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.shift_factor = shift_factor

        self.conv_in = nn.Conv2d(ch_in, channels[0], kernel_size=3, stride=1, padding=1)

        self.down = nn.Sequential(
            *[
                DownBlock(
                    n_layers=n_layers_per_down_block,
                    ch_in=channels[i - 1] if i > 0 else channels[0],
                    ch_out=channels[i],
                    downsample=i < len(channels) - 1,
                )
                for i in range(len(channels))
            ]
        )

        self.mid = mid_block(channels[-1])

        self.end = end_block(channels[-1], 2 * ch_z)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): input images (shape = [bsz, ch_in, img resolution, img resolution])

        Returns:
            Tensor: latent encodings (shape = [bsz, ch_z, latent resolution, latent resolution])
        """
        h = self.conv_in(x)
        h = self.down(h)
        h = self.mid(h)
        h = self.end(h)
        z = diagonal_gaussian(h)
        return self.scale_factor * (z - self.shift_factor)


class Decoder(nn.Module):
    """
    The encoder half of the Flux diffusion model's image autoencoder.

    Args:
        ch_out (int): The number of channels of the output image.
        ch_z (int): The number of latent channels (dimension of the latent vector `z`).
        channels (List[int]): The number of output channels for each upsample block.
        n_layers_per_up_block (int): Number of resnet layers per upsample block.
        scale_factor (float): Constant for scaling `z`.
        shift_factor (float): Constant for shifting `z`.
    """

    def __init__(
        self,
        ch_out: int,
        ch_z: int,
        channels: List[int],
        n_layers_per_up_block: int,
        scale_factor: float,
        shift_factor: float,
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.shift_factor = shift_factor

        self.conv_in = nn.Conv2d(ch_z, channels[0], kernel_size=3, stride=1, padding=1)

        self.mid = mid_block(channels[0])

        self.up = nn.Sequential(
            *[
                UpBlock(
                    n_layers=n_layers_per_up_block,
                    ch_in=channels[i - 1] if i > 0 else channels[0],
                    ch_out=channels[i],
                    upsample=i < len(channels) - 1,
                )
                for i in range(len(channels))
            ]
        )

        self.end = end_block(channels[-1], ch_out)

    def forward(self, z: Tensor) -> Tensor:
        """
        Args:
            z (Tensor): latent encodings (shape = [bsz, ch_z, latent resolution, latent resolution])

        Returns:
            Tensor: output images (shape = [bsz, ch_in, img resolution, img resolution])
        """
        z = z / self.scale_factor + self.shift_factor
        h = self.conv_in(z)
        h = self.mid(h)
        h = self.up(h)
        x = self.end(h)
        return x


def mid_block(ch: int) -> nn.Module:
    return nn.Sequential(
        ResnetLayer(ch_in=ch, ch_out=ch),
        AttnLayer(ch),
        ResnetLayer(ch_in=ch, ch_out=ch),
    )


def end_block(ch_in: int, ch_out: int) -> nn.Module:
    return nn.Sequential(
        nn.GroupNorm(num_groups=32, num_channels=ch_in, eps=1e-6, affine=True),
        nn.SiLU(),
        nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
    )


class DownBlock(nn.Module):
    def __init__(self, n_layers: int, ch_in: int, ch_out: int, downsample: bool):
        super().__init__()
        self.layers = resnet_layers(n_layers, ch_in, ch_out)
        self.downsample = Downsample(ch_out) if downsample else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.downsample(self.layers(x))


class UpBlock(nn.Module):
    def __init__(self, n_layers: int, ch_in: int, ch_out: int, upsample: bool):
        super().__init__()
        self.layers = resnet_layers(n_layers, ch_in, ch_out)
        self.upsample = Upsample(ch_out) if upsample else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.upsample(self.layers(x))


def resnet_layers(n: int, ch_in: int, ch_out: int) -> nn.Module:
    return nn.Sequential(
        *[
            ResnetLayer(ch_in=ch_in if i == 0 else ch_out, ch_out=ch_out)
            for i in range(n)
        ]
    )


class AttnLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.norm = nn.GroupNorm(num_groups=32, num_channels=dim, eps=1e-6, affine=True)
        self.attn = MultiHeadAttention(
            embed_dim=dim,
            num_heads=1,
            num_kv_heads=1,
            head_dim=dim,
            q_proj=nn.Linear(dim, dim),
            k_proj=nn.Linear(dim, dim),
            v_proj=nn.Linear(dim, dim),
            output_proj=nn.Linear(dim, dim),
            is_causal=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        residual = x

        x = self.norm(x)

        # b c h w -> b (h w) c
        x = torch.einsum("bchw -> bhwc", x)
        x = x.reshape(b, h * w, c)

        x = self.attn(x, x)

        # b (h w) c -> b c h w
        x = x.reshape(b, h, w, c)
        x = torch.einsum("bhwc -> bchw", x)

        return x + residual


class ResnetLayer(nn.Module):
    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.main = nn.Sequential(
            *[
                nn.GroupNorm(num_groups=32, num_channels=ch_in, eps=1e-6, affine=True),
                nn.SiLU(),
                nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(num_groups=32, num_channels=ch_out, eps=1e-6, affine=True),
                nn.SiLU(),
                nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1),
            ]
        )
        self.shortcut = (
            nn.Identity()
            if ch_in == ch_out
            else nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.main(x) + self.shortcut(x)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(F.pad(x, (0, 1, 0, 1), mode="constant", value=0))


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


def diagonal_gaussian(z: Tensor) -> Tensor:
    mean, logvar = torch.chunk(z, 2, dim=1)
    std = torch.exp(0.5 * logvar)
    return mean + std * torch.randn_like(mean)
