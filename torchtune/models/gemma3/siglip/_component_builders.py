import torch
import torch.nn.functional as F

from torch import nn
from torchtune.modules import MultiHeadAttention, FeedForward


def siglip_attention(dim: int, num_heads: int, head_dim: int) -> MultiHeadAttention:
    """
    Builds the attention associated with the siglip model.

    Args:

    """
    num_kv_heads = num_heads
    q_proj = nn.Linear(dim, num_heads * head_dim, bias=True)
    k_proj = nn.Linear(dim, num_kv_heads * head_dim, bias=True)
    v_proj = nn.Linear(dim, num_kv_heads * head_dim, bias=True)
    output_proj = nn.Linear(num_heads * head_dim, dim, bias=True)

    return MultiHeadAttention(
        embed_dim=dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        q_proj=q_proj,
        k_proj=k_proj,
        v_proj=v_proj,
        output_proj=output_proj,
        pos_embeddings=None,
        q_norm=None,
        k_norm=None,
        kv_cache=None,
        is_causal=False,
        attn_dropout=0.0,
    )


def siglip_mlp(hidden_size: int, intermediate_size: int) -> FeedForward:
    gate_proj = nn.Linear(hidden_size, intermediate_size)
    down_proj = nn.Linear(intermediate_size, hidden_size)

    siglip_mlp = FeedForward(
        gate_proj=gate_proj,
        down_proj=down_proj,
        up_proj=None,
        # There is no reason to reimplemate it, like in SigLIP done.
        activation=nn.GELU(approximate="tanh"),
    )

    return siglip_mlp


class SiglipAveragePooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        batch_size, seq_len, channels = x.shape
        width = int(seq_len**0.5)
        if width * width != seq_len:
            raise ValueError(
                f"Sequence length {seq_len} is not a perfect square. Cannot reshape to a square image."
            )
        # Bx(64^2)x1152 -> Bx1152x(64^2) -> Bx1152x64x64
        x = x.transpose(1, 2).reshape(batch_size, channels, width, width)
        # Bx1152x64x64-> Bx1152x16x16
        x = F.avg_pool2d(x, kernel_size=4, stride=4)
        # Bx1152x64x64-> Bx1152x256 -> Bx256x1152
        x = x.flatten(2).transpose(1, 2)
        return x


class SiglipEncoderBlock(nn.Module):
    """
    Reference implementation: https://github.com/google/gemma_pytorch/blob/014acb7ac4563a5f77c76d7ff98f31b568c16508/gemma/siglip_vision/siglip_vision_model.py#L120
    """

    def __init__(
        self,
        embedding_dim: int,
        num_attention_heads: int,
        head_dim: int,
        intermediate_size: int,
        layer_norm_eps: float,
    ):
        super().__init__()
        self.self_attn = siglip_attention(embedding_dim, num_attention_heads, head_dim)
        self.layer_norm1 = nn.LayerNorm(embedding_dim, eps=layer_norm_eps)
        self.mlp = siglip_mlp(embedding_dim, intermediate_size)
        self.layer_norm2 = nn.LayerNorm(embedding_dim, eps=layer_norm_eps)

    def forward(self, x):
        residual = x
        x = self.layer_norm1(x)
        x = self.self_attn(x)
        x = x + residual  # Residual connection *after* LayerNorm

        residual = x
        x = self.layer_norm2(x)
        x = self.mlp(x)
        x = x + residual  # Residual connection *after* LayerNorm
        return x


class SiglipVisionModel(nn.Module):
    """
    Reference implementation: https://github.com/google/gemma_pytorch/blob/014acb7ac4563a5f77c76d7ff98f31b568c16508/gemma/siglip_vision/siglip_vision_model.py#L151
    """

    def __init__(
        self,
        input_channels: int,
        embedding_dim: int,
        conv2d_patch_size: int,
        image_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        head_dim: int,
        intermediate_size: int,
        layer_norm_eps: float,
        embedding_use_bias: bool,
    ):
        super().__init__()

        # SigLiPFromPatches_0/siglip_encoder/embedding
        self.patch_embedding = nn.Conv2d(
            in_channels=input_channels,
            out_channels=embedding_dim,
            kernel_size=conv2d_patch_size,
            stride=conv2d_patch_size,
            padding=0,
            bias=embedding_use_bias,
        )
        self.num_patches = (image_size // conv2d_patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, embedding_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

        self.encoder_blocks = nn.ModuleList(
            SiglipEncoderBlock(
                embedding_dim=embedding_dim,
                num_attention_heads=num_attention_heads,
                head_dim=head_dim,
                intermediate_size=intermediate_size,
                layer_norm_eps=layer_norm_eps,
            )
            for _ in range(num_hidden_layers)
        )

        self.final_norm = nn.LayerNorm(embedding_dim, layer_norm_eps)
        self.avg_pool = SiglipAveragePooling()

    @torch.inference_mode
    def forward(
        self,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        # Embed the image according to SiplipVisionEmbeddings.
        x = self.patch_embedding(pixel_values)
        # (batch_size ,channels, height, width)->(batch_size, height*width, channels)
        x = x.flatten(2).transpose(1, 2)

        position_ids = self.position_ids.to(pixel_values.device)
        x = x + self.position_embedding(position_ids)

        for block in self.encoder_blocks:
            x = block(x)  # batch_size, height*width, embedding_dim (1152)
        x = self.final_norm(x)

        # siglip exit https://source.corp.google.com/piper///depot/google3/third_party/py/gemma/multimodal/vision.py;l=220
        return self.avg_pool(x)
