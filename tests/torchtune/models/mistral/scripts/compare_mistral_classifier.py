# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from tests.test_utils import fixed_init_model
from torch import nn
from torchtune.models.mistral import mistral_classifier
from torchtune.models.mistral._component_builders import mistral_mlp
from torchtune.modules import (
    MultiHeadAttention,
    RMSNorm,
    RotaryPositionalEmbeddings,
    TransformerDecoder,
    TransformerSelfAttentionLayer,
)


# Copying our mistral implementation here to allow access to `output_proj`
def mistral(
    vocab_size: int,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    embed_dim: int,
    intermediate_dim: int,
    max_seq_len: int,
    output_proj: nn.Linear,
    attn_dropout: float = 0.0,
    norm_eps: float = 1e-5,
    rope_base: int = 10_000,
) -> TransformerDecoder:
    """
    Build the decoder associated with the mistral model. This includes:
    - Token embeddings
    - num_layers number of TransformerSelfAttentionLayer blocks
    - RMS Norm layer applied to the output of the transformer
    - Final projection into token space

    This does NOT currently include inference-time optimizations such as
    sliding-window attention

    Args:
        vocab_size (int): number of tokens in vocabulary.
        num_layers (int): number of layers in the transformer decoder.
        num_heads (int): number of query heads. For MHA this is also the
            number of heads for key and value
        num_kv_heads (int): number of key and value heads. User should ensure
            `num_heads` % `num_kv_heads` == 0. For standard MHA set `num_kv_heads` == `num_heads`,
            for GQA `num_kv_heads` < `num_heads`, and for MQA set `num_kv_heads` == 1.
        embed_dim (int): embedding dimension for self-attention
        intermediate_dim (int): intermediate dimension for MLP
        max_seq_len (int): maximum sequence length the model will be run with,
        attn_dropout (float): dropout value passed onto scaled_dot_product_attention.
            Default: 0.0
        norm_eps (float): epsilon in RMS norms
        rope_base (int): base for the rotary positional embeddings. Default: 10_000

    Returns:
        TransformerDecoder: Instantiation of mistral model.
    """
    head_dim = embed_dim // num_heads
    num_kv_heads = num_kv_heads if num_kv_heads else num_heads

    rope = RotaryPositionalEmbeddings(
        dim=head_dim, max_seq_len=max_seq_len, base=rope_base
    )
    self_attn = MultiHeadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False),
        k_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
        v_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
        output_proj=nn.Linear(embed_dim, embed_dim, bias=False),
        pos_embeddings=rope,
        kv_cache=None,
        max_seq_len=max_seq_len,
        attn_dropout=attn_dropout,
    )
    mlp = mistral_mlp(dim=embed_dim, hidden_dim=intermediate_dim)
    layer = TransformerSelfAttentionLayer(
        attn=self_attn,
        mlp=mlp,
        sa_norm=RMSNorm(dim=embed_dim, eps=norm_eps),
        mlp_norm=RMSNorm(dim=embed_dim, eps=norm_eps),
    )
    tok_embeddings = nn.Embedding(vocab_size, embed_dim)
    return TransformerDecoder(
        tok_embeddings=tok_embeddings,
        layers=layer,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        norm=RMSNorm(embed_dim, eps=norm_eps),
        output=output_proj,
    )


def compare_mistral_classifier(
    bsz: int,
    seq_len: int,
    num_classes: int,
    vocab_size: int,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    embed_dim: int,
    intermediate_dim: int,
    max_seq_len: int,
):

    # setting up the right seed for generating outputs
    torch.manual_seed(16)

    # generate input tensor to be used by both implementations
    x = torch.randint(low=0, high=vocab_size, size=(bsz, seq_len))

    # our implementation
    classifier = mistral_classifier(
        num_classes=num_classes,
        vocab_size=vocab_size,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        embed_dim=embed_dim,
        intermediate_dim=intermediate_dim,
        max_seq_len=max_seq_len,
    )
    fixed_init_model(classifier)

    with torch.no_grad():
        out = classifier(x)

    # reference implementation: manually specify nn.Linear after base mistral
    output_proj = nn.Linear(embed_dim, num_classes, bias=False)
    classifier_ref = mistral(
        vocab_size=vocab_size,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        embed_dim=embed_dim,
        intermediate_dim=intermediate_dim,
        max_seq_len=max_seq_len,
        output_proj=output_proj,
    )

    fixed_init_model(classifier_ref)

    with torch.no_grad():
        out_ref = classifier_ref(x)

    print(
        f"output layer: {classifier.output}\n reference output layer: {classifier_ref.output}"
    )
    print(f"output mean: {out.mean()}\n reference output mean: {out_ref.mean()}")
    print(f"output shape: {out.shape}\n reference output shape: {out_ref.shape}")

    # output tensors should be similar within precision tolerance
    torch.testing.assert_close(out, out_ref, atol=1e-5, rtol=1e-3)
    assert out.shape == (bsz, seq_len, num_classes)


if __name__ == "__main__":
    # (bsz, embed_dim, seq_len, n_classes) # expected
    test_cases = [
        (2, 64, 64, 2),  # 22.6879
        (64, 128, 256, 200),  # 36.8238
        (1, 256, 512, 1),  # 110.2561
    ]
    for bsz, embed_dim, seq_len, n_classes in test_cases:
        compare_mistral_classifier(
            bsz,
            seq_len,
            n_classes,
            vocab_size=32000,
            num_layers=4,
            num_heads=16,
            num_kv_heads=8,
            embed_dim=embed_dim,
            intermediate_dim=512,
            max_seq_len=2048,
        )
