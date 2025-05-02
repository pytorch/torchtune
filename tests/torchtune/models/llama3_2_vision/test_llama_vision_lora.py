# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from tests.test_utils import fixed_init_model
from torchtune.models.llama3_2_vision._component_builders import (
    lora_llama3_2_vision_decoder,
    lora_llama3_2_vision_encoder,
)
from torchtune.modules.model_fusion import DeepFusionModel
from torchtune.modules.peft import get_adapter_params, TrainableParams
from torchtune.training.seed import set_seed

EMBED_DIM = 128
NUM_LAYERS = 4
NUM_HEADS = 16
NUM_KV_HEADS = 8
VOCAB_SIZE = 32000
MAX_SEQ_LEN = 2048
BSZ = 2
SEQ_LEN = 100
LORA_ATTN_MODULES = ["q_proj", "k_proj", "v_proj", "output_proj"]
LORA_RANK = 8
LORA_ALPHA = 16
IMAGE_SIZE = 140
PATCH_SIZE = 14


def lora_llama3_2_vision(
    decoder_type,
    encoder_type,
    fusion_type,
) -> DeepFusionModel:
    encoder = lora_llama3_2_vision_encoder(
        encoder_lora=encoder_type == TrainableParams.LORA,
        fusion_lora=fusion_type == TrainableParams.LORA,
        lora_attn_modules=LORA_ATTN_MODULES,
        apply_lora_to_mlp=False,
        apply_lora_to_output=False,
        patch_size=PATCH_SIZE,
        num_heads=NUM_HEADS,
        clip_embed_dim=EMBED_DIM,
        clip_num_layers=NUM_LAYERS,
        clip_hidden_states=[2],
        decoder_embed_dim=EMBED_DIM,
        num_layers_projection=NUM_LAYERS,
        tile_size=IMAGE_SIZE,
        max_num_tiles=1,
        in_channels=3,
        lora_rank=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.0,
        use_dora=False,
        quantize_base=False,
    )
    decoder = lora_llama3_2_vision_decoder(
        decoder_lora=decoder_type == TrainableParams.LORA,
        fusion_lora=fusion_type == TrainableParams.LORA,
        lora_attn_modules=LORA_ATTN_MODULES,
        apply_lora_to_mlp=False,
        apply_lora_to_output=False,
        vocab_size=VOCAB_SIZE,
        num_layers=NUM_LAYERS,
        fusion_interval=2,
        num_special_tokens=8,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        embed_dim=EMBED_DIM,
        max_seq_len=MAX_SEQ_LEN,
        encoder_max_seq_len=2020,  # 20*101
        rope_base=500_000,
        intermediate_dim=14336,
        lora_rank=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.0,
        use_dora=False,
        quantize_base=False,
    )
    return DeepFusionModel(
        encoder=encoder,
        decoder=decoder,
        encoder_trainable=encoder_type != TrainableParams.FROZEN,
        decoder_trainable=decoder_type != TrainableParams.FROZEN,
        fusion_trainable=fusion_type != TrainableParams.FROZEN,
    )


@pytest.fixture(autouse=True)
def random():
    set_seed(16)


class TestLlamaVisionLora:
    @pytest.fixture
    def inputs(self):
        return torch.randint(0, VOCAB_SIZE, (BSZ, SEQ_LEN))

    def test_lora_args(self):
        model = lora_llama3_2_vision(
            TrainableParams.LORA,
            TrainableParams.FROZEN,
            TrainableParams.FROZEN,
        )
        encoder = set(get_adapter_params(model).keys())
        assert len(encoder) == 32, "Only the clip encoder should be trainable."

        model = lora_llama3_2_vision(
            TrainableParams.FROZEN,
            TrainableParams.LORA,
            TrainableParams.FROZEN,
        )
        decoder = set(get_adapter_params(model).keys())
        assert (
            len(decoder) == 32
        ), "Only the decoder self attention layers should be trainable."

        model = lora_llama3_2_vision(
            TrainableParams.FROZEN,
            TrainableParams.FROZEN,
            TrainableParams.LORA,
        )
        fusion = set(get_adapter_params(model).keys())
        assert len(fusion) == 48, "Only the fusion layers should be trainable."

        all_params = set.union(encoder, decoder, fusion)
        assert (
            len(all_params) == 48 + 32 + 32
        ), "There should be no overlap between options."

    def test_forward(self, inputs):
        model = lora_llama3_2_vision(
            TrainableParams.LORA,
            TrainableParams.LORA,
            TrainableParams.LORA,
        )
        fixed_init_model(model, min_val=-0.25, max_val=0.5)
        tokens = torch.randint(0, VOCAB_SIZE, (BSZ, SEQ_LEN))
        image = torch.randn(BSZ, 1, 1, 3, IMAGE_SIZE, IMAGE_SIZE)
        aspect_ratio = torch.tensor([[1, 1] for _ in range(BSZ)])
        actual = model(
            tokens, encoder_input={"images": image, "aspect_ratio": aspect_ratio}
        )
        expected = torch.tensor(-3.9763)
        assert actual.shape == (BSZ, SEQ_LEN, VOCAB_SIZE)
        torch.testing.assert_close(actual.mean(), expected, atol=1e-4, rtol=1e-4)
