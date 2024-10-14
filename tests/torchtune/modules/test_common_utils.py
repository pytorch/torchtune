# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch
from tests.test_utils import fixed_init_model
from torchtune.models.llama3_2._component_builders import llama3_2
from torchtune.models.llama3_2_vision._component_builders import (
    llama3_2_vision_decoder,
    llama3_2_vision_encoder,
)
from torchtune.modules import delete_kv_caches, disable_kv_cache, local_kv_cache
from torchtune.modules.model_fusion import DeepFusionModel


@pytest.fixture
def llama_vision_model():
    vision_encoder = llama3_2_vision_encoder(
        clip_embed_dim=32,
        clip_num_layers=4,
        num_heads=4,
        tile_size=49,
        patch_size=9,
        max_num_tiles=4,
        in_channels=3,
        clip_hidden_states=[0, 1],
        num_layers_projection=2,
        decoder_embed_dim=128,
    ).eval()
    vision_decoder = llama3_2_vision_decoder(
        vocab_size=256,
        num_layers=4,
        fusion_interval=2,
        num_special_tokens=2,
        num_heads=8,
        num_kv_heads=4,
        embed_dim=128,
        max_seq_len=4096,
        encoder_max_seq_len=4096,
    ).eval()
    fixed_init_model(vision_encoder, min_val=-1, max_val=1)
    fixed_init_model(vision_decoder, min_val=-1, max_val=1)
    model = DeepFusionModel(
        encoder=vision_encoder,
        decoder=vision_decoder,
        encoder_trainable=False,
        decoder_trainable=False,
        fusion_trainable=False,
    )
    return model


@pytest.fixture
def llama_decoder_model():
    model = llama3_2(
        vocab_size=256,
        num_layers=2,
        num_heads=8,
        num_kv_heads=4,
        embed_dim=256,
        max_seq_len=4096,
    )
    fixed_init_model(model, min_val=-1, max_val=1)
    model.eval()
    return model


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def inputs():
    return torch.randint(low=0, high=256, size=(4, 2048))


@pytest.fixture
def causal_mask():
    return torch.tril(torch.ones((2048, 4096))).unsqueeze(0).repeat(4, 1, 1)


@pytest.fixture
def input_pos():
    return torch.arange(0, 2048).unsqueeze(0).repeat(4, 1)


class TestLocalKVCache:
    @pytest.mark.parametrize("model", ["llama_decoder_model", "llama_vision_model"])
    def test_local_kv_cache(
        self, device, inputs, causal_mask, input_pos, model, request
    ):
        model = request.getfixturevalue(model)
        outs = model(inputs)

        with local_kv_cache(model, batch_size=4, device=device, dtype=torch.float32):
            outs_cached = model(inputs, mask=causal_mask, input_pos=input_pos)
            assert model.caches_are_setup()
            assert model.caches_are_enabled()

        for module in model.modules():
            if hasattr(module, "kv_cache"):
                assert module.kv_cache is None

        assert not model.caches_are_setup()
        assert not model.caches_are_enabled()

        torch.testing.assert_close(
            outs_cached.mean(), outs.mean(), atol=1e-4, rtol=1e-6
        )

    @pytest.mark.parametrize("model", ["llama_decoder_model", "llama_vision_model"])
    def test_local_kv_cache_raises_error_caches_setup(self, device, model, request):

        model = request.getfixturevalue(model)
        model.setup_caches(batch_size=4, dtype=torch.float32)
        with pytest.raises(ValueError, match="Model caches must be not setup"):
            with local_kv_cache(
                model, batch_size=4, device=device, dtype=torch.float32
            ):
                pass


class TestDeleteKVCaches:
    @pytest.mark.parametrize("model", ["llama_decoder_model", "llama_vision_model"])
    def test_delete_kv_cache(self, model, request):
        model = request.getfixturevalue(model)
        model.setup_caches(batch_size=4, dtype=torch.float32)

        delete_kv_caches(model)

        assert not model.caches_are_setup()
        assert not model.caches_are_enabled()

        for module in model.modules():
            if hasattr(module, "kv_cache"):
                assert module.kv_cache is None
                assert not module.cache_enabled

    @pytest.mark.parametrize("model", ["llama_decoder_model", "llama_vision_model"])
    def test_delete_kv_cache_raises_error_without_caches_setup(self, model, request):
        model = request.getfixturevalue(model)
        with pytest.raises(ValueError, match="You have tried to delete model caches"):
            delete_kv_caches(model)


class TestDisableKVCaches:
    @pytest.mark.parametrize("model", ["llama_decoder_model", "llama_vision_model"])
    def test_disable_kv_cache(self, inputs, causal_mask, input_pos, model, request):

        # firstly, setup kv-caches and update the cache state
        model = request.getfixturevalue(model)
        model.setup_caches(batch_size=4, dtype=torch.float32)
        model(inputs, mask=causal_mask, input_pos=input_pos)

        # let's grab this initial cache state for later
        expected_kv_cache_states = []
        for module in model.modules():
            if hasattr(module, "kv_cache") and callable(module.kv_cache):
                expected_kv_cache_states.append(module.kv_cache.k_cache.clone())

        with disable_kv_cache(model):
            assert model.caches_are_setup()
            assert not model.caches_are_enabled()

            # these model forward passes should *not* be updating the cache
            model(inputs)
            model(inputs)

        # grab the cache states after exiting the context manager
        kv_cache_states = []
        for module in model.modules():
            if hasattr(module, "kv_cache") and callable(module.kv_cache):
                assert module.cache_enabled
                kv_cache_states.append(module.kv_cache.k_cache.clone())

        # should be the same!
        for expected, output in zip(expected_kv_cache_states, kv_cache_states):
            assert torch.equal(expected, output)

        assert model.caches_are_setup()
        assert model.caches_are_enabled()

    @pytest.mark.parametrize("model", ["llama_decoder_model", "llama_vision_model"])
    def test_disable_kv_cache_raises_error_caches_not_setup(self, model, request):
        model = request.getfixturevalue(model)
        with pytest.raises(ValueError, match="Model caches must be setup"):
            with disable_kv_cache(model):
                pass
