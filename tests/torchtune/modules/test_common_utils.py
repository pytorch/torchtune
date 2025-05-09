# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy

import pytest

import torch
from tests.test_utils import fixed_init_model, gpu_test
from torch import nn
from torch.distributed._composable.fsdp import fully_shard
from torch.testing._internal.common_fsdp import FSDPTest
from torchtune.models.llama3_2._component_builders import llama3_2
from torchtune.models.llama3_2_vision._component_builders import (
    llama3_2_vision_decoder,
    llama3_2_vision_encoder,
)
from torchtune.modules import (
    delete_kv_caches,
    disable_kv_cache,
    local_kv_cache,
    TiedLinear,
)
from torchtune.modules.common_utils import resize_token_embeddings, slice_str_to_array
from torchtune.modules.model_fusion import DeepFusionModel
from torchtune.training import gather_cpu_state_dict


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


class TestSliceStrToArray:
    def test_single_index(self):
        assert slice_str_to_array("0", 5) == [True, False, False, False, False]

    def test_slice_with_start_and_end(self):
        assert slice_str_to_array("1:3", 5) == [False, True, True, False, False]

    def test_slice_with_start_and_step(self):
        assert slice_str_to_array("1::2", 5) == [False, True, False, True, False]

    def test_slice_with_start_end_and_step(self):
        assert slice_str_to_array("1:4:2", 5) == [False, True, False, True, False]

    def test_multiple_indices(self):
        assert slice_str_to_array("0,2,4", 6) == [True, False, True, False, True, False]

    def test_out_of_range_index(self):
        with pytest.raises(AssertionError):
            slice_str_to_array("10", 5)

    def test_invalid_slice_format(self):
        with pytest.raises(AssertionError):
            slice_str_to_array("1:2:3:4", 5)


class TestResizeTokenEmbeddings:
    @pytest.fixture
    def model_params(self):
        """Provides common model parameters."""
        return {
            "vocab_size": 100,
            "embed_dim": 16,
        }

    def get_model(self, vocab_size, embed_dim, is_tied):
        model = llama3_2(
            vocab_size=vocab_size,
            num_layers=2,
            num_heads=8,
            num_kv_heads=4,
            embed_dim=embed_dim,
            max_seq_len=512,
            tie_word_embeddings=is_tied,
        )
        fixed_init_model(model, min_val=-1, max_val=1)
        return model

    def test_resize_single_device_untied(self, model_params):
        """Test resizing upwards with nn.Linear output on single device."""
        model = self.get_model(
            vocab_size=model_params["vocab_size"],
            embed_dim=model_params["embed_dim"],
            is_tied=False,
        )
        old_vocab_size = model_params["vocab_size"]
        embed_dim = model_params["embed_dim"]
        new_vocab_size = 120

        old_embed_weight = model.tok_embeddings.weight.data.clone()
        old_output_weight = model.output.weight.data.clone()
        old_embed_requires_grad = model.tok_embeddings.weight.requires_grad
        old_output_requires_grad = model.output.weight.requires_grad

        resize_token_embeddings(model, new_vocab_size)

        assert model.tok_embeddings.num_embeddings == new_vocab_size
        assert model.tok_embeddings.weight.shape == (new_vocab_size, embed_dim)
        assert isinstance(model.output, nn.Linear)
        assert model.output.out_features == new_vocab_size
        assert model.output.weight.shape == (new_vocab_size, embed_dim)

        assert model.tok_embeddings.weight.requires_grad == old_embed_requires_grad
        assert model.output.weight.requires_grad == old_output_requires_grad

        torch.testing.assert_close(
            model.tok_embeddings.weight.data[:old_vocab_size], old_embed_weight
        )
        torch.testing.assert_close(
            model.output.weight.data[:old_vocab_size], old_output_weight
        )

        expected_embed_mean = old_embed_weight.mean(dim=0, keepdim=True)
        torch.testing.assert_close(
            model.tok_embeddings.weight.data[old_vocab_size:],
            expected_embed_mean.repeat(new_vocab_size - old_vocab_size, 1),
        )

        expected_output_mean = old_output_weight.mean(dim=0, keepdim=True)
        torch.testing.assert_close(
            model.output.weight.data[old_vocab_size:],
            expected_output_mean.repeat(new_vocab_size - old_vocab_size, 1),
        )

    def test_resize_single_device_tied(self, model_params):
        """Test resizing upwards with TiedLinear output on single device."""
        model = self.get_model(
            vocab_size=model_params["vocab_size"],
            embed_dim=model_params["embed_dim"],
            is_tied=True,
        )
        old_vocab_size = model_params["vocab_size"]
        embed_dim = model_params["embed_dim"]
        new_vocab_size = 120

        old_embed_weight = model.tok_embeddings.weight.data.clone()
        old_embed_requires_grad = model.tok_embeddings.weight.requires_grad

        resize_token_embeddings(model, new_vocab_size)

        assert model.tok_embeddings.num_embeddings == new_vocab_size
        assert model.tok_embeddings.weight.shape == (new_vocab_size, embed_dim)
        assert isinstance(model.output, TiedLinear)
        assert model.output.tied_module is model.tok_embeddings
        assert model.tok_embeddings.weight.requires_grad == old_embed_requires_grad

        torch.testing.assert_close(
            model.tok_embeddings.weight.data[:old_vocab_size], old_embed_weight
        )

        expected_embed_mean = old_embed_weight.mean(dim=0, keepdim=True)
        torch.testing.assert_close(
            model.tok_embeddings.weight.data[old_vocab_size:],
            expected_embed_mean.repeat(new_vocab_size - old_vocab_size, 1),
        )

    def test_resize_same_size_single_device(self, model_params):
        """Test resizing to the same size (should be no-op)."""
        model = self.get_model(
            vocab_size=model_params["vocab_size"],
            embed_dim=model_params["embed_dim"],
            is_tied=False,
        )
        old_vocab_size = model_params["vocab_size"]
        new_vocab_size = old_vocab_size
        original_model_state = {k: v.clone() for k, v in model.state_dict().items()}

        resize_token_embeddings(model, new_vocab_size)

        assert model.tok_embeddings.num_embeddings == old_vocab_size
        assert model.output.out_features == old_vocab_size

        current_model_state = model.state_dict()
        assert original_model_state.keys() == current_model_state.keys()
        for k in original_model_state:
            torch.testing.assert_close(original_model_state[k], current_model_state[k])

    def test_resize_not_decoder(self, llama_vision_model):
        with pytest.raises(AssertionError):
            resize_token_embeddings(llama_vision_model, 512)


class TestFSDPEmbeddingResize(FSDPTest):
    def setUp(self):
        super().setUp()

    @property
    def world_size(self):
        return 2

    def _test_fsdp_embedding_resize(self, output_type: str, resize_type: str):
        device = torch.device(f"cuda:{self.rank}")
        torch.cuda.set_device(device)
        is_rank_zero = self.rank == 0
        old_vocab_size = 256
        embed_dim = 256

        model = llama3_2(
            vocab_size=old_vocab_size,
            num_layers=2,
            num_heads=8,
            num_kv_heads=4,
            embed_dim=embed_dim,
            max_seq_len=4096,
            tie_word_embeddings=(output_type == "tied"),
        )
        fixed_init_model(model, min_val=-1, max_val=1)
        base_model = copy.deepcopy(model)

        if resize_type == "up":
            new_vocab_size = old_vocab_size + 10
        elif resize_type == "down":
            new_vocab_size = old_vocab_size - 10

        original_embed_weight = base_model.tok_embeddings.weight.data.clone().cpu()
        original_output_weight = None
        if isinstance(base_model.output, nn.Linear):
            original_output_weight = base_model.output.weight.data.clone().cpu()

        for n, m in reversed(list(base_model.named_modules())):
            if isinstance(m, nn.ModuleList):
                continue
            fully_shard(m)
        fsdp_model = base_model

        resize_token_embeddings(fsdp_model, new_vocab_size)

        if output_type == "tied":
            output_layer = fsdp_model.output
            embedding_layer = fsdp_model.tok_embeddings
            assert isinstance(
                output_layer, TiedLinear
            ), f"Rank {self.rank}: Output layer is not TiedLinear with output_type={output_type}"
            assert (
                output_layer.tied_module.weight.data_ptr()
                == embedding_layer.weight.data_ptr()
            ), f"Rank: {self.rank}: TiedLinear is not pointing to the same nn.Embedding after resize"

        gathered_state_dict = gather_cpu_state_dict(
            fsdp_model, is_rank_zero=is_rank_zero
        )

        if is_rank_zero:
            assert gathered_state_dict is not None
            assert "tok_embeddings.weight" in gathered_state_dict
            if output_type == "linear":
                assert "output.weight" in gathered_state_dict
            elif output_type == "tied":
                assert "output.weight" not in gathered_state_dict

            final_embed_weight = gathered_state_dict["tok_embeddings.weight"]
            assert final_embed_weight.shape == (new_vocab_size, embed_dim)

            if output_type == "linear":
                final_output_weight = gathered_state_dict["output.weight"]
                assert final_output_weight.shape == (new_vocab_size, embed_dim)
            else:
                final_output_weight = None

            n = min(old_vocab_size, new_vocab_size)
            torch.testing.assert_close(
                final_embed_weight[:n],
                original_embed_weight[:n],
                msg="Embedding weight preservation failed",
            )
            if output_type == "linear":
                torch.testing.assert_close(
                    final_output_weight[:n],
                    original_output_weight[:n],
                    msg="Untied output weight preservation failed",
                )

            if resize_type == "up":
                expected_embed_mean = original_embed_weight.mean(dim=0, keepdim=True)
                torch.testing.assert_close(
                    final_embed_weight[old_vocab_size:],
                    expected_embed_mean.repeat(new_vocab_size - old_vocab_size, 1),
                    msg="Embedding weight mean initialization failed",
                )
                if output_type == "linear":
                    expected_output_mean = original_output_weight.mean(
                        dim=0, keepdim=True
                    )
                    torch.testing.assert_close(
                        final_output_weight[old_vocab_size:],
                        expected_output_mean.repeat(new_vocab_size - old_vocab_size, 1),
                        msg="Untied output weight mean initialization failed",
                    )
        else:
            assert gathered_state_dict == {}

    @gpu_test(gpu_count=2)
    def test_fsdp_embedding_resize(self):
        self.run_subtests(
            {
                "output_type": ["tied", "linear"],
                "resize_type": ["up", "down"],
            },
            self._test_fsdp_embedding_resize,
        )
