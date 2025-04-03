# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

import pytest

import torch

import torch.distributed
from tests.test_utils import fixed_init_model, gpu_test, mps_ignored_test
from torch import nn
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed._tensor import DTensor, Replicate

# from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torchao.dtypes.nf4tensor import NF4Tensor, to_nf4
from torchtune import training
from torchtune.modules.common_utils import reparametrize_as_dtype_state_dict_post_hook
from torchtune.modules.feed_forward import FeedForward
from torchtune.modules.peft import DoRALinear
from torchtune.training.seed import set_seed

RANK = 4
ALPHA = 1.0
BSZ = 2
SEQ_LEN = 32
EXPECTED_VAL = 0.05201


@pytest.fixture(autouse=True)
def random():
    set_seed(16)


class TestDoRALinear:
    """
    Class for testing our DoRALinear implementation. Expected values are computed
    from the reference implementation and calculated in scripts/compare_lora.py.
    """

    @pytest.fixture
    def in_dim(self) -> int:
        return 64

    @pytest.fixture
    def out_dim(self) -> int:
        return 128

    @pytest.fixture
    def inputs(self, in_dim) -> torch.Tensor:
        inputs = torch.randn(BSZ, SEQ_LEN, in_dim)
        return inputs

    @pytest.fixture
    def dora_linear(self, in_dim, out_dim):
        def create_dora_linear(
            use_bias,
            dtype,
            should_init=True,
            in_dim=in_dim,
            out_dim=out_dim,
        ):
            with training.set_default_dtype(dtype):
                dora_linear = DoRALinear(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    rank=RANK,
                    alpha=ALPHA,
                    use_bias=use_bias,
                )
                if should_init:
                    fixed_init_model(dora_linear)
            return dora_linear

        return create_dora_linear

    @pytest.fixture
    def qdora_linear(self):
        def create_qdora_linear(
            use_bias=False,
            dtype=torch.bfloat16,
            in_dim=512,
            out_dim=512,
            quantize_base=True,
            **quantization_kwargs,
        ):
            with training.set_default_dtype(dtype):
                qdora_linear = DoRALinear(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    rank=RANK,
                    alpha=ALPHA,
                    use_bias=use_bias,
                    quantize_base=quantize_base,
                    **quantization_kwargs,
                )
                fixed_init_model(qdora_linear)
            return qdora_linear

        return create_qdora_linear

    def test_forward(self, inputs, dora_linear, out_dim) -> None:
        dora_linear = dora_linear(use_bias=False, dtype=torch.float32)
        expected = torch.tensor(EXPECTED_VAL)
        actual = dora_linear(inputs)
        assert actual.shape == (BSZ, SEQ_LEN, out_dim)
        torch.testing.assert_close(actual.mean(), expected, atol=1e-4, rtol=1e-6)

    @pytest.mark.parametrize("use_bias", [True, False])
    def test_dora_weight_nf4_when_quantized(self, use_bias, qdora_linear):
        qdora_linear = qdora_linear(use_bias=use_bias, dtype=torch.bfloat16)
        assert isinstance(qdora_linear.weight, NF4Tensor)
        if use_bias:
            assert not isinstance(qdora_linear.bias, NF4Tensor)
            assert qdora_linear.bias.dtype == torch.bfloat16

    def test_dora_weight_nf4_when_quantized_with_quantization_kwargs(
        self, qdora_linear
    ):
        qdora_linear = qdora_linear(
            use_bias=True, dtype=torch.bfloat16, block_size=8, scaler_block_size=4
        )
        assert isinstance(qdora_linear.weight, NF4Tensor)
        assert qdora_linear.weight.block_size == 8
        assert qdora_linear.weight.scaler_block_size == 4
        assert not isinstance(qdora_linear.bias, NF4Tensor)

    def test_dora_weight_nf4_when_quantized_raises_value_error_with_bad_args(
        self, qdora_linear
    ):
        """Ensure that if quantize_base is False, but we pass in quantization kwargs,
        we raise a ValueError."""
        with pytest.raises(
            ValueError,
            match="``quantize_base`` is False, but received the following quantization arguments",
        ):
            qdora_linear(
                use_bias=True,
                dtype=torch.bfloat16,
                block_size=8,
                scaler_block_size=4,
                quantize_base=False,
            )

    # Note: with bfloat16 F.linear(x, weight, bias) != F.linear(x, weight) + bias.
    # This means we would get different results (irrespective of QDoRA).
    # So we leave that test case out
    @pytest.mark.parametrize(
        "use_bias, dtype",
        [(False, torch.bfloat16), (True, torch.float32), (False, torch.float32)],
    )
    @mps_ignored_test()
    def test_qdora_parity(self, use_bias, dtype, dora_linear, qdora_linear):
        with training.set_default_dtype(dtype):
            qdora_linear = qdora_linear(
                use_bias=use_bias, dtype=dtype, in_dim=512, out_dim=512
            )
            dora_linear = dora_linear(
                use_bias=use_bias, dtype=dtype, in_dim=512, out_dim=512
            )

        # set weight of dora_linear to unquantized weight of qdora_linear and check
        # parity.
        dora_linear.weight.data = qdora_linear.weight.to(dtype)
        if use_bias:
            dora_linear.bias.data = qdora_linear.bias.detach().clone()
        qdora_linear.initialize_dora_magnitude()
        dora_linear.initialize_dora_magnitude()

        # Ensure forward passes are the same. This is because DoRALinear should use a special
        # quantized linear operator that runs compute in higher prec (but only saves the 4 bit quantized tensor)
        # for autograd.
        inputs = torch.randn(BSZ, SEQ_LEN, 512, dtype=dtype)
        torch.manual_seed(0)
        dora_linear_out = dora_linear(inputs)
        torch.manual_seed(0)
        qdora_linear_out = qdora_linear(inputs)
        torch.testing.assert_close(dora_linear_out, qdora_linear_out)

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    def test_quantized_state_dict(self, dtype):
        with training.set_default_dtype(dtype):
            dora_linear = DoRALinear(
                in_dim=512,
                out_dim=512,
                rank=RANK,
                alpha=ALPHA,
                use_bias=False,
                quantize_base=True,
            )

        dora_linear._register_state_dict_hook(
            partial(
                reparametrize_as_dtype_state_dict_post_hook,
                dtype=dtype,
                offload_to_cpu=False,
            )
        )
        sd = dora_linear.state_dict()
        # No nf4 tensors, all have type dtype
        for v in sd.values():
            assert v.dtype == dtype
            assert not isinstance(v, NF4Tensor)

        # Load back in results in re-quant and creates the same nf4 tensor.
        # This also ensures that DoRALinear can load a bf16 state_dict.
        dora_linear_reload = DoRALinear(
            in_dim=512,
            out_dim=512,
            rank=RANK,
            alpha=ALPHA,
            use_bias=False,
            quantize_base=True,
        )
        # Zero out weight to verify reloading works
        dora_linear_reload.weight = nn.Parameter(
            to_nf4(
                torch.zeros_like(
                    dora_linear.weight.get_original_weight(),
                    dtype=dtype,
                    device=dora_linear.weight.device,
                )
            )
        )
        # nf4 tensors should be different
        assert not torch.allclose(
            dora_linear.weight.quantized_data, dora_linear_reload.weight.quantized_data
        )
        # but should be the same after loading
        dora_linear_reload.load_state_dict(sd)
        assert torch.allclose(
            dora_linear.weight.quantized_data, dora_linear_reload.weight.quantized_data
        )

    def test_dora_single_device_init(self, dora_linear):
        dora_linear = dora_linear(
            use_bias=False, dtype=torch.float32, should_init=False
        )

        # Randomly initialize LoRA A and B weights to some nonzero value
        dora_linear.lora_a.weight = nn.Parameter(
            torch.randn_like(dora_linear.lora_a.weight)
        )
        dora_linear.lora_b.weight = nn.Parameter(
            torch.randn_like(dora_linear.lora_b.weight)
        )

        expected_magnitude = torch.linalg.norm(
            dora_linear.weight
            + dora_linear.scaling
            * dora_linear.lora_b.weight
            @ dora_linear.lora_a.weight,
            dim=1,
        )
        assert not torch.allclose(dora_linear.magnitude, expected_magnitude)
        dora_linear.initialize_dora_magnitude()
        assert torch.allclose(dora_linear.magnitude, expected_magnitude)

    def test_dora_meta_device_init_error(self):
        with torch.device("meta"):
            dora_linear = DoRALinear(
                in_dim=512,
                out_dim=512,
                rank=RANK,
                alpha=ALPHA,
                use_bias=False,
                quantize_base=False,
            )
        with pytest.raises(RuntimeError, match="Cannot initialize DoRA magnitude"):
            dora_linear.initialize_dora_magnitude()


class TestDistributedDoRALinear(MultiProcessTestCase):
    @property
    def world_size(self) -> int:
        return 2

    @property
    def embed_dim(self):
        return 128

    def setUp(self):
        super().setUp()

    @gpu_test(gpu_count=2)
    def test_dora_distributed_init(self):
        torch.cuda.set_device(f"cuda:{self.rank}")
        self.run_subtests(
            {
                "load_dora_weights": [True, False],
            },
            self._test_dora_distributed_init,
        )

    def _test_dora_distributed_init(self, load_dora_weights):
        rank = self.rank
        is_rank_zero = rank == 0
        device = f"cuda:{rank}"
        layers = ["w1", "w2", "w3"]
        base_model_state_dict = {
            "w1.weight": torch.randn(self.embed_dim, self.embed_dim),
            "w2.weight": torch.randn(self.embed_dim, self.embed_dim),
            "w3.weight": torch.randn(self.embed_dim, self.embed_dim),
        }

        adapter_state_dict = {
            "w1.lora_a.weight": torch.randn(RANK, self.embed_dim),
            "w1.lora_b.weight": torch.randn(self.embed_dim, RANK),
            "w1.magnitude": torch.randn(self.embed_dim),
            "w2.lora_a.weight": torch.randn(RANK, self.embed_dim),
            "w2.lora_b.weight": torch.randn(self.embed_dim, RANK),
            "w2.magnitude": torch.randn(self.embed_dim),
            "w3.lora_a.weight": torch.randn(RANK, self.embed_dim),
            "w3.lora_b.weight": torch.randn(self.embed_dim, RANK),
            "w3.magnitude": torch.randn(self.embed_dim),
        }

        # Define an FFN containing 3 DoRALinear layers and instantiate on meta device
        with torch.device("meta"):
            linears = [
                DoRALinear(
                    in_dim=self.embed_dim,
                    out_dim=self.embed_dim,
                    rank=RANK,
                    alpha=ALPHA,
                    use_bias=False,
                    quantize_base=False,
                )
                for _ in range(3)
            ]
            ffn = FeedForward(
                gate_proj=linears[0],
                down_proj=linears[1],
                up_proj=linears[2],
            )

        # Shard the FFN
        fully_shard(ffn)

        # Assert that everything is on meta device to start
        if is_rank_zero:
            for dora_linear in [ffn.w1, ffn.w2, ffn.w3]:
                assert dora_linear.weight.is_meta
                assert dora_linear.lora_a.weight.is_meta
                assert dora_linear.lora_b.weight.is_meta
                assert dora_linear.magnitude.is_meta

        # Optionally load adapter weights (as though we are resuming from checkpoint)
        # Now lora_a, lora_b, and magnitude should not be on meta device, but base weight should be.
        if load_dora_weights:
            training.load_from_full_model_state_dict(
                ffn,
                adapter_state_dict,
                device,
            )
            if is_rank_zero:
                for dora_linear in [ffn.w1, ffn.w2, ffn.w3]:
                    assert dora_linear.weight.is_meta
                    assert not dora_linear.lora_a.weight.is_meta
                    assert not dora_linear.lora_b.weight.is_meta
                    assert not dora_linear.magnitude.is_meta

        # If not loading adapter weights, initialize LoRA params as usual
        if not load_dora_weights:
            for m in ffn.modules():
                if isinstance(m, DoRALinear):
                    m.to_empty(device=device)
                    m.initialize_parameters()

        # At this point (assuming load_dora_weights=False) we should have
        # zero-initialized LoRA B, Kaiming-uniform initialized LoRA A, and magnitude off meta device
        if is_rank_zero:
            for dora_linear in [ffn.w1, ffn.w2, ffn.w3]:
                assert dora_linear.weight.is_meta
                assert not dora_linear.lora_a.weight.is_meta
                assert not dora_linear.lora_b.weight.is_meta
                assert not dora_linear.magnitude.is_meta

        # Load base model weights
        training.load_from_full_model_state_dict(
            ffn,
            base_model_state_dict,
            device,
        )

        # After this, everything should be off meta device
        if is_rank_zero:
            for dora_linear in [ffn.w1, ffn.w2, ffn.w3]:
                assert not dora_linear.weight.is_meta
                assert not dora_linear.lora_a.weight.is_meta
                assert not dora_linear.lora_b.weight.is_meta
                assert not dora_linear.magnitude.is_meta

        # Finally, initialize the magnitudes
        for m in ffn.modules():
            if hasattr(m, "initialize_dora_magnitude"):
                m.initialize_dora_magnitude()

        # Explicitly check that the magnitudes match their expected value
        for layer in ["w1", "w2", "w3"]:
            weight = base_model_state_dict[f"{layer}.weight"]
            if load_dora_weights:
                weight += (
                    (ALPHA / RANK)
                    * adapter_state_dict[f"{layer}.lora_b.weight"]
                    @ adapter_state_dict[f"{layer}.lora_a.weight"]
                )
            expected_magnitude = torch.linalg.norm(weight, axis=1).to(device=device)
            actual_magnitude = getattr(ffn, layer).magnitude.full_tensor()
            # to explicit replicate the tensor before comparing with DTensor
            if isinstance(expected_magnitude, DTensor):
                device_mesh = torch.distributed.init_device_mesh("cuda", (2,))
                actual_magnitude = DTensor.from_local(
                    actual_magnitude,
                    device_mesh=device_mesh,
                    placements=[Replicate()],
                )
            torch.testing.assert_close(expected_magnitude, actual_magnitude)
