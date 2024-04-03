# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from itertools import chain

import pytest
import torch
import torch.nn as nn

from tests.test_utils import single_box_init
from torch.distributed import launcher
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torchtune import utils
from torchtune.models.llama2._component_builders import lora_llama2
from torchtune.modules import TransformerDecoderLayer
from torchtune.modules.peft import LoRALinear
from torchtune.modules.peft.peft_utils import get_adapter_params, set_trainable_params


class TestDistributed:
    def test_init_distributed(self) -> None:
        """Integration test to confirm consistency across device initialization utilities."""
        distributed = utils.init_distributed()
        assert (
            not distributed
        ), "Should return False as there are no distributed environment variables"

    @staticmethod
    def _test_worker_fn(init_pg_explicit: bool) -> None:
        """
        Integration test to confirm distributed initialization and consistency with process group backend utilities.
        """
        if init_pg_explicit:
            torch.distributed.init_process_group(backend="gloo")
        if not torch.distributed.is_initialized():
            utils.init_distributed(backend="gloo")
        if not torch.distributed.is_initialized():
            raise AssertionError("Expected torch.distributed to be initialized")
        pg_backend = torch.distributed.get_backend()
        assert (
            pg_backend == "gloo"
        ), f"Expected 'gloo' backend, but received {pg_backend}"

    @staticmethod
    def _test_world_size_with_cpu_device(expected_world_size: int) -> None:
        utils.init_distributed(backend="gloo")
        world_size, _ = utils.get_world_size_and_rank()
        if world_size != expected_world_size:
            raise AssertionError(
                f"Expected different world size: received {world_size}, expected {expected_world_size}"
            )

    def _test_launch_worker(
        self,
        get_pet_launch_config,
        num_processes: int,
        init_pg_explicit: bool,
    ) -> None:
        lc = get_pet_launch_config(num_processes)
        launcher.elastic_launch(lc, entrypoint=self._test_worker_fn)(init_pg_explicit)

    def test_init_from_env_no_dup(self, get_pet_launch_config) -> None:
        self._test_launch_worker(get_pet_launch_config, 2, init_pg_explicit=False)
        # trivial test case to ensure test passes with no exceptions
        assert True

    def test_init_from_env_dup(self, get_pet_launch_config) -> None:
        self._test_launch_worker(get_pet_launch_config, 2, init_pg_explicit=True)
        # trivial test case to ensure test passes with no exceptions
        assert True

    def test_world_size_with_cpu(self, get_pet_launch_config) -> None:
        desired_world_size = 4
        lc = get_pet_launch_config(desired_world_size)
        launcher.elastic_launch(lc, entrypoint=self._test_world_size_with_cpu_device)(
            desired_world_size
        )

    def test_default_wrap_fsdp(self) -> None:
        with single_box_init():
            model = nn.Linear(5, 5)
            fsdp_model = utils.wrap_fsdp(
                model, device=torch.device("cpu"), dtype=torch.float32
            )
            # Should create a single FSDP unit with FULL_SHARD
            fsdp_units = [m for m in fsdp_model.modules() if isinstance(m, FSDP)]
            assert len(fsdp_units) == 1

    def test_wrap_fsdp_wrapping(self) -> None:
        with single_box_init():
            model = nn.Sequential(nn.Linear(3, 3), nn.Linear(3, 3))
            orig_num_modules = len([m for m in model.modules()])
            fsdp_model = utils.wrap_fsdp(
                model,
                device=torch.device("cpu"),
                dtype=torch.float32,
                auto_wrap_policy={nn.Linear},
            )
            # Should create orig_num_modules FSDP units.
            fsdp_units = [m for m in fsdp_model.modules() if isinstance(m, FSDP)]
            assert len(fsdp_units) == orig_num_modules

    def test_wrap_fsdp_custom_policy(self) -> None:
        def always_wrap(*args, **kwargs):
            return True

        model = nn.Sequential(
            nn.Linear(3, 3), nn.BatchNorm1d(10), nn.Dropout(0.25), nn.Softmax(dim=1)
        )
        num_modules = len([m for m in model.modules()])
        with single_box_init():
            fsdp_model = utils.wrap_fsdp(
                model,
                device=torch.device("cpu"),
                dtype=torch.float32,
                auto_wrap_policy=always_wrap,
            )
            fsdp_units = [m for m in fsdp_model.modules() if isinstance(m, FSDP)]
            assert len(fsdp_units) == num_modules

    def test_validate_no_params_on_meta_device(self) -> None:
        with torch.device("meta"):
            model = torch.nn.Linear(3, 3)

        with pytest.raises(RuntimeError, match="Unexpected param or buffer"):
            utils.validate_no_params_on_meta_device(model)

        # Test model with only buffer
        model = torch.nn.Linear(3, 3)
        buffer = torch.ones(1, device="meta")
        model.register_buffer("buffer", buffer)

        with pytest.raises(RuntimeError, match="Unexpected param or buffer"):
            utils.validate_no_params_on_meta_device(model)


N_LAYERS = 3
IN_DIM = 5
OUT_DIM = 10
VOCAB_SIZE = 50
NUM_HEADS = 4
NUM_KV_HEADS = 2
EMBED_DIM = 64
MAX_SEQ_LEN = 64


def _get_n_lora_and_tformer_layers(model):
    num_lora_ab = 0
    num_transformer_layers = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            num_nested_linears = len(
                [m for m in module.modules() if isinstance(m, nn.Linear)]
            )
            num_lora_ab += num_nested_linears
        if isinstance(module, TransformerDecoderLayer):
            num_transformer_layers += 1

    return num_lora_ab, num_transformer_layers


# TODO: figure out a permanent home for FSDP + LoRA code
class TestLoRAFSDP:
    def test_lora_fsdp_wrap(self):
        with torch.device("meta"):
            model = lora_llama2(
                lora_attn_modules=["q_proj", "v_proj"],
                vocab_size=VOCAB_SIZE,
                num_layers=N_LAYERS,
                num_heads=NUM_HEADS,
                num_kv_heads=NUM_KV_HEADS,
                embed_dim=EMBED_DIM,
                max_seq_len=MAX_SEQ_LEN,
                lora_rank=4,
                lora_alpha=1.0,
            )

        adapter_params = get_adapter_params(model)
        set_trainable_params(model, adapter_params)
        num_lora_ab, num_transformer_layers = _get_n_lora_and_tformer_layers(model)
        with single_box_init():
            lora_wrap_policy = utils.lora_fsdp_wrap_policy(
                modules_to_wrap={TransformerDecoderLayer}
            )
            utils.prepare_model_for_fsdp_with_meta_device(model)
            wrapped_lora = FSDP(
                model,
                auto_wrap_policy=lora_wrap_policy,
                device_id=torch.device("cpu"),
            )

            # After FSDP wrap, nothing should be left on meta device, and LoRA params
            # should be initialized.
            for p in chain(wrapped_lora.parameters(), wrapped_lora.buffers()):
                assert not p.is_meta

            for m in wrapped_lora.modules():
                if isinstance(m, LoRALinear):
                    torch.testing.assert_close(
                        m.lora_b.weight, torch.zeros_like(m.lora_b.weight)
                    )

            # Total # FSDP modules should be num_transformer + num_lora_ab + 1
            total_fsdp_submodules = len([m for m in FSDP.fsdp_modules(wrapped_lora)])
            assert total_fsdp_submodules == (num_lora_ab + num_transformer_layers + 1)
            # LoRA a & b linears should be individually wrapped.
            # And TransformerDecoderLayers should be individually wrapped.
            for fsdp_submodule in FSDP.fsdp_modules(wrapped_lora):
                if isinstance(fsdp_submodule.module, nn.Linear):
                    num_lora_ab -= 1
                elif isinstance(fsdp_submodule.module, TransformerDecoderLayer):
                    num_transformer_layers -= 1
            assert num_lora_ab == 0
            assert num_transformer_layers == 0

    def test_lora_meta_device_init_fsdp(self):
        with torch.device("meta"):
            lora = lora_llama2(
                lora_attn_modules=["q_proj", "v_proj"],
                vocab_size=VOCAB_SIZE,
                num_layers=N_LAYERS,
                num_heads=NUM_HEADS,
                num_kv_heads=NUM_KV_HEADS,
                embed_dim=EMBED_DIM,
                max_seq_len=MAX_SEQ_LEN,
                lora_rank=4,
                lora_alpha=1.0,
            )
        utils.prepare_model_for_fsdp_with_meta_device(lora)
        for m in lora.modules():
            m.to_empty(device=torch.device("cpu"), recurse=False)
            m.reset_parameters()
        # No params should be left on meta device
        for n, p in lora.named_parameters():
            assert not p.is_meta, f"parameter {n} is still on meta device!"
        # Neither should buffers
        for n, b in lora.named_buffers():
            assert not b.is_meta, f"buffer {n} is still on meta device!"
