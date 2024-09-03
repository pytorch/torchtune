# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import copy
from itertools import chain

import pytest
import torch
import torch.nn as nn
from packaging import version
from tests.test_utils import gpu_test, single_box_init
from torch.distributed import launcher

from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointWrapper,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from torch.testing._internal.common_fsdp import FSDPTest, MLP
from torchao.dtypes.nf4tensor import NF4Tensor
from torchtune import modules, training
from torchtune.models.llama2._component_builders import llama2, lora_llama2
from torchtune.models.llama3._component_builders import llama3
from torchtune.modules import TransformerSelfAttentionLayer
from torchtune.modules.peft import (
    DoRALinear,
    get_adapter_params,
    LoRALinear,
    set_trainable_params,
)


class TestDistributed:
    def test_init_distributed(self) -> None:
        """Integration test to confirm consistency across device initialization utilities."""
        distributed = training.init_distributed()
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
            training.init_distributed(backend="gloo")
        if not torch.distributed.is_initialized():
            raise AssertionError("Expected torch.distributed to be initialized")
        pg_backend = torch.distributed.get_backend()
        assert (
            pg_backend == "gloo"
        ), f"Expected 'gloo' backend, but received {pg_backend}"

    @staticmethod
    def _test_world_size_with_cpu_device(expected_world_size: int) -> None:
        training.init_distributed(backend="gloo")
        world_size, _ = training.get_world_size_and_rank()
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

    def test_validate_no_params_on_meta_device(self) -> None:
        with torch.device("meta"):
            model = torch.nn.Linear(3, 3)

        with pytest.raises(RuntimeError, match="Unexpected param or buffer"):
            training.validate_no_params_on_meta_device(model)

        # Test model with only buffer
        model = torch.nn.Linear(3, 3)
        buffer = torch.ones(1, device="meta")
        model.register_buffer("buffer", buffer)

        with pytest.raises(RuntimeError, match="Unexpected param or buffer"):
            training.validate_no_params_on_meta_device(model)

    def test_get_fsdp_wrap_policies(self) -> None:
        with single_box_init():
            llama3_policy = training.get_full_finetune_fsdp_wrap_policy(
                memory_efficient_fsdp_wrap=True,
                modules_to_wrap={modules.TransformerSelfAttentionLayer},
            )
            l3 = llama3(
                vocab_size=64,
                num_layers=1,
                num_heads=4,
                num_kv_heads=4,
                embed_dim=64,
                max_seq_len=128,
            )
            wrapped_l3 = FSDP(
                l3, auto_wrap_policy=llama3_policy, device_id=torch.device("cpu")
            )
            # Ensure embedding, output proj, and transformer decoder blocks are wrapped
            assert isinstance(wrapped_l3.tok_embeddings, FSDP)
            assert isinstance(wrapped_l3.output, FSDP)
            for layer in wrapped_l3.layers:
                assert isinstance(layer, FSDP)

            llama2_policy = training.get_full_finetune_fsdp_wrap_policy(
                memory_efficient_fsdp_wrap=False,
                modules_to_wrap={modules.TransformerSelfAttentionLayer},
            )
            l2 = llama2(
                vocab_size=64,
                num_layers=1,
                num_heads=4,
                num_kv_heads=4,
                embed_dim=64,
                max_seq_len=128,
            )
            wrapped_l2 = FSDP(
                l2, auto_wrap_policy=llama2_policy, device_id=torch.device("cpu")
            )
            # Ensure embedding, output proj, and transformer decoder blocks are not wrapped
            assert not isinstance(wrapped_l2.tok_embeddings, FSDP)
            assert not isinstance(wrapped_l2.output, FSDP)
            # Ensure transformer decoder blocks are wrapped
            for layer in wrapped_l2.layers:
                assert isinstance(layer, FSDP)


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
        if isinstance(module, LoRALinear) or isinstance(module, DoRALinear):
            num_nested_linears = len(
                [m for m in module.modules() if isinstance(m, nn.Linear)]
            )
            num_lora_ab += num_nested_linears
        if isinstance(module, TransformerSelfAttentionLayer):
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
            lora_wrap_policy = training.lora_fsdp_wrap_policy(
                modules_to_wrap={TransformerSelfAttentionLayer}
            )
            training.prepare_model_for_fsdp_with_meta_device(model)
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
                if isinstance(m, LoRALinear) or isinstance(m, DoRALinear):
                    torch.testing.assert_close(
                        m.lora_b.weight, torch.zeros_like(m.lora_b.weight)
                    )
            # Total # FSDP modules should be num_transformer + num_lora_ab + 1
            total_fsdp_submodules = len([m for m in FSDP.fsdp_modules(wrapped_lora)])
            assert total_fsdp_submodules == (num_lora_ab + num_transformer_layers + 1)
            # LoRA a & b linears should be individually wrapped.
            # And TransformerSelfAttentionLayers should be individually wrapped.
            for fsdp_submodule in FSDP.fsdp_modules(wrapped_lora):
                if isinstance(fsdp_submodule.module, nn.Linear):
                    num_lora_ab -= 1
                elif isinstance(fsdp_submodule.module, TransformerSelfAttentionLayer):
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
                lora_alpha=8,
            )
        training.prepare_model_for_fsdp_with_meta_device(lora)
        for m in lora.modules():
            m.to_empty(device=torch.device("cpu"), recurse=False)
            m.reset_parameters()
        # No params should be left on meta device
        for n, p in lora.named_parameters():
            assert not p.is_meta, f"parameter {n} is still on meta device!"
        # Neither should buffers
        for n, b in lora.named_buffers():
            assert not b.is_meta, f"buffer {n} is still on meta device!"


class TestFullyShardState(FSDPTest):
    @property
    def world_size(self) -> int:
        return 2

    @gpu_test(gpu_count=2)
    @pytest.mark.skipif(
        version.parse(torch.__version__).base_version < "2.4.0",
        reason="torch >= 2.4 required",
    )
    def test_lora_state_dict(self):
        rank = self.rank
        is_rank_zero = rank == 0
        mlp_dim = 4
        epochs = 5
        torch.manual_seed(42)
        # base_model is simple DDP
        with torch.device("cuda"):
            base_model = nn.Sequential(
                MLP(mlp_dim),
                nn.Sequential(MLP(mlp_dim), nn.Linear(mlp_dim, mlp_dim)),
                MLP(mlp_dim),
            )
            base_optim = torch.optim.Adam(
                base_model.parameters(), weight_decay=0.01, lr=0.01
            )

        fsdp_model_to_save = copy.deepcopy(base_model)
        for module in fsdp_model_to_save:
            fully_shard(module)
        fully_shard(fsdp_model_to_save)
        fsdp_optim_to_save = torch.optim.Adam(
            fsdp_model_to_save.parameters(), weight_decay=0.01, lr=0.01
        )

        # inp is different for each rank
        torch.manual_seed(42 + rank)

        # test get full state dict
        for _ in range(epochs):
            inp = torch.randn((2, mlp_dim), device="cuda")
            base_model(inp).sum().backward()
            for param in base_model.parameters():
                torch.distributed.all_reduce(
                    param.grad, op=torch.distributed.ReduceOp.AVG
                )
            base_optim.step()
            base_optim.zero_grad()
            fsdp_model_to_save(inp).sum().backward()
            fsdp_optim_to_save.step()
            fsdp_optim_to_save.zero_grad()
        expected_model_sd = base_model.state_dict()
        expected_optim_sd = base_optim.state_dict()
        model_full_sd = training.get_full_model_state_dict(
            fsdp_model_to_save, is_rank_zero
        )
        optim_full_sd = training.get_full_optimizer_state_dict(
            fsdp_optim_to_save,
            is_rank_zero,
        )
        if is_rank_zero:
            self.assertEqual(set(model_full_sd.keys()), set(expected_model_sd.keys()))
            for key, value in model_full_sd.items():
                self.assertEqual(value, expected_model_sd[key])
            self.assertEqual(len(optim_full_sd["param_groups"]), 1)
            self.assertEqual(
                len(optim_full_sd["param_groups"]),
                len(expected_optim_sd["param_groups"]),
            )
            self.assertEqual(
                len(optim_full_sd["param_groups"][0].keys()),
                len(expected_optim_sd["param_groups"][0].keys()),
            )
            for key, value in optim_full_sd["param_groups"][0].items():
                if key == "params":
                    self.assertEqual(
                        len(value), len(expected_optim_sd["param_groups"][0][key])
                    )
                else:
                    self.assertEqual(value, expected_optim_sd["param_groups"][0][key])
            self.assertEqual(
                len(optim_full_sd["state"].keys()),
                len(expected_optim_sd["state"].keys()),
            )
            for actual, expected in zip(
                optim_full_sd["state"].values(), expected_optim_sd["state"].values()
            ):
                self.assertEqual(actual, expected)
        else:
            self.assertEqual(len(model_full_sd), 0)
            self.assertEqual(len(optim_full_sd), 0)

        # test set full state dict
        with torch.device("meta"):
            fsdp_model_to_load = nn.Sequential(
                MLP(mlp_dim),
                nn.Sequential(MLP(mlp_dim), nn.Linear(mlp_dim, mlp_dim)),
                MLP(mlp_dim),
            )
        for module in fsdp_model_to_load:
            fully_shard(module)
        fully_shard(fsdp_model_to_load)
        training.load_from_full_model_state_dict(
            fsdp_model_to_load,
            copy.deepcopy(base_model.state_dict()),
            torch.device("cuda"),
            is_rank_zero,
        )
        fsdp_optim_to_load = torch.optim.Adam(
            fsdp_model_to_load.parameters(), weight_decay=0.01, lr=0.01
        )
        training.load_from_full_optimizer_state_dict(
            fsdp_optim_to_load,
            # mimic mmap=True where every rank see full SD
            copy.deepcopy(self._broadcast_full_state_dict(optim_full_sd)),
            torch.device("cuda"),
        )
        for _ in range(epochs):
            inp = torch.randn((2, mlp_dim), device="cuda")
            fsdp_model_to_load(inp).sum().backward()
            fsdp_model_to_save(inp).sum().backward()
            fsdp_optim_to_load.step()
            fsdp_optim_to_save.step()
            fsdp_optim_to_load.zero_grad()
            fsdp_optim_to_save.zero_grad()
        sharded_optim_sd = fsdp_optim_to_load.state_dict()
        expected_sharded_optim_sd = fsdp_optim_to_save.state_dict()
        self.assertEqual(
            sharded_optim_sd["param_groups"],
            expected_sharded_optim_sd["param_groups"],
        )
        self.assertEqual(
            set(sharded_optim_sd["state"].keys()),
            set(expected_sharded_optim_sd["state"].keys()),
        )
        for key, value in sharded_optim_sd["state"].items():
            self.assertEqual(value, expected_sharded_optim_sd["state"][key])

        sharded_model_sd = fsdp_model_to_load.state_dict()
        expected_sharded_model_sd = fsdp_model_to_save.state_dict()
        self.assertEqual(
            set(sharded_model_sd.keys()), set(expected_sharded_model_sd.keys())
        )
        for key, value in sharded_model_sd.items():
            self.assertEqual(value, expected_sharded_model_sd[key])

    @pytest.mark.skipif(
        version.parse(torch.__version__).base_version < "2.4.0",
        reason="torch >= 2.4 required",
    )
    @gpu_test(gpu_count=2)
    def test_qlora_state_dict(self):
        self.run_subtests(
            {
                "enable_activation_checkpointing": [False, True],
            },
            self._test_qlora_state_dict,
        )

    def _test_qlora_state_dict(self, enable_activation_checkpointing: bool):
        is_rank_zero = self.rank == 0
        torch.manual_seed(42)
        kwargs = {
            "lora_attn_modules": ["q_proj", "v_proj", "k_proj", "output_proj"],
            "apply_lora_to_mlp": True,
            "apply_lora_to_output": False,
            "vocab_size": 1024,
            "num_layers": 3,
            "num_heads": 4,
            "num_kv_heads": 2,
            "embed_dim": 1024,
            "max_seq_len": 64,
            "lora_rank": 4,
            "lora_alpha": 1.0,
            "quantize_base": True,
        }
        # single-device model as groundtruth
        with torch.device("cuda"):
            base_model = lora_llama2(**kwargs)
        set_trainable_params(base_model, get_adapter_params(base_model))
        if enable_activation_checkpointing:
            training.set_activation_checkpointing(
                base_model, auto_wrap_policy={modules.TransformerSelfAttentionLayer}
            )

        # fsdp model for saving state dict
        fsdp_model_to_save = copy.deepcopy(base_model)
        for m in fsdp_model_to_save.modules():
            if enable_activation_checkpointing:
                if isinstance(m, CheckpointWrapper):
                    fully_shard(m)
            else:
                if isinstance(m, modules.TransformerSelfAttentionLayer):
                    fully_shard(m)
        fully_shard(fsdp_model_to_save)

        # one forward pass for lazy init
        torch.manual_seed(42 + self.rank)
        inp = torch.randint(
            low=0,
            high=kwargs["vocab_size"],
            size=(2, kwargs["max_seq_len"]),
            device="cuda",
        )
        base_model(inp)
        fsdp_model_to_save(inp)

        expected_model_sd = {k: v.cpu() for k, v in base_model.state_dict().items()}
        model_full_sd = training.get_full_model_state_dict(
            fsdp_model_to_save, is_rank_zero
        )
        if is_rank_zero:
            self.assertEqual(set(model_full_sd.keys()), set(expected_model_sd.keys()))
            for key, value in model_full_sd.items():
                self.assertEqual(value, expected_model_sd[key])

        # fsdp model for loading tate dict
        torch.manual_seed(42)
        with torch.device("meta"):
            fsdp_model_to_load = lora_llama2(**kwargs)
        set_trainable_params(fsdp_model_to_load, get_adapter_params(fsdp_model_to_load))
        if enable_activation_checkpointing:
            training.set_activation_checkpointing(
                fsdp_model_to_load,
                auto_wrap_policy={modules.TransformerSelfAttentionLayer},
            )
        # init rope since it's not covered in state dict
        for m in fsdp_model_to_load.modules():
            if isinstance(m, modules.RotaryPositionalEmbeddings):
                m.reset_parameters()
        for m in fsdp_model_to_load.modules():
            if enable_activation_checkpointing:
                if isinstance(m, CheckpointWrapper):
                    fully_shard(m)
            else:
                if isinstance(m, modules.TransformerSelfAttentionLayer):
                    fully_shard(m)
        fully_shard(fsdp_model_to_load)
        training.load_from_full_model_state_dict(
            fsdp_model_to_load, expected_model_sd, torch.device("cuda"), is_rank_zero
        )
        fsdp_model_to_load(inp)
        sharded_model_sd = fsdp_model_to_load.state_dict()
        expected_sharded_model_sd = fsdp_model_to_save.state_dict()
        self.assertEqual(
            set(sharded_model_sd.keys()), set(expected_sharded_model_sd.keys())
        )
        for key, value in sharded_model_sd.items():
            if isinstance(value._local_tensor, NF4Tensor):
                self.assertEqual(
                    value._local_tensor.get_original_weight(),
                    expected_sharded_model_sd[key]._local_tensor.get_original_weight(),
                )
            else:
                self.assertEqual(value, expected_sharded_model_sd[key])

    def _broadcast_full_state_dict(self, full_sd):
        result = []
        if torch.distributed.get_rank() == 0:
            result.append(full_sd)
        else:
            result.append(None)
        torch.distributed.broadcast_object_list(result, src=0)
        return result[0]
