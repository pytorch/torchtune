# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import copy

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from packaging import version
from tests.test_utils import gpu_test
from torch.distributed import init_process_group, launcher
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointWrapper,
)
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_fsdp import MLP
from torchao.dtypes.nf4tensor import NF4Tensor
from torchtune import modules, training
from torchtune.models.llama2._component_builders import lora_llama2
from torchtune.models.llama3_1._component_builders import llama3_mlp
from torchtune.models.llama3_1._position_embeddings import Llama3ScaledRoPE
from torchtune.modules import RMSNorm, TransformerSelfAttentionLayer
from torchtune.modules.attention import MultiHeadAttention
from torchtune.modules.peft import (
    DoRALinear,
    get_adapter_params,
    LoRALinear,
    set_trainable_params,
)


class TestDistributed:
    @staticmethod
    def _test_worker_fn(init_pg_explicit: bool) -> None:
        """
        Integration test to confirm distributed initialization and consistency with process group backend utilities.
        """
        if init_pg_explicit:
            torch.distributed.init_process_group(backend="gloo")
        if not torch.distributed.is_initialized():
            init_process_group(backend="gloo")
        if not torch.distributed.is_initialized():
            raise AssertionError("Expected torch.distributed to be initialized")
        pg_backend = torch.distributed.get_backend()
        assert (
            pg_backend == "gloo"
        ), f"Expected 'gloo' backend, but received {pg_backend}"

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

    def test_get_distributed_backend(self) -> None:
        assert training.get_distributed_backend("cuda") == "nccl"
        assert training.get_distributed_backend("cpu") == "gloo"
        assert (
            training.get_distributed_backend("cuda", offload_ops_to_cpu=True)
            == "cuda:nccl,cpu:gloo"
        )


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


class TestFullyShardState(MultiProcessTestCase):
    @property
    def world_size(self) -> int:
        return 2

    @gpu_test(gpu_count=2)
    @pytest.mark.skipif(
        version.parse(torch.__version__).base_version < "2.4.0",
        reason="torch >= 2.4 required",
    )
    def test_lora_state_dict(self):
        torch.cuda.set_device(f"cuda:{self.rank}")  # Set device for this process
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
        model_full_sd = training.gather_cpu_state_dict(fsdp_model_to_save, is_rank_zero)
        optim_full_sd = training.get_full_optimizer_state_dict(
            fsdp_model_to_save,
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
        )
        fsdp_optim_to_load = torch.optim.Adam(
            fsdp_model_to_load.parameters(), weight_decay=0.01, lr=0.01
        )
        training.load_from_full_optimizer_state_dict(
            fsdp_model_to_load,
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
        torch.cuda.set_device(f"cuda:{self.rank}")  # Set device for this process
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
        model_full_sd = training.gather_cpu_state_dict(fsdp_model_to_save, is_rank_zero)
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
            if isinstance(m, modules.RotaryPositionalEmbeddings) or isinstance(
                m, modules.VisionRotaryPositionalEmbeddings
            ):
                m.rope_init()
        for m in fsdp_model_to_load.modules():
            if enable_activation_checkpointing:
                if isinstance(m, CheckpointWrapper):
                    fully_shard(m)
            else:
                if isinstance(m, modules.TransformerSelfAttentionLayer):
                    fully_shard(m)
        fully_shard(fsdp_model_to_load)
        training.load_from_full_model_state_dict(
            fsdp_model_to_load, expected_model_sd, torch.device("cuda")
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


class TestTensorParalell(MultiProcessTestCase):
    @property
    def world_size(self) -> int:
        return 2

    @gpu_test(gpu_count=2)
    def test_prepare_mha_for_tp(self) -> None:
        """Test tensor parallelism preparation for multi-head attention."""
        # Create a device mesh for tensor parallelism
        mesh = dist.init_device_mesh("cuda", mesh_shape=(2,))

        # Parameters for TransformerSelfAttentionLayer
        embed_dim = 64
        hidden_dim = 64
        num_heads = 4
        num_kv_heads = 4
        max_seq_len = 128
        rope_base = 500000
        head_dim = embed_dim // num_heads
        rope = Llama3ScaledRoPE(dim=head_dim, max_seq_len=max_seq_len, base=rope_base)
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
            max_seq_len=max_seq_len,
            attn_dropout=0.0,
        )
        mlp = llama3_mlp(dim=embed_dim, hidden_dim=hidden_dim)
        decoder_layer = TransformerSelfAttentionLayer(
            attn=self_attn,
            mlp=mlp,
            sa_norm=RMSNorm(dim=embed_dim, eps=1e-5),
            mlp_norm=RMSNorm(dim=embed_dim, eps=1e-5),
        )

        orig_num_heads = self_attn.num_heads
        orig_num_kv_heads = self_attn.num_kv_heads
        orig_embed_dim = self_attn.embed_dim

        # Apply tensor parallelism preparation
        decoder_layer = training.prepare_mha_for_tp(decoder_layer, mesh)

        # Verify that parameters were scaled correctly
        assert decoder_layer.attn.num_heads == orig_num_heads // 2
        assert decoder_layer.attn.num_kv_heads == orig_num_kv_heads // 2
        assert decoder_layer.attn.embed_dim == orig_embed_dim // 2
