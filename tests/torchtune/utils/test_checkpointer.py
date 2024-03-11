# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json

from pathlib import Path
from typing import Tuple

import pytest
import torch
from torch import randn

from torchtune.models import llama2
from torchtune.utils._checkpointing import FullModelHFCheckpointer, ModelType
from torchtune.utils._checkpointing._checkpointer_utils import safe_torch_load
from torchtune.utils.seed import set_seed

_VOCAB_SIZE = 100
_DIM = 64
_HIDDEN_DIM = 256
_NUM_HEADS = 4
_NUM_KV_HEADS = 4


@pytest.fixture(autouse=True)
def random():
    set_seed(16)


class TestHFLlama2FullModelCheckpointer:
    @pytest.fixture
    def weight_dtype(self):
        return torch.float16

    @pytest.fixture
    def state_dict_1(self, weight_dtype):
        """
        State dict for a HF format checkpoint. This state dict is "complete" and
        can be loaded into a TorchTune model once correctly converted.
        """
        state_dict = {
            "model.embed_tokens.weight": randn(_VOCAB_SIZE, _DIM, dtype=weight_dtype),
            "model.layers.0.input_layernorm.weight": randn(_DIM, dtype=weight_dtype),
            "model.layers.0.self_attn.q_proj.weight": randn(
                _DIM, _DIM, dtype=weight_dtype
            ),
            "model.layers.0.self_attn.k_proj.weight": randn(
                _DIM, _DIM, dtype=weight_dtype
            ),
            "model.layers.0.self_attn.v_proj.weight": randn(
                _DIM, _DIM, dtype=weight_dtype
            ),
            "model.layers.0.self_attn.o_proj.weight": randn(
                _DIM, _DIM, dtype=weight_dtype
            ),
            "model.layers.0.post_attention_layernorm.weight": randn(
                _DIM, dtype=weight_dtype
            ),
            "model.layers.0.self_attn.rotary_emb.inv_freq": randn(
                _DIM, dtype=weight_dtype
            ),
            "model.layers.0.mlp.gate_proj.weight": randn(
                _HIDDEN_DIM, _DIM, dtype=weight_dtype
            ),
            "model.layers.0.mlp.down_proj.weight": randn(
                _DIM, _HIDDEN_DIM, dtype=weight_dtype
            ),
            "model.layers.0.mlp.up_proj.weight": randn(
                _HIDDEN_DIM, _DIM, dtype=weight_dtype
            ),
            "model.norm.weight": torch.randn(_DIM, dtype=weight_dtype),
            "lm_head.weight": torch.randn(_VOCAB_SIZE, _DIM, dtype=weight_dtype),
        }
        return state_dict

    @pytest.fixture
    def state_dict_2(self, weight_dtype):
        """
        State dict for a HF format checkpoint. This state dict is "incomplete" and
        should be used along with ``state_dict_1`` to test multi-file checkpointing. Specifically
        it's missing the embedding, norm and lm_head keys.
        """
        state_dict = {
            "model.layers.1.input_layernorm.weight": randn(_DIM, dtype=weight_dtype),
            "model.layers.1.self_attn.q_proj.weight": randn(
                _DIM, _DIM, dtype=weight_dtype
            ),
            "model.layers.1.self_attn.k_proj.weight": randn(
                _DIM, _DIM, dtype=weight_dtype
            ),
            "model.layers.1.self_attn.v_proj.weight": randn(
                _DIM, _DIM, dtype=weight_dtype
            ),
            "model.layers.1.self_attn.o_proj.weight": randn(
                _DIM, _DIM, dtype=weight_dtype
            ),
            "model.layers.1.post_attention_layernorm.weight": randn(
                _DIM, dtype=weight_dtype
            ),
            "model.layers.1.self_attn.rotary_emb.inv_freq": randn(
                _DIM, dtype=weight_dtype
            ),
            "model.layers.1.mlp.gate_proj.weight": randn(
                _HIDDEN_DIM, _DIM, dtype=weight_dtype
            ),
            "model.layers.1.mlp.down_proj.weight": randn(
                _DIM, _HIDDEN_DIM, dtype=weight_dtype
            ),
            "model.layers.1.mlp.up_proj.weight": randn(
                _HIDDEN_DIM, _DIM, dtype=weight_dtype
            ),
        }
        return state_dict

    @pytest.fixture
    def llama2_hf_checkpoints(self, tmp_path, state_dict_1, state_dict_2):
        """
        Fixture which creates two checkpoint files for the Llama2 model. The
        state dict follows the HF_FORMAT for the checkpoint format.

        The state dicts are structured in such a way that both single file and
        multiple file checkpoints can be tested.
            * The first checkpoint contains layer0 + embed + norm + lm_head keys
            and can be tested in isolation
            * The second checkpoint contains all layer1 keys and should be tested
            in the multiple file checkpoint test along with the first checkpoint

        The model corresponds to the following config:
            * vocab_size: 100
            * num_layers: 1 for single checkpoint and 2 for multiple checkpoint
            * num_heads: 4
            * num_kv_heads: 4
            * embed_dim: 64
            * max_seq_len: 128
        """
        checkpoint_file_1 = tmp_path / "llama2_hf_checkpoint_01.pt"
        checkpoint_file_2 = tmp_path / "llama2_hf_checkpoint_02.pt"

        torch.save(state_dict_1, checkpoint_file_1)
        torch.save(state_dict_2, checkpoint_file_2)

        config = {
            "hidden_size": 64,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
        }
        config_file = Path.joinpath(tmp_path, "config.json")
        with config_file.open("w") as f:
            json.dump(config, f)

        return (checkpoint_file_1, checkpoint_file_2)

    @pytest.fixture
    def single_file_checkpointer(
        self, llama2_hf_checkpoints, tmp_path
    ) -> FullModelHFCheckpointer:
        checkpoint_file, _ = llama2_hf_checkpoints
        return FullModelHFCheckpointer(
            checkpoint_dir=tmp_path,
            checkpoint_files=[checkpoint_file],
            model_type=ModelType.LLAMA2,
            output_dir=tmp_path,
        )

    @pytest.fixture
    def multi_file_checkpointer(
        self, llama2_hf_checkpoints, tmp_path
    ) -> FullModelHFCheckpointer:
        checkpoint_file_1, checkpoint_file_2 = llama2_hf_checkpoints
        return FullModelHFCheckpointer(
            checkpoint_dir=tmp_path,
            checkpoint_files=[checkpoint_file_1, checkpoint_file_2],
            model_type=ModelType.LLAMA2,
            output_dir=tmp_path,
        )

    def test_load_save_checkpoint_single_file(
        self,
        single_file_checkpointer: FullModelHFCheckpointer,
        llama2_hf_checkpoints: Tuple[Path, Path],
    ):
        """
        Test ``load_checkpoint`` and ``save_checkpoint`` method within the
        FullModelHFCheckpointer for a single checkpoint file.

        We test:
        * ``load_checkpoint`` loads the right sets of keys
        * Internal state of the checkpointer is correctly updated
        * Converted checkpoint can be loaded into the llama2 TorchTune implementation
        * Saved checkpoint keys match the original checkpoint
        """
        # Read the state dict directly from file using torch.load. This will be the state
        # dict we test against
        checkpoint_file, _ = llama2_hf_checkpoints
        orig_state_dict = safe_torch_load(checkpoint_file)

        # Converted state dict from the checkpointer
        state_dict = single_file_checkpointer.load_checkpoint()

        # Check that we've loaded all the keys; We ignore inv_freq as is standard practice
        assert len(state_dict["model"].keys()) + 1 == len(orig_state_dict.keys())

        # the keys in original state dict should match up with the keys in the weight_map
        for key in orig_state_dict.keys():
            if "inv_freq" in key:
                continue
            assert key in single_file_checkpointer._weight_map

        # loading the state dict into the model implementation should work correctly
        model = llama2.llama2(
            vocab_size=_VOCAB_SIZE,
            num_layers=1,
            num_heads=_NUM_HEADS,
            num_kv_heads=_NUM_KV_HEADS,
            embed_dim=_DIM,
            max_seq_len=128,
        )
        model.load_state_dict(state_dict["model"])

        single_file_checkpointer.save_checkpoint(state_dict, epoch=1)

        # Reload the output checkpoint file and compare to the original checkpoint. This
        # assumes we know what the name of the file is. This is fine, breaking this logic
        # should be something we capture through this test
        output_file = Path.joinpath(checkpoint_file.parent, "hf_model_0001_1.pt")
        output_state_dict = safe_torch_load(output_file)

        # We ignore inv_freq as is standard practice and so output dict will have one less key
        assert len(output_state_dict.keys()) + 1 == len(orig_state_dict.keys())

    def test_save_load_checkpoint_multiple_file(
        self,
        multi_file_checkpointer: FullModelHFCheckpointer,
        llama2_hf_checkpoints: Tuple[Path, Path],
    ):
        """
        Test ``load_checkpoint`` method within the FullModelCheckpointer for multiple
        checkpoint file.

        We test:
        * ``load_checkpoint`` loads the right sets of keys
        * Internal state of the checkpointer is correctly updated
        * Converted checkpoint can be loaded into the llama2 TorchTune implementation
        """
        # Read the state dict directly from files
        checkpoint_file_1, checkpoint_file_2 = llama2_hf_checkpoints
        orig_state_dict_1 = safe_torch_load(checkpoint_file_1)
        orig_state_dict_2 = safe_torch_load(checkpoint_file_2)

        # merged state dict from checkpointer
        state_dict = multi_file_checkpointer.load_checkpoint()

        # We ignore inv_freq as is standard practice
        assert len(state_dict["model"].keys()) + 2 == len(
            orig_state_dict_1.keys()
        ) + len(orig_state_dict_2.keys())

        # the keys in the weight_map should match up with the keys in the weight_map
        for key in orig_state_dict_1.keys():
            if "inv_freq" in key:
                continue
            assert key in multi_file_checkpointer._weight_map

        for key in orig_state_dict_2.keys():
            if "inv_freq" in key:
                continue
            assert key in multi_file_checkpointer._weight_map

        # finally loading into the model should work
        model = llama2.llama2(
            vocab_size=_VOCAB_SIZE,
            num_layers=2,
            num_heads=_NUM_HEADS,
            num_kv_heads=_NUM_KV_HEADS,
            embed_dim=_DIM,
            max_seq_len=128,
        )
        model.load_state_dict(state_dict["model"])

        multi_file_checkpointer.save_checkpoint(state_dict, epoch=1)

        # Reload the output checkpoint file and compare to the original checkpoint. This
        # assumes we know what the name of the file is. This is fine, breaking this logic
        # should be something we capture through this test
        output_file_1 = Path.joinpath(checkpoint_file_1.parent, "hf_model_0001_1.pt")
        output_file_2 = Path.joinpath(checkpoint_file_2.parent, "hf_model_0002_1.pt")
        output_state_dict_1 = safe_torch_load(output_file_1)
        output_state_dict_2 = safe_torch_load(output_file_2)

        assert len(output_state_dict_1.keys()) + 1 == len(orig_state_dict_1.keys())
        assert len(output_state_dict_2.keys()) + 1 == len(orig_state_dict_2.keys())
