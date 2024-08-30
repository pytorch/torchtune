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

from torchtune.models import gemma, llama2, mistral
from torchtune.modules.peft import (
    get_adapter_params,
    get_lora_module_names,
    validate_missing_and_unexpected_for_lora,
)

from torchtune.training.checkpointing import FullModelHFCheckpointer
from torchtune.training.checkpointing._utils import (
    ADAPTER_CONFIG,
    ADAPTER_KEY,
    safe_torch_load,
)
from torchtune.training.seed import set_seed

_VOCAB_SIZE = 100
_DIM = 64
_HIDDEN_DIM = 256
_NUM_HEADS = 4
_NUM_KV_HEADS = 4
_HEAD_DIM = 16


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
        can be loaded into a torchtune model once correctly converted.
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
            model_type="LLAMA2",
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
            model_type="LLAMA2",
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
        * Converted checkpoint can be loaded into the llama2 torchtune implementation
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
        * Converted checkpoint can be loaded into the llama2 torchtune implementation
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

    def test_load_save_adapter_only(
        self, tmp_path, single_file_checkpointer, llama2_hf_checkpoints
    ):
        """ """
        state_dict = single_file_checkpointer.load_checkpoint()

        with pytest.raises(
            ValueError, match="Adapter checkpoint not found in state_dict"
        ):
            single_file_checkpointer.save_checkpoint(
                state_dict, epoch=2, adapter_only=True
            )

        state_dict[ADAPTER_KEY] = {}
        single_file_checkpointer.save_checkpoint(state_dict, epoch=2, adapter_only=True)

        output_file_1 = Path.joinpath(tmp_path, "hf_model_0001_2.pt")
        output_file_2 = Path.joinpath(tmp_path, "adapter_2.pt")

        with pytest.raises(ValueError, match="Unable to load checkpoint from"):
            _ = safe_torch_load(output_file_1)

        output_state_dict_2 = safe_torch_load(output_file_2)
        # Check that the empty adapter we saved is the one loaded succesfully
        assert len(output_state_dict_2.keys()) == 0

    def test_save_checkpoint_in_peft_format(
        self,
        single_file_checkpointer: FullModelHFCheckpointer,
        llama2_hf_checkpoints: Tuple[Path, Path],
    ):
        """
        Test save_checkpoint method within the FullModelCheckpointer for
        integration with HF PEFT (i.e. save_in_peft_format=True).

        We test that:
        * The file adapter_config.json contains the fields required by PEFT
        and the correct values
        * The state dict keys of the saved adapter checkpoint are remapped as expected
        * The state dict values of the saved adapter checkpoint (after key remapping)
        match those in torchtune for parameters that are not permuted by HF
        # The state dict values of the saved adapter checkpoint (after key remapping)
        do not match those in torchtune for parameters that are permuted by HF, but the
        sums along the dimension of permutation match
        """

        # Define LoRA params for this test
        lora_attn_modules = ["q_proj", "output_proj"]
        apply_lora_to_mlp = True
        apply_lora_to_output = True
        lora_rank = 4
        lora_alpha = 8

        checkpoint_file, _ = llama2_hf_checkpoints
        state_dict = single_file_checkpointer.load_checkpoint()

        # Build LoRA Llama2 model and load in base model weights
        model = llama2.lora_llama2(
            lora_attn_modules=lora_attn_modules,
            apply_lora_to_mlp=apply_lora_to_mlp,
            apply_lora_to_output=apply_lora_to_output,
            vocab_size=_VOCAB_SIZE,
            num_layers=1,
            num_heads=_NUM_HEADS,
            num_kv_heads=_NUM_KV_HEADS,
            embed_dim=_DIM,
            max_seq_len=128,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )
        missing, unexpected = model.load_state_dict(state_dict["model"], strict=False)
        validate_missing_and_unexpected_for_lora(
            lora_attn_modules=lora_attn_modules,
            apply_lora_to_mlp=apply_lora_to_mlp,
            apply_lora_to_output=apply_lora_to_output,
            base_missing=missing,
            base_unexpected=unexpected,
        )

        # LoRA B params are zero-initialized, randomly initialize them to make
        # the test of their permutation on checkpoint save nontrivial
        lora_b_sd = {
            k: torch.randn_like(v)
            for k, v in model.state_dict().items()
            if "lora_b" in k
        }
        model.load_state_dict(lora_b_sd, strict=False)

        # Construct the adapter weights and config and save using checkpointer
        adapter_params = get_adapter_params(model)
        adapter_key_filter = lambda x: x in adapter_params
        expected_adapter_state_dict = {
            k: v for k, v in model.state_dict().items() if adapter_key_filter(k)
        }
        adapter_config = {
            "r": lora_rank,
            "lora_alpha": lora_alpha,
            "target_modules": get_lora_module_names(
                lora_attn_modules,
                apply_lora_to_mlp,
                apply_lora_to_output,
            ),
            "peft_type": "LORA",
        }
        state_dict.update({ADAPTER_KEY: expected_adapter_state_dict})
        state_dict.update({ADAPTER_CONFIG: adapter_config})
        single_file_checkpointer.save_checkpoint(state_dict, epoch=1)

        # Load saved adapter weights and config from file for comparison
        adapter_weights_file = Path.joinpath(
            checkpoint_file.parent, "adapter_model.bin"
        )
        actual_adapter_state_dict = safe_torch_load(adapter_weights_file)

        adapter_config_file = Path.joinpath(
            checkpoint_file.parent, "adapter_config.json"
        )
        with open(adapter_config_file, "r") as f:
            adapter_config = json.load(f)

        expected_target_modules = [
            "down_proj",
            "gate_proj",
            "lm_head",
            "o_proj",
            "q_proj",
            "up_proj",
        ]
        assert sorted(adapter_config["target_modules"]) == expected_target_modules

        # Map PEFT keys back to torchtune keys
        peft_to_tt = {
            "o_proj": "output_proj",
            "gate_proj": "w1",
            "down_proj": "w2",
            "up_proj": "w3",
            "lm_head": "output",
        }
        for k, v in actual_adapter_state_dict.items():
            new_k = k.replace("base_model.model.", "").replace("self_attn", "attn")
            if "lm_head" not in new_k:
                new_k = new_k.replace("model.", "")
            for kk, vv in peft_to_tt.items():
                if kk in k:
                    new_k = new_k.replace(kk, vv)
            new_k = new_k.replace("lora_A", "lora_a").replace("lora_B", "lora_b")

            # LoRA B matrix for Q should not match due to Q and K permutation
            # However, since they're permuted along embed dim, their sum along that axis should match
            if "lora_b" in new_k and "q_proj" in new_k:
                assert not torch.allclose(
                    actual_adapter_state_dict[k], expected_adapter_state_dict[new_k]
                )
                torch.testing.assert_close(
                    actual_adapter_state_dict[k].sum(dim=0),
                    expected_adapter_state_dict[new_k].sum(dim=0),
                )

            # All other matrices should match exactly
            if "lora_b" not in new_k:
                torch.testing.assert_close(
                    actual_adapter_state_dict[k], expected_adapter_state_dict[new_k]
                )


class TestHFMistralRewardModelFullModelCheckpointer:
    @pytest.fixture
    def weight_dtype(self):
        return torch.float16

    @pytest.fixture
    def state_dict(self, weight_dtype):
        """
        State dict for a HF format mistral reward model checkpoint. This state dict is
        "complete" and can be loaded into a torchtune model once correctly converted.
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
            "model.layers.0.mlp.gate_proj.weight": randn(
                _HIDDEN_DIM, _DIM, dtype=weight_dtype
            ),
            "model.layers.0.mlp.down_proj.weight": randn(
                _DIM, _HIDDEN_DIM, dtype=weight_dtype
            ),
            "model.layers.0.mlp.up_proj.weight": randn(
                _HIDDEN_DIM, _DIM, dtype=weight_dtype
            ),
            "model.norm.weight": randn(_DIM, dtype=weight_dtype),
            "score.weight": randn(1, _DIM, dtype=weight_dtype),
            # adding bias to ensure it doesn't cause an unexpected key
            "score.bias": randn(1, _DIM, dtype=weight_dtype),
        }
        return state_dict

    @pytest.fixture
    def mistral_reward_model_hf_checkpoint(self, tmp_path, state_dict):
        """
        Fixture which creates a checkpoint file for the Mistral reward model. The
        state dict follows the HF_FORMAT for the checkpoint format.

        The state dicts supports testing for a single-file checkpoint.
        Multiple file checkpoints are already tested for Llama2.
            * The checkpoint contains layer0 + embed + norm + score keys
            and can be tested in isolation

        The model corresponds to the following config:
            * num_layers: 1
            * num_heads: 4
            * num_kv_heads: 4
            * embed_dim: 64
            * max_seq_len: 128
            * num_classes: 1
            * intermediate_dim: 256

        """
        checkpoint_file = tmp_path / "mistral_reward_model_hf_checkpoint.pt"

        torch.save(state_dict, checkpoint_file)

        config = {
            "hidden_size": 64,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "num_classes": 1,
        }
        config_file = Path.joinpath(tmp_path, "config.json")
        with config_file.open("w") as f:
            json.dump(config, f)

        return checkpoint_file

    @pytest.fixture
    def single_file_checkpointer(
        self, mistral_reward_model_hf_checkpoint, tmp_path
    ) -> FullModelHFCheckpointer:
        checkpoint_file = mistral_reward_model_hf_checkpoint
        return FullModelHFCheckpointer(
            checkpoint_dir=tmp_path,
            checkpoint_files=[checkpoint_file],
            model_type="REWARD",
            output_dir=tmp_path,
        )

    def test_load_save_checkpoint_single_file(
        self,
        single_file_checkpointer: FullModelHFCheckpointer,
        mistral_reward_model_hf_checkpoint: Path,
    ):
        """
        Test ``load_checkpoint`` and ``save_checkpoint`` method within the
        FullModelHFCheckpointer for a single checkpoint file for a mistral reward model.

        We test:
        * ``load_checkpoint`` loads the right sets of keys
        * Internal state of the checkpointer is correctly updated
        * Converted checkpoint can be loaded into the `mistral_classifier` torchtune implementation
        * Saved checkpoint keys match the original checkpoint
        """
        # Read the state dict directly from file using torch.load. This will be the state
        # dict we test against
        checkpoint_file = mistral_reward_model_hf_checkpoint
        orig_state_dict = safe_torch_load(checkpoint_file)

        # Converted state dict from the checkpointer
        state_dict = single_file_checkpointer.load_checkpoint()
        # Check that we've loaded all the keys minus the output bias
        assert len(state_dict["model"].keys()) == len(orig_state_dict.keys()) - 1

        # the keys in original state dict should match up with the keys in the weight_map
        for key in orig_state_dict.keys():
            if "inv_freq" in key or "output.bias" in key:
                continue
            assert key in single_file_checkpointer._weight_map

        # loading the state dict into the model implementation should work correctly
        model = mistral.mistral_classifier(
            num_classes=1,
            vocab_size=_VOCAB_SIZE,
            num_layers=1,
            num_heads=_NUM_HEADS,
            num_kv_heads=_NUM_KV_HEADS,
            embed_dim=_DIM,
            intermediate_dim=_HIDDEN_DIM,
            max_seq_len=128,
        )
        model.load_state_dict(state_dict["model"])

        single_file_checkpointer.save_checkpoint(state_dict, epoch=1)

        # Reload the output checkpoint file and compare to the original checkpoint. This
        # assumes we know what the name of the file is. This is fine, breaking this logic
        # should be something we capture through this test
        output_file = Path.joinpath(checkpoint_file.parent, "hf_model_0001_1.pt")
        output_state_dict = safe_torch_load(output_file)

        assert len(output_state_dict.keys()) == len(orig_state_dict.keys()) - 1


class TestHFGemmaFullModelCheckpointer:
    @pytest.fixture
    def weight_dtype(self):
        return torch.float16

    @pytest.fixture
    def state_dict(self, weight_dtype):
        """
        State dict for a HF format Gemma checkpoint. This state dict is
        "complete" and can be loaded into a TorchTune model once correctly converted.
        """
        state_dict = {
            "model.embed_tokens.weight": randn(_VOCAB_SIZE, _DIM, dtype=weight_dtype),
            "model.layers.0.input_layernorm.weight": randn(_DIM, dtype=weight_dtype),
            "model.layers.0.self_attn.q_proj.weight": randn(
                _DIM, _NUM_HEADS * _HEAD_DIM, dtype=weight_dtype
            ),
            # setting num_kv_heads to 1
            "model.layers.0.self_attn.k_proj.weight": randn(
                _HEAD_DIM, _DIM, dtype=weight_dtype
            ),
            "model.layers.0.self_attn.v_proj.weight": randn(
                _HEAD_DIM, _DIM, dtype=weight_dtype
            ),
            "model.layers.0.self_attn.o_proj.weight": randn(
                _NUM_HEADS * _HEAD_DIM, _DIM, dtype=weight_dtype
            ),
            "model.layers.0.post_attention_layernorm.weight": randn(
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
            "model.norm.weight": randn(_DIM, dtype=weight_dtype),
        }
        return state_dict

    @pytest.fixture
    def gemma_hf_checkpoint(self, tmp_path, state_dict):
        """
        Fixture which creates a checkpoint file for Gemma. The
        state dict follows the HF_FORMAT for the checkpoint format.

        The state dicts supports testing for a single-file checkpoint.
        Multiple file checkpoints are already tested for Llama2.

        The model corresponds to the following config:
            * num_layers: 1
            * num_heads: 4
            * num_kv_heads: 1
            * embed_dim: 64
            * max_seq_len: 128
            * num_classes: 1
            * intermediate_dim: 256
            * head_dim : 16

        """
        checkpoint_file = tmp_path / "gemma_hf_checkpoint.pt"

        torch.save(state_dict, checkpoint_file)

        config = {
            "hidden_size": _DIM,
            "num_attention_heads": _NUM_HEADS,
            "num_key_value_heads": 1,
            "head_dim": _HEAD_DIM,
            "intermediate_size": _HIDDEN_DIM,
        }
        config_file = Path.joinpath(tmp_path, "config.json")
        with config_file.open("w") as f:
            json.dump(config, f)

        return checkpoint_file

    @pytest.fixture
    def single_file_checkpointer(
        self, gemma_hf_checkpoint, tmp_path
    ) -> FullModelHFCheckpointer:
        checkpoint_file = gemma_hf_checkpoint
        return FullModelHFCheckpointer(
            checkpoint_dir=tmp_path,
            checkpoint_files=[checkpoint_file],
            model_type="GEMMA",
            output_dir=tmp_path,
        )

    def test_load_save_checkpoint_single_file(
        self,
        single_file_checkpointer: FullModelHFCheckpointer,
        gemma_hf_checkpoint: Path,
    ):
        """
        Test ``load_checkpoint`` and ``save_checkpoint`` method within the
        FullModelHFCheckpointer for a single checkpoint file for Gemma.

        We test:
        * ``load_checkpoint`` loads the right sets of keys
        * Internal state of the checkpointer is correctly updated
        * Converted checkpoint can be loaded into the `gemma` TorchTune implementation
        * lm_head weights are tied to the embed_tokens weights during saving
        * lmhead weights are popped during loading
        """
        # Read the state dict directly from file using torch.load. This will be the state
        # dict we test against
        checkpoint_file = gemma_hf_checkpoint
        orig_state_dict = safe_torch_load(checkpoint_file)

        # Converted state dict from the checkpointer

        state_dict = single_file_checkpointer.load_checkpoint()
        assert len(state_dict["model"].keys()) == len(orig_state_dict.keys())

        # the keys in original state dict should match up with the keys in the weight_map
        for key in orig_state_dict.keys():
            if "inv_freq" in key:
                continue
            assert key in single_file_checkpointer._weight_map

        # loading the state dict into the model implementation should work correctly
        model = gemma.gemma(
            vocab_size=_VOCAB_SIZE,
            num_layers=1,
            num_heads=_NUM_HEADS,
            head_dim=_HEAD_DIM,
            num_kv_heads=1,
            embed_dim=_DIM,
            intermediate_dim=_HIDDEN_DIM,
            max_seq_len=128,
        )
        model.load_state_dict(state_dict["model"])

        single_file_checkpointer.save_checkpoint(state_dict, epoch=1)

        # Reload the output checkpoint file and compare to the original checkpoint. This
        # assumes we know what the name of the file is. This is fine, breaking this logic
        # should be something we capture through this test
        output_file = Path.joinpath(checkpoint_file.parent, "hf_model_0001_1.pt")
        output_state_dict = safe_torch_load(output_file)

        assert len(output_state_dict.keys()) == len(orig_state_dict.keys())
