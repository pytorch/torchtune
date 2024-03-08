
import pytest
import torch

from pathlib import Path
from torch import randn

from torchtune.models import llama2
from torchtune.utils._checkpointing import (
    CheckpointFormat,
    FullModelCheckpointer,
    ModelType
)
from torchtune.utils.seed import set_seed


@pytest.fixture(autouse=True)
def random():
    set_seed(16)


class TestHFLlama2FullModelCheckpointer:

    @pytest.fixture
    def vocab_size(self):
        return 100

    @pytest.fixture
    def dim(self):
        return 64

    @pytest.fixture
    def hidden_dim(self):
        return 256

    @pytest.fixture
    def num_heads(self):
        return 4

    @pytest.fixture
    def llama2_hf_checkpoints(self, tmp_path, vocab_size, dim, hidden_dim):
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

        weight_dtype=torch.bfloat16

        state_dict_1 = {
            'model.embed_tokens.weight': randn(vocab_size, dim, dtype=weight_dtype),
            'model.layers.0.input_layernorm.weight': randn(dim, dtype=weight_dtype),
            'model.layers.0.self_attn.q_proj.weight': randn(dim, dim, dtype=weight_dtype),
            'model.layers.0.self_attn.k_proj.weight': randn(dim, dim, dtype=weight_dtype),
            'model.layers.0.self_attn.v_proj.weight': randn(dim, dim, dtype=weight_dtype),
            'model.layers.0.self_attn.o_proj.weight': randn(dim, dim, dtype=weight_dtype),
            'model.layers.0.post_attention_layernorm.weight': randn(dim, dtype=weight_dtype),
            'model.layers.0.self_attn.rotary_emb.inv_freq': randn(dim, dtype=weight_dtype),
            'model.layers.0.mlp.gate_proj.weight': randn(hidden_dim, dim, dtype=weight_dtype),
            'model.layers.0.mlp.down_proj.weight': randn(dim, hidden_dim, dtype=weight_dtype),
            'model.layers.0.mlp.up_proj.weight': randn(hidden_dim, dim, dtype=weight_dtype),
            'model.norm.weight': torch.randn(dim, dtype=weight_dtype),
            'lm_head.weight': torch.randn(vocab_size, dim, dtype=weight_dtype),
        }

        state_dict_2 = {
            'model.layers.1.input_layernorm.weight': randn(dim, dtype=weight_dtype),
            'model.layers.1.self_attn.q_proj.weight': randn(dim, dim, dtype=weight_dtype),
            'model.layers.1.self_attn.k_proj.weight': randn(dim, dim, dtype=weight_dtype),
            'model.layers.1.self_attn.v_proj.weight': randn(dim, dim, dtype=weight_dtype),
            'model.layers.1.self_attn.o_proj.weight': randn(dim, dim, dtype=weight_dtype),
            'model.layers.1.post_attention_layernorm.weight': randn(dim, dtype=weight_dtype),
            'model.layers.1.self_attn.rotary_emb.inv_freq': randn(dim, dtype=weight_dtype),
            'model.layers.1.mlp.gate_proj.weight': randn(hidden_dim, dim, dtype=weight_dtype),
            'model.layers.1.mlp.down_proj.weight': randn(dim, hidden_dim, dtype=weight_dtype),
            'model.layers.1.mlp.up_proj.weight': randn(hidden_dim, dim, dtype=weight_dtype),
        }

        torch.save(state_dict_1, checkpoint_file_1)
        torch.save(state_dict_2, checkpoint_file_2)
        return (checkpoint_file_1, checkpoint_file_2)

    @pytest.fixture
    def single_file_checkpointer(self, llama2_hf_checkpoints, tmp_path) -> FullModelCheckpointer:
        checkpoint_file, _ = llama2_hf_checkpoints
        return FullModelCheckpointer(
            checkpoint_dir=tmp_path,
            checkpoint_files=[checkpoint_file],
            checkpoint_format=CheckpointFormat.HF_FORMAT,
            model_type=ModelType.LLAMA2_7B,
            output_dir=tmp_path,
        )

    @pytest.fixture
    def multi_file_checkpointer(self, llama2_hf_checkpoints, tmp_path) -> FullModelCheckpointer:
        checkpoint_file_1, checkpoint_file_2 = llama2_hf_checkpoints
        return FullModelCheckpointer(
            checkpoint_dir=tmp_path,
            checkpoint_files=[checkpoint_file_1, checkpoint_file_2],
            checkpoint_format=CheckpointFormat.HF_FORMAT,
            model_type=ModelType.LLAMA2_7B,
            output_dir=tmp_path,
        )

    def test_load_checkpoint_single_file(
        self,
        single_file_checkpointer,
        llama2_hf_checkpoints,
        vocab_size,
        dim,
        hidden_dim,
        num_heads
    ):
        """
        Test ``load_checkpoint`` method within the FullModelCheckpointer for a single
        checkpoint file.

        We test:
        * ``load_checkpoint`` loads the right sets of keys
        * Internal state of the checkpointer is correctly updated
        * Converted checkpoint can be loaded into the llama2 TorchTune implementation
        """
        # Read the state dict directly from file
        checkpoint_file, _ = llama2_hf_checkpoints
        orig_state_dict = torch.load(checkpoint_file, mmap=True, map_location='cpu')

        # Converted state dict from the checkpointer
        converted_state_dict = single_file_checkpointer.load_checkpoint(num_heads=4, dim=64)

        # We ignore inv_freq as is standard practice
        assert len(converted_state_dict.keys()) + 1 == len(orig_state_dict.keys())

        # The dtype for a random weight should match up with the checkpoint state
        _, original_weight = next(iter(orig_state_dict.items()))
        assert original_weight.dtype == single_file_checkpointer._checkpoint_dtype

        # the keys in original state dict should match up with the keys in the weight_map
        for key in orig_state_dict.keys():
            if 'inv_freq' in key:
                continue
            assert key in single_file_checkpointer._weight_map

        # loading the state dict into the model implementation should work correctly
        model = llama2.llama2(
            vocab_size=vocab_size,
            num_layers=1,
            num_heads=num_heads,
            num_kv_heads=num_heads,
            embed_dim=dim,
            max_seq_len=128
        )
        model.load_state_dict(converted_state_dict)


    def test_save_checkpoint_single_file(self, single_file_checkpointer, llama2_hf_checkpoints):
        """
        Test ``save_checkpoint`` method within the FullModelCheckpointer for a single
        checkpoint file.

        We test:
        * The output checkpoint keys match the original checkpoint
        * The output checkpoint dtype matches the original checkpoint
        """
        # Read the state dict directly from file
        checkpoint_file, _ = llama2_hf_checkpoints
        orig_state_dict = torch.load(checkpoint_file, mmap=True, map_location='cpu')

        converted_state_dict = single_file_checkpointer.load_checkpoint(num_heads=4, dim=64)
        single_file_checkpointer.save_checkpoint(converted_state_dict, num_heads=4, dim=64)

        # Reload the output checkpoint file and compare to the original checkpoint. This
        # assumes we know what the name of the file is. This is fine, breaking this logic
        # should be something we capture through this test
        output_file = Path.joinpath(

            checkpoint_file.parent, ('torchtune_' + checkpoint_file.name)
        )
        output_state_dict = torch.load(output_file, mmap=True, map_location='cpu')

        # We ignore inv_freq as is standard practice and so output dict will have one less key
        assert len(output_state_dict.keys()) + 1 == len(orig_state_dict.keys())

        _, orig_weight = next(iter(orig_state_dict.items()))
        _, output_weight = next(iter(output_state_dict.items()))
        assert orig_weight.dtype == output_weight.dtype


    def test_load_checkpoint_multiple_file(self, multi_file_checkpointer, llama2_hf_checkpoints):
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
        orig_state_dict_1 = torch.load(checkpoint_file_1, mmap=True, map_location='cpu')
        orig_state_dict_2 = torch.load(checkpoint_file_2, mmap=True, map_location='cpu')

        # merged state dict from checkpointer
        merged_state_dict = multi_file_checkpointer.load_checkpoint(num_heads=4, dim=64)

        # We ignore inv_freq as is standard practice
        assert (
            len(merged_state_dict.keys()) + 2 ==
            len(orig_state_dict_1.keys()) + len(orig_state_dict_2.keys())
        )

        # The dtype for a random weight should match up exactly
        _, orig_weight = next(iter(orig_state_dict_1.items()))
        assert orig_weight.dtype == multi_file_checkpointer._checkpoint_dtype

        # the keys in the weight_map should match up with the keys in the weight_map
        for key in orig_state_dict_1.keys():
            if 'inv_freq' in key:
                continue
            assert key in multi_file_checkpointer._weight_map

        for key in orig_state_dict_2.keys():
            if 'inv_freq' in key:
                continue
            assert key in multi_file_checkpointer._weight_map

        # finally loading into the model should work
        model = llama2.llama2(
            vocab_size=100,
            num_layers=2,
            num_heads=4,
            num_kv_heads=4,
            embed_dim=64,
            max_seq_len=128
        )
        model.load_state_dict(merged_state_dict)

    def test_save_checkpoint_single_file(self, multi_file_checkpointer, llama2_hf_checkpoints):
        """
        Test ``save_checkpoint`` method within the FullModelCheckpointer for multiple
        checkpoint file.

        This test ensures:
        * The output checkpoint keys match the original checkpoints
        * The output checkpoint dtype matches the original checkpoint
        """
        # Read the state dict directly from files
        checkpoint_file_1, checkpoint_file_2 = llama2_hf_checkpoints
        orig_state_dict_1 = torch.load(checkpoint_file_1, mmap=True, map_location='cpu')
        orig_state_dict_2 = torch.load(checkpoint_file_2, mmap=True, map_location='cpu')

        # merged state dict from checkpointer
        merged_state_dict = multi_file_checkpointer.load_checkpoint(num_heads=4, dim=64)
        multi_file_checkpointer.save_checkpoint(merged_state_dict, num_heads=4, dim=64)

        # Reload the output checkpoint file and compare to the original checkpoint. This
        # assumes we know what the name of the file is. This is fine, breaking this logic
        # should be something we capture through this test
        output_file_1 = Path.joinpath(
            checkpoint_file_1.parent, ('torchtune_' + checkpoint_file_1.name)
        )
        output_file_2 = Path.joinpath(
            checkpoint_file_2.parent, ('torchtune_' + checkpoint_file_2.name)
        )
        output_state_dict_1 = torch.load(output_file_1, mmap=True, map_location='cpu')
        output_state_dict_2 = torch.load(output_file_2, mmap=True, map_location='cpu')

        assert len(output_state_dict_1.keys()) + 1 == len(orig_state_dict_1.keys())
        assert len(output_state_dict_2.keys()) + 1 == len(orig_state_dict_2.keys())

        _, orig_weight = next(iter(orig_state_dict_1.items()))
        _, output_weight = next(iter(output_state_dict_1.items()))

        assert orig_weight.dtype == output_weight.dtype
