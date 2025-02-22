# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torchtune.models.llama2 import llama2, llama2_classifier
from torchtune.training.checkpointing._utils import (
    check_outdir_not_in_ckptdir,
    FormattedCheckpointFiles,
    get_all_checkpoints_in_dir,
    prune_surplus_checkpoints,
    safe_torch_load,
    update_state_dict_for_classifier,
)

N_LAYERS = 3
IN_DIM = 5
OUT_DIM = 10
VOCAB_SIZE = 50
NUM_HEADS = 4
NUM_KV_HEADS = 2
EMBED_DIM = 64
MAX_SEQ_LEN = 64
NUM_CLASSES = 6


class TestCheckpointerUtils:
    @pytest.fixture
    def model_checkpoint(self, tmp_path):
        """
        Fixture which creates a checkpoint file for testing checkpointer utils.
        """
        checkpoint_file = tmp_path / "model_checkpoint_01.pt"

        state_dict = {
            "token_embeddings.weight": torch.ones(1, 10),
            "output.weight": torch.ones(1, 10),
        }

        torch.save(state_dict, checkpoint_file)

        return checkpoint_file

    @pytest.mark.parametrize("weights_only", [True, False])
    def test_safe_torch_load(self, model_checkpoint, weights_only):
        state_dict = safe_torch_load(Path(model_checkpoint), weights_only)

        assert "token_embeddings.weight" in state_dict
        assert "output.weight" in state_dict

        assert state_dict["token_embeddings.weight"].shape[1] == 10
        assert state_dict["output.weight"].shape[0] == 1


class TestUpdateStateDictForClassifer:
    @pytest.fixture()
    def llama2_state_dict(self):
        model = llama2(
            vocab_size=VOCAB_SIZE,
            num_layers=N_LAYERS,
            num_heads=NUM_KV_HEADS,
            num_kv_heads=NUM_KV_HEADS,
            embed_dim=EMBED_DIM,
            max_seq_len=MAX_SEQ_LEN,
        )
        return model.state_dict()

    @pytest.fixture()
    def llama2_classifier_model(self):
        return llama2_classifier(
            num_classes=NUM_CLASSES,
            vocab_size=VOCAB_SIZE,
            num_layers=N_LAYERS,
            num_heads=NUM_KV_HEADS,
            num_kv_heads=NUM_KV_HEADS,
            embed_dim=EMBED_DIM,
            max_seq_len=MAX_SEQ_LEN,
        )

    def test_bias_in_classifier_checkpoint_is_removed(self, llama2_classifier_model):
        # construct bogus state dict with output.bias included
        state_dict_with_bias = llama2_classifier_model.state_dict().copy()
        state_dict_with_bias["output.bias"] = torch.tensor([NUM_CLASSES])

        # function should remove output.bias
        update_state_dict_for_classifier(
            state_dict_with_bias, llama2_classifier_model.named_parameters()
        )

        assert "output.bias" not in state_dict_with_bias

    def test_loading_base_checkpoint_into_classifier(
        self, llama2_state_dict, llama2_classifier_model
    ):
        # grabbing the expected output.weight - the correct outcome here
        # is for all weights aside from output.weight to be loaded in
        # from the base model, so output.weight will remain in its rand init state
        expected_output_weight = llama2_classifier_model.state_dict()[
            "output.weight"
        ].clone()

        # update the state dict to load with the classifier's output.weight
        update_state_dict_for_classifier(
            llama2_state_dict, llama2_classifier_model.named_parameters()
        )

        # load in all the base params
        llama2_classifier_model.load_state_dict(llama2_state_dict)

        # now we can assert that output.weight was unchanged
        output_weight = llama2_classifier_model.state_dict()["output.weight"]
        assert torch.equal(expected_output_weight, output_weight)

    def test_assertion_error_when_missing_output_in_state_dict(
        self, llama2_state_dict, llama2_classifier_model
    ):
        llama2_state_dict.pop("output.weight")
        with pytest.raises(
            AssertionError, match="Expected output.weight in state_dict"
        ):
            update_state_dict_for_classifier(
                llama2_state_dict, llama2_classifier_model.named_parameters()
            )

    def test_assertion_error_when_missing_output_in_model_named_parameters(
        self, llama2_state_dict, llama2_classifier_model
    ):
        named_params = [
            (k, v)
            for (k, v) in llama2_classifier_model.named_parameters()
            if k != "output.weight"
        ]
        with pytest.raises(
            AssertionError, match="Expected output.weight in model_named_parameters"
        ):
            update_state_dict_for_classifier(llama2_state_dict, named_params)

    def test_loading_classifier_weights(self, llama2_classifier_model):
        state_dict_to_load = deepcopy(llama2_classifier_model.state_dict())
        state_dict_to_load["output.weight"] = torch.ones_like(
            state_dict_to_load["output.weight"]
        )

        update_state_dict_for_classifier(
            state_dict_to_load, llama2_classifier_model.named_parameters()
        )
        llama2_classifier_model.load_state_dict(state_dict_to_load)

        model_state_dict = llama2_classifier_model.state_dict()

        assert set(model_state_dict.keys()) == set(state_dict_to_load.keys())
        assert torch.equal(
            model_state_dict["output.weight"],
            torch.ones_like(model_state_dict["output.weight"]),
        )

    def test_loading_classifier_weights_force_override(self, llama2_classifier_model):
        state_dict_to_load = deepcopy(llama2_classifier_model.state_dict())
        state_dict_to_load["output.weight"] = torch.ones_like(
            state_dict_to_load["output.weight"]
        )

        expected_output_weight = llama2_classifier_model.state_dict()[
            "output.weight"
        ].clone()

        update_state_dict_for_classifier(
            state_dict_to_load, llama2_classifier_model.named_parameters(), True
        )
        llama2_classifier_model.load_state_dict(state_dict_to_load)

        model_state_dict = llama2_classifier_model.state_dict()

        assert set(model_state_dict.keys()) == set(state_dict_to_load.keys())
        assert torch.equal(model_state_dict["output.weight"], expected_output_weight)


class TestFormattedCheckpointFiles:
    @pytest.fixture
    def expected_filenames(self):
        return [
            "model_0001_of_0012.pt",
            "model_0002_of_0012.pt",
            "model_0003_of_0012.pt",
            "model_0004_of_0012.pt",
            "model_0005_of_0012.pt",
            "model_0006_of_0012.pt",
            "model_0007_of_0012.pt",
            "model_0008_of_0012.pt",
            "model_0009_of_0012.pt",
            "model_0010_of_0012.pt",
            "model_0011_of_0012.pt",
            "model_0012_of_0012.pt",
        ]

    def test_invalid_from_dict_no_filename_format(self):
        invalid_dict = {"bad_key": "model_{}_of_{}.pt", "max_filename": "0005"}
        with pytest.raises(ValueError, match="Must pass 'filename_format'"):
            _ = FormattedCheckpointFiles.from_dict(invalid_dict)

    def test_invalid_from_dict_int_max_filename(self):
        # the 0o0005 is an octal number. we use this insane value in this test
        # as YAML treats numbers with a leading 0 as an octal number, so this
        # may be a good example of `from_dict` being called with an invalid config
        invalid_dict = {"filename_format": "model_{}_of_{}.pt", "max_filename": 0o00025}
        with pytest.raises(ValueError, match="`max_filename` must be a string"):
            _ = FormattedCheckpointFiles.from_dict(invalid_dict)

    def test_invalid_filename_format(self):
        formatted_string = "invalid_format_{}.pt"
        formatted_file_dict = {
            "filename_format": formatted_string,
            "max_filename": "0005",
        }
        with pytest.raises(ValueError, match="must have exactly two placeholders"):
            FormattedCheckpointFiles.from_dict(formatted_file_dict)

    def test_build_checkpoint_filenames(self, expected_filenames):
        formatted_file_dict = {
            "filename_format": "model_{}_of_{}.pt",
            "max_filename": "0012",
        }
        formatted_files = FormattedCheckpointFiles.from_dict(formatted_file_dict)
        actual_filenames = formatted_files.build_checkpoint_filenames()
        assert actual_filenames == expected_filenames


class TestCheckOutdirNotInCkptdir:
    def test_sibling_directories(self):
        # Sibling directories should pass without raising an error
        ckpt_dir = Path("/path/to/ckpt")
        out_dir = Path("/path/to/output")
        check_outdir_not_in_ckptdir(ckpt_dir, out_dir)

    def test_ckpt_dir_in_output_dir(self):
        # out_dir is a parent of ckpt_dir, should pass without raising an error
        ckpt_dir = Path("/path/to/output/ckpt_dir")
        out_dir = Path("/path/to/output")
        check_outdir_not_in_ckptdir(ckpt_dir, out_dir)

    def test_equal_directories(self):
        # Equal directories should raise a ValueError
        ckpt_dir = Path("/path/to/ckpt")
        out_dir = Path("/path/to/ckpt")
        with pytest.raises(
            ValueError,
            match="The output directory cannot be the same as or a subdirectory of the checkpoint directory.",
        ):
            check_outdir_not_in_ckptdir(ckpt_dir, out_dir)

    def test_output_dir_in_ckpt_dir(self):
        # out_dir is a subdirectory of ckpt_dir, should raise a ValueError
        ckpt_dir = Path("/path/to/ckpt")
        out_dir = Path("/path/to/ckpt/subdir")
        with pytest.raises(
            ValueError,
            match="The output directory cannot be the same as or a subdirectory of the checkpoint directory.",
        ):
            check_outdir_not_in_ckptdir(ckpt_dir, out_dir)

    def test_output_dir_ckpt_dir_few_levels_down(self):
        # out_dir is a few levels down in ckpt_dir, should raise a ValueError
        ckpt_dir = Path("/path/to/ckpt")
        out_dir = Path("/path/to/ckpt/subdir/another_subdir")
        with pytest.raises(
            ValueError,
            match="The output directory cannot be the same as or a subdirectory of the checkpoint directory.",
        ):
            check_outdir_not_in_ckptdir(ckpt_dir, out_dir)


class TestGetAllCheckpointsInDir:
    """Series of tests for the ``get_all_checkpoints_in_dir`` function."""

    def test_get_all_ckpts_simple(self, tmpdir):
        tmpdir = Path(tmpdir)
        ckpt_dir_0 = tmpdir / "epoch_0"
        ckpt_dir_0.mkdir(parents=True, exist_ok=True)

        ckpt_dir_1 = tmpdir / "epoch_1"
        ckpt_dir_1.mkdir()

        all_ckpts = get_all_checkpoints_in_dir(tmpdir)
        assert len(all_ckpts) == 2
        assert ckpt_dir_0 in all_ckpts
        assert ckpt_dir_1 in all_ckpts

    def test_get_all_ckpts_with_pattern_that_matches_some(self, tmpdir):
        """Test that we only return the checkpoints that match the pattern."""
        tmpdir = Path(tmpdir)
        ckpt_dir_0 = tmpdir / "epoch_0"
        ckpt_dir_0.mkdir(parents=True, exist_ok=True)

        ckpt_dir_1 = tmpdir / "step_1"
        ckpt_dir_1.mkdir()

        all_ckpts = get_all_checkpoints_in_dir(tmpdir)
        assert len(all_ckpts) == 1
        assert all_ckpts == [ckpt_dir_0]

    def test_get_all_ckpts_override_pattern(self, tmpdir):
        """Test that we can override the default pattern and it works."""
        tmpdir = Path(tmpdir)
        ckpt_dir_0 = tmpdir / "epoch_0"
        ckpt_dir_0.mkdir(parents=True, exist_ok=True)

        ckpt_dir_1 = tmpdir / "step_1"
        ckpt_dir_1.mkdir()

        all_ckpts = get_all_checkpoints_in_dir(tmpdir, pattern="step_*")
        assert len(all_ckpts) == 1
        assert all_ckpts == [ckpt_dir_1]

    def test_get_all_ckpts_only_return_dirs(self, tmpdir):
        """Test that even if a file matches the pattern, we only return directories."""
        tmpdir = Path(tmpdir)
        ckpt_dir_0 = tmpdir / "epoch_0"
        ckpt_dir_0.mkdir(parents=True, exist_ok=True)

        file = tmpdir / "epoch_1"
        file.touch()

        all_ckpts = get_all_checkpoints_in_dir(tmpdir)
        assert len(all_ckpts) == 1
        assert all_ckpts == [ckpt_dir_0]


class TestPruneSurplusCheckpoints:
    """Series of tests for the ``prune_surplus_checkpoints`` function."""

    def test_prune_surplus_checkpoints_simple(self, tmpdir):
        tmpdir = Path(tmpdir)
        ckpt_dir_0 = tmpdir / "epoch_0"
        ckpt_dir_0.mkdir(parents=True, exist_ok=True)

        ckpt_dir_1 = tmpdir / "epoch_1"
        ckpt_dir_1.mkdir()

        prune_surplus_checkpoints([ckpt_dir_0, ckpt_dir_1], 1)
        remaining_ckpts = os.listdir(tmpdir)
        assert len(remaining_ckpts) == 1
        assert remaining_ckpts == ["epoch_1"]

    def test_prune_surplus_checkpoints_keep_last_invalid(self, tmpdir):
        """Test that we raise an error if keep_last_n_checkpoints is not >= 1"""
        tmpdir = Path(tmpdir)
        ckpt_dir_0 = tmpdir / "epoch_0"
        ckpt_dir_0.mkdir(parents=True, exist_ok=True)

        ckpt_dir_1 = tmpdir / "epoch_1"
        ckpt_dir_1.mkdir()

        with pytest.raises(
            ValueError,
            match="keep_last_n_checkpoints must be greater than or equal to 1",
        ):
            prune_surplus_checkpoints([ckpt_dir_0, ckpt_dir_1], 0)
