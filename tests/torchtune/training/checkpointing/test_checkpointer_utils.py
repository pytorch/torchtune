# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torchtune.models.llama2 import llama2, llama2_classifier
from torchtune.training.checkpointing._utils import (
    FormattedCheckpointFiles,
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

    def test_invalid_to_dict(self):
        invalid_dict = {"bad_key": "model_{}_of_{}.pt", "max_filename": "0005"}
        with pytest.raises(ValueError, match="Must pass 'filename_format'"):
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
