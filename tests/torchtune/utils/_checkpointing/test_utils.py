# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torchtune.models.llama2 import llama2, llama2_classifier
from torchtune.utils import update_state_dict_for_classifier

N_LAYERS = 3
IN_DIM = 5
OUT_DIM = 10
VOCAB_SIZE = 50
NUM_HEADS = 4
NUM_KV_HEADS = 2
EMBED_DIM = 64
MAX_SEQ_LEN = 64
NUM_CLASSES = 6


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
        expected_output_weight = torch.tensor(
            llama2_classifier_model.state_dict()["output.weight"]
        )

        # update the state dict to load with the classifier's output.weight
        update_state_dict_for_classifier(
            llama2_state_dict, llama2_classifier_model.named_parameters()
        )

        # load in all the base params
        llama2_classifier_model.load_state_dict(llama2_state_dict)

        # now we can assert that output.weight was unchanged
        output_weight = llama2_classifier_model.state_dict()["output.weight"]
        torch.testing.assert_close(
            expected_output_weight, output_weight, atol=0, rtol=0
        )

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
