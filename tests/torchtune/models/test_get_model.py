# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torchtune import models


class TestModelTokenizerGetter:
    def test_get_model(self):
        """
        Test getting a named model
        """
        models.ALL_MODELS["test"] = lambda x: x
        model = models.get_model("test", "cpu", x=1)
        assert model == 1

    def test_get_model_device(self):
        models.ALL_MODELS["test"] = lambda x: x
        model = models.get_model("test", device=torch.device("cpu"), x=1)
        assert model == 1

    def test_list_models(self):
        """
        Test accuracy of model list
        """
        model_names = models.list_models()
        assert "test" in model_names

    def test_get_tokenizer(self):
        """
        Test getting a named tokenizer
        """
        models.ALL_TOKENIZERS["test"] = lambda x: x
        tokenizer = models.get_tokenizer("test", x=1)
        assert tokenizer == 1

    def test_list_tokenizer(self):
        """
        Test accuracy of tokenizer list
        """
        tokenizer_names = models.list_tokenizers()
        assert "test" in tokenizer_names
