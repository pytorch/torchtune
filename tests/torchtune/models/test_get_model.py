# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtune import models


class TestModelTokenizerGetter:
    def test_get_model(self):
        """
        Test getting a named models
        """
        models._MODEL_DICT["test"] = lambda x: x
        model = models.get_model("test", "cpu", x=1)
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
        models._TOKENIZER_DICT["test"] = lambda x: x
        tokenizer = models.get_tokenizer("test", x=1)
        assert tokenizer == 1

    def test_list_tokenizer(self):
        """
        Test accuracy of tokenizer list
        """
        tokenizer_names = models.list_tokenizers()
        assert "test" in tokenizer_names
