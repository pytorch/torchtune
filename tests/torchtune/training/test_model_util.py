# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import warnings

import torch
from torchtune.training._model_util import disable_dropout


class TestDisableDropout:
    def test_disable_dropout(self):
        """
        Tests that dropout layers in the model are disabled.
        """
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.Dropout(p=0.5),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.3),
        )
        disable_dropout(model)
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                assert module.p == 0, f"Dropout layer {module} was not disabled."

    def test_disable_dropout_warning(self):
        """
        Tests that correct number warning is issued when dropout layers are found and disabled.
        """
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.Dropout(p=0.5),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Dropout(p=0.0),
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            disable_dropout(model)
            assert len(w) == 2, "Expected 2 warnings for 2 dropout layers."
            assert issubclass(w[-1].category, UserWarning)
            assert "Found Dropout with" in str(w[-1].message)

    def test_disable_dropout_no_warning(self):
        """
        Tests that no warning is issued when there are no dropout layers.
        """
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 10),
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            disable_dropout(model)
            assert len(w) == 0, "Expected no warnings when there are no dropout layers."
