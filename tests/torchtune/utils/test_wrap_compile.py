# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.


import torch
from torchtune import utils


class TestWrapCompile:
    def test_wrap_compile(self) -> None:
        """
        Ensures that compile prefix is removed in compiled model
        state_dict and can be loaded into non-compiled model.
        """
        m = torch.nn.Linear(5, 5)
        m = utils.wrap_compile(m)
        assert isinstance(m, torch._dynamo.eval_frame.OptimizedModule)
        load_m = torch.nn.Linear(5, 5)
        missing, unexpected = load_m.load_state_dict(m.state_dict(), strict=False)
        assert not missing
        assert not unexpected
