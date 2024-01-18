# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.


from torchtune.utils.distributed import get_world_size_and_rank


class TestDistributed:
    def test_world_rank(self) -> None:
        # add rest of distributed tests before landing
        ws, r = get_world_size_and_rank()
        assert ws == 1
        assert r == 0
