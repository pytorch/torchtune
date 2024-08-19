# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from torchtune.modules.transforms.vision_utils.get_inscribed_size import (
    get_inscribed_size,
)


class TestTransforms:
    @pytest.mark.parametrize(
        "params",
        [
            {
                "image_size": (200, 100),
                "target_size": (1000, 1200),
                "max_size": 600,
                "expected_inscribed_size": (600, 300),
            },
            {
                "image_size": (2000, 200),
                "target_size": (1000, 1200),
                "max_size": 600,
                "expected_inscribed_size": (1000, 100),
            },
            {
                "image_size": (400, 200),
                "target_size": (1000, 1200),
                "max_size": 2000,
                "expected_inscribed_size": (1000, 500),
            },
            {
                "image_size": (400, 200),
                "target_size": (1000, 1200),
                "max_size": None,
                "expected_inscribed_size": (1000, 500),
            },
            {
                "image_size": (1000, 500),
                "target_size": (400, 300),
                "max_size": None,
                "expected_inscribed_size": (400, 200),
            },
        ],
    )
    def test_get_inscribed_size(self, params):
        image_size = params["image_size"]
        target_size = params["target_size"]
        max_size = params["max_size"]
        expected_inscribed_size = params["expected_inscribed_size"]

        inscribed_size = get_inscribed_size(
            image_size=image_size,
            target_size=target_size,
            max_size=max_size,
        )

        assert inscribed_size == expected_inscribed_size
