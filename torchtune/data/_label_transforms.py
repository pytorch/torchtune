# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Literal, Mapping, Optional, Union

from torchtune.modules.transforms import Transform


class MultiClassLabelMap(Transform):
    def __init__(self, label_map):
        self.label_map = label_map

    def __call__(self, label_string):
        return self.label_map[label_string]