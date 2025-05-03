# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import vllm
from tensordict import from_dataclass

VllmCompletionOutput = from_dataclass(vllm.outputs.CompletionOutput)
