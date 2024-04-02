# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List


def truncate(
    tokens: List[Any],
    max_seq_len: int,
    eos_id: Any,
) -> List[Any]:
    tokens_truncated = tokens[:max_seq_len]
    if tokens_truncated[-1] != eos_id:
        tokens_truncated[-1] = eos_id
    return tokens_truncated
