# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass


@dataclass
class MistralTestConfig:
    BSZ = 2
    SEQ_LEN = 128
    EMBED_DIM = 64
    VOCAB_SIZE = 512
    NUM_LAYERS = 4
    NUM_HEADS = 4
    NUM_KV_HEADS = 2
    INTERMEDIATE_DIM = 512
    MAX_SEQ_LEN = 256
    ROPE_BASE = 10000
    NORM_EPS = 1e-5
    SEED = 16
