# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# TODO: (rohan-varma): co-locate the asset with the test using it

from pathlib import Path

import torch
from torchtune.models.llama2.transformer import TransformerDecoder

ASSETS = Path(__file__).parent.parent.parent / "assets"
_DEBUG_CKPT_PATH = str(ASSETS / "small_ckpt.model")


class TestDebugModelLoad:
    def test_debug_load(self):
        debug_m = TransformerDecoder(
            vocab_size=32_000,
            num_layers=4,
            num_heads=16,
            embed_dim=256,
            max_seq_len=2048,
            norm_eps=1e-5,
            num_kv_heads=8,
        )
        loaded = torch.load(_DEBUG_CKPT_PATH, weights_only=True)
        missing, unexpected = debug_m.load_state_dict(loaded)
        assert not missing, f"Missing keys {missing}"
        assert not unexpected, f"Unexpected keys {unexpected}"
