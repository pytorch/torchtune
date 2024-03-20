# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import re
from typing import Dict, List, Optional

import torch
from tests.test_utils import get_assets_path
from torch.utils.data import Dataset

_ASSETS = get_assets_path()


class DummyDataset(Dataset):
    def __init__(self, **kwargs):
        self._data = torch.LongTensor(
            [
                [0, 2, 4, 2, 5, 6, 7, 8, 9, 1, 2, 4, 3, 3, 5, 6, 8, 2, 1, 1],
                [1, 2, 5, 6, 7, 8, 2, 3, 1, 9, 9, 9, 5, 6, 7, 0, 0, 0, 1, 2],
                [5, 6, 8, 2, 1, 0, 3, 4, 0, 0, 0, 2, 4, 7, 8, 8, 2, 2, 1, 0],
                [4, 6, 7, 1, 0, 2, 0, 2, 0, 2, 3, 9, 9, 9, 7, 5, 1, 8, 4, 1],
            ]
        )
        self._labels = torch.LongTensor(
            [
                [2, 6, 7, 8, 2, 2, 1, 0, 0, 1],
                [1, 2, 5, 6, 7, 8, 2, 3, 1, 9],
                [6, 1, 1, 2, 5, 0, 9, 0, 2, 1],
                [5, 8, 6, 0, 2, 0, 0, 3, 2, 1],
            ]
        )

    def __getitem__(self, index):
        return (self._data[index], self._labels[index])

    def __len__(self):
        return len(self._data)


def llama2_test_config(max_batch_size: Optional[int] = None) -> List[str]:
    return [
        "model._component_=torchtune.models.llama2.llama2",
        "model.vocab_size=32_000",
        "model.num_layers=4",
        "model.num_heads=16",
        "model.embed_dim=256",
        "model.max_seq_len=2048",
        "model.norm_eps=1e-5",
        "model.num_kv_heads=8",
        f"model.max_batch_size={max_batch_size if max_batch_size else 'null'}",
    ]


def lora_llama2_test_config(
    lora_attn_modules,
    apply_lora_to_mlp: bool = False,
    apply_lora_to_output: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 16,
    max_batch_size: Optional[int] = None,
) -> List[str]:
    lora_attn_modules_str = "['" + "','".join([x for x in lora_attn_modules]) + "']"
    return [
        # Note: we explicitly use _component_ so that we can also call
        # config.instantiate directly for easier comparison
        "model._component_=torchtune.models.llama2.lora_llama2",
        f"model.lora_attn_modules={lora_attn_modules}",
        f"model.apply_lora_to_mlp={apply_lora_to_mlp}",
        f"model.apply_lora_to_output={apply_lora_to_output}",
        "model.vocab_size=32000",
        "model.num_layers=4",
        "model.num_heads=16",
        "model.embed_dim=256",
        "model.max_seq_len=2048",
        "model.norm_eps=1e-5",
        "model.num_kv_heads=8",
        f"model.max_batch_size={max_batch_size if max_batch_size else 'null'}",
        f"model.lora_rank={lora_rank}",
        f"model.lora_alpha={lora_alpha}",
        "model.lora_dropout=0.0",
    ]


def fetch_ckpt_model_path(ckpt) -> str:
    # TODO: same checkpoint is returned for small scale llama2
    # and lora. This should be fine as the lora adapter params
    # are initialized, but we may want to load in a lora specific
    # checkpoint.
    if ckpt == "small_test_ckpt_tune":
        return "/tmp/test-artifacts/small-ckpt-tune-03082024.pt"
    if ckpt == "small_test_ckpt_meta":
        return "/tmp/test-artifacts/small-ckpt-meta-03082024.pt"
    if ckpt == "small_test_ckpt_hf":
        return "/tmp/test-artifacts/small-ckpt-hf-03082024.pt"
    if "llama2_7b" in ckpt:
        return "/tmp/test-artifacts/llama2-7b-torchtune.pt"
    if "tiny_test_ckpt" in ckpt:
        return _ASSETS / "tiny_llama2_checkpoint.pt"
    raise ValueError(f"Unknown ckpt {ckpt}")


def get_checkpointer_class_path_for_test_ckpt(ckpt_name: str) -> str:
    test_weights_to_checkpointer_class_mapping = {
        "small_test_ckpt_tune": "torchtune.utils.FullModelTorchTuneCheckpointer",
        "llama2.llama2_7b": "torchtune.utils.FullModelTorchTuneCheckpointer",
        "small_test_ckpt_meta": "torchtune.utils.FullModelMetaCheckpointer",
        "small_test_ckpt_hf": "torchtune.utils.FullModelHFCheckpointer",
    }
    checkpoint_class_path = test_weights_to_checkpointer_class_mapping.get(
        ckpt_name, None
    )
    if not checkpoint_class_path:
        raise ValueError("Invalid checkpointer class for checkpoint {ckpt_name}")
    return checkpoint_class_path


def get_loss_values_from_metric_logger(
    out_dir: str, remove_found_file: bool = False
) -> Dict[str, float]:
    txt_files = [f for f in os.listdir(out_dir) if f.endswith(".txt")]
    assert len(txt_files) == 1, "Should have exactly one log file"
    log_file_path = os.path.join(out_dir, txt_files[0])
    with open(log_file_path, "r") as f:
        logs = f.read()
    losses = [float(x) for x in re.findall(r"loss:(\d+\.\d+)", logs)]
    if remove_found_file:
        os.remove(log_file_path)
    return losses
