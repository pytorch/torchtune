# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset

CKPT_COMPONENT_MAP = {
    "tune": "torchtune.training.FullModelTorchTuneCheckpointer",
    "meta": "torchtune.training.FullModelMetaCheckpointer",
    "hf": "torchtune.training.FullModelHFCheckpointer",
}


class DummyDataset(Dataset):
    def __init__(self, *args, **kwargs):
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
        return {"tokens": self._data[index], "labels": self._labels[index]}

    def __len__(self):
        return len(self._data)


def get_assets_path():
    return Path(__file__).parent.parent / "assets"


def dummy_stack_exchange_dataset_config():
    data_files = os.path.join(get_assets_path(), "stack_exchange_paired_tiny.json")
    out = [
        "dataset._component_=torchtune.datasets.stack_exchange_paired_dataset",
        "dataset.source='json'",
        f"dataset.data_files={data_files}",
        "dataset.split='train'",
    ]
    return out


def dummy_alpaca_dataset_config():
    data_files = os.path.join(get_assets_path(), "alpaca_tiny.json")
    out = [
        "dataset._component_=torchtune.datasets.alpaca_dataset",
        "dataset.source='json'",
        f"dataset.data_files={data_files}",
        "dataset.split='train'",
    ]
    return out


def dummy_text_completion_alpaca_dataset_config():
    """
    Constructs a minimal text-completion-style dataset from ``alpaca_tiny.json``.
    This is used for testing PPO fine-tuning.
    """
    data_files = os.path.join(get_assets_path(), "alpaca_tiny.json")
    out = [
        "dataset._component_=torchtune.datasets.text_completion_dataset",
        "dataset.source='json'",
        f"dataset.data_files={data_files}",
        "dataset.column='instruction'",
        "dataset.split='train[:10%]'",  # 10% of the dataset gets us 8 batches
        "dataset.add_eos=False",
    ]
    return out


def llama2_test_config() -> List[str]:
    return [
        "model._component_=torchtune.models.llama2.llama2",
        "model.vocab_size=32_000",
        "model.num_layers=4",
        "model.num_heads=16",
        "model.embed_dim=256",
        "model.max_seq_len=2048",
        "model.norm_eps=1e-5",
        "model.num_kv_heads=8",
    ]


def llama2_classifier_test_config() -> List[str]:
    return [
        "model._component_=torchtune.models.llama2.llama2_classifier",
        "model.num_classes=1",
        "model.vocab_size=32_000",
        "model.num_layers=4",
        "model.num_heads=16",
        "model.embed_dim=256",
        "model.max_seq_len=2048",
        "model.norm_eps=1e-5",
        "model.num_kv_heads=8",
    ]


def llama3_test_config() -> List[str]:
    return [
        "model._component_=torchtune.models.llama3.llama3",
        "model.vocab_size=128_256",
        "model.num_layers=2",
        "model.num_heads=8",
        "model.embed_dim=64",
        "model.max_seq_len=1024",
        "model.norm_eps=1e-5",
        "model.num_kv_heads=4",
    ]


def lora_llama2_test_config(
    lora_attn_modules,
    apply_lora_to_mlp: bool = False,
    apply_lora_to_output: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 16,
    quantize_base: bool = False,
) -> List[str]:
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
        f"model.lora_rank={lora_rank}",
        f"model.lora_alpha={lora_alpha}",
        "model.lora_dropout=0.0",
        f"model.quantize_base={quantize_base}",
    ]


def lora_llama3_test_config(
    lora_attn_modules,
    apply_lora_to_mlp: bool = False,
    apply_lora_to_output: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 16,
    quantize_base: bool = False,
) -> List[str]:
    return [
        # Note: we explicitly use _component_ so that we can also call
        # config.instantiate directly for easier comparison
        "model._component_=torchtune.models.llama3.lora_llama3",
        f"model.lora_attn_modules={lora_attn_modules}",
        f"model.apply_lora_to_mlp={apply_lora_to_mlp}",
        f"model.apply_lora_to_output={apply_lora_to_output}",
        "model.vocab_size=128_256",
        "model.num_layers=2",
        "model.num_heads=8",
        "model.embed_dim=64",
        "model.max_seq_len=1024",
        "model.norm_eps=1e-5",
        "model.num_kv_heads=4",
        f"model.lora_rank={lora_rank}",
        f"model.lora_alpha={lora_alpha}",
        "model.lora_dropout=0.0",
        f"model.quantize_base={quantize_base}",
    ]


def write_hf_ckpt_config(ckpt_dir: str):
    config = {
        "hidden_size": 256,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
    }
    config_file = Path.joinpath(Path(ckpt_dir), "config.json")
    with config_file.open("w") as f:
        json.dump(config, f)


MODEL_TEST_CONFIGS = {
    "llama2": llama2_test_config(),
    "llama3": llama3_test_config(),
    "llama2_lora": lora_llama2_test_config(
        lora_attn_modules=["q_proj", "k_proj", "v_proj", "output_proj"],
        apply_lora_to_mlp=False,
        apply_lora_to_output=False,
        lora_rank=8,
        lora_alpha=16,
    ),
    "llama2_qlora": lora_llama2_test_config(
        lora_attn_modules=["q_proj", "k_proj", "v_proj", "output_proj"],
        apply_lora_to_mlp=True,
        apply_lora_to_output=False,
        lora_rank=8,
        lora_alpha=16,
        quantize_base=True,
    ),
    "llama3_lora": lora_llama3_test_config(
        lora_attn_modules=["q_proj", "k_proj", "v_proj", "output_proj"],
        apply_lora_to_mlp=False,
        apply_lora_to_output=False,
        lora_rank=8,
        lora_alpha=16,
    ),
}
