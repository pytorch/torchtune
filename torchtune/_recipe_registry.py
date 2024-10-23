# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import List


@dataclass
class Config:
    name: str
    file_path: str


@dataclass
class Recipe:
    name: str
    file_path: str
    configs: List[Config]
    supports_distributed: bool


_ALL_RECIPES = [
    Recipe(
        name="full_finetune_single_device",
        file_path="full_finetune_single_device.py",
        configs=[
            Config(
                name="llama2/7B_full_low_memory",
                file_path="llama2/7B_full_low_memory.yaml",
            ),
            Config(
                name="code_llama2/7B_full_low_memory",
                file_path="code_llama2/7B_full_low_memory.yaml",
            ),
            Config(
                name="llama3/8B_full_single_device",
                file_path="llama3/8B_full_single_device.yaml",
            ),
            Config(
                name="llama3_1/8B_full_single_device",
                file_path="llama3_1/8B_full_single_device.yaml",
            ),
            Config(
                name="llama3_2/1B_full_single_device",
                file_path="llama3_2/1B_full_single_device.yaml",
            ),
            Config(
                name="llama3_2/3B_full_single_device",
                file_path="llama3_2/3B_full_single_device.yaml",
            ),
            Config(
                name="mistral/7B_full_low_memory",
                file_path="mistral/7B_full_low_memory.yaml",
            ),
            Config(
                name="phi3/mini_full_low_memory",
                file_path="phi3/mini_full_low_memory.yaml",
            ),
            Config(
                name="qwen2/7B_full_single_device",
                file_path="qwen2/7B_full_single_device.yaml",
            ),
            Config(
                name="qwen2/0.5B_full_single_device",
                file_path="qwen2/0.5B_full_single_device.yaml",
            ),
            Config(
                name="qwen2/1.5B_full_single_device",
                file_path="qwen2/1.5B_full_single_device.yaml",
            ),
            Config(
                name="llama3_2_vision/11B_full_single_device",
                file_path="llama3_2_vision/11B_full_single_device.yaml",
            ),
        ],
        supports_distributed=False,
    ),
    Recipe(
        name="full_finetune_distributed",
        file_path="full_finetune_distributed.py",
        configs=[
            Config(name="llama2/7B_full", file_path="llama2/7B_full.yaml"),
            Config(name="llama2/13B_full", file_path="llama2/13B_full.yaml"),
            Config(name="llama3/8B_full", file_path="llama3/8B_full.yaml"),
            Config(name="llama3_1/8B_full", file_path="llama3_1/8B_full.yaml"),
            Config(name="llama3_2/1B_full", file_path="llama3_2/1B_full.yaml"),
            Config(name="llama3_2/3B_full", file_path="llama3_2/3B_full.yaml"),
            Config(name="llama3/70B_full", file_path="llama3/70B_full.yaml"),
            Config(name="llama3_1/70B_full", file_path="llama3_1/70B_full.yaml"),
            Config(name="mistral/7B_full", file_path="mistral/7B_full.yaml"),
            Config(name="gemma/2B_full", file_path="gemma/2B_full.yaml"),
            Config(name="gemma/7B_full", file_path="gemma/7B_full.yaml"),
            Config(name="phi3/mini_full", file_path="phi3/mini_full.yaml"),
            Config(name="qwen2/7B_full", file_path="qwen2/7B_full.yaml"),
            Config(name="qwen2/0.5B_full", file_path="qwen2/0.5B_full.yaml"),
            Config(name="qwen2/1.5B_full", file_path="qwen2/1.5B_full.yaml"),
            Config(
                name="llama3_2_vision/11B_full",
                file_path="llama3_2_vision/11B_full.yaml",
            ),
        ],
        supports_distributed=True,
    ),
    Recipe(
        name="lora_finetune_single_device",
        file_path="lora_finetune_single_device.py",
        configs=[
            Config(
                name="llama2/7B_lora_single_device",
                file_path="llama2/7B_lora_single_device.yaml",
            ),
            Config(
                name="llama2/7B_qlora_single_device",
                file_path="llama2/7B_qlora_single_device.yaml",
            ),
            Config(
                name="code_llama2/7B_lora_single_device",
                file_path="code_llama2/7B_lora_single_device.yaml",
            ),
            Config(
                name="code_llama2/7B_qlora_single_device",
                file_path="code_llama2/7B_qlora_single_device.yaml",
            ),
            Config(
                name="llama3/8B_lora_single_device",
                file_path="llama3/8B_lora_single_device.yaml",
            ),
            Config(
                name="llama3_1/8B_lora_single_device",
                file_path="llama3_1/8B_lora_single_device.yaml",
            ),
            Config(
                name="llama3/8B_qlora_single_device",
                file_path="llama3/8B_qlora_single_device.yaml",
            ),
            Config(
                name="llama3_2/1B_lora_single_device",
                file_path="llama3_2/1B_lora_single_device.yaml",
            ),
            Config(
                name="llama3_2/3B_lora_single_device",
                file_path="llama3_2/3B_lora_single_device.yaml",
            ),
            Config(
                name="llama3/8B_dora_single_device",
                file_path="llama3/8B_dora_single_device.yaml",
            ),
            Config(
                name="llama3/8B_qdora_single_device",
                file_path="llama3/8B_qdora_single_device.yaml",
            ),
            Config(
                name="llama3_1/8B_qlora_single_device",
                file_path="llama3_1/8B_qlora_single_device.yaml",
            ),
            Config(
                name="llama3_2/1B_qlora_single_device",
                file_path="llama3_2/1B_qlora_single_device.yaml",
            ),
            Config(
                name="llama3_2/3B_qlora_single_device",
                file_path="llama3_2/3B_qlora_single_device.yaml",
            ),
            Config(
                name="llama2/13B_qlora_single_device",
                file_path="llama2/13B_qlora_single_device.yaml",
            ),
            Config(
                name="mistral/7B_lora_single_device",
                file_path="mistral/7B_lora_single_device.yaml",
            ),
            Config(
                name="mistral/7B_qlora_single_device",
                file_path="mistral/7B_qlora_single_device.yaml",
            ),
            Config(
                name="gemma/2B_lora_single_device",
                file_path="gemma/2B_lora_single_device.yaml",
            ),
            Config(
                name="gemma/2B_qlora_single_device",
                file_path="gemma/2B_qlora_single_device.yaml",
            ),
            Config(
                name="gemma/7B_lora_single_device",
                file_path="gemma/7B_lora_single_device.yaml",
            ),
            Config(
                name="gemma/7B_qlora_single_device",
                file_path="gemma/7B_qlora_single_device.yaml",
            ),
            Config(
                name="phi3/mini_lora_single_device",
                file_path="phi3/mini_lora_single_device.yaml",
            ),
            Config(
                name="phi3/mini_qlora_single_device",
                file_path="phi3/mini_qlora_single_device.yaml",
            ),
            Config(
                name="qwen2/7B_lora_single_device",
                file_path="qwen2/7B_lora_single_device.yaml",
            ),
            Config(
                name="qwen2/0.5B_lora_single_device",
                file_path="qwen2/0.5B_lora_single_device.yaml",
            ),
            Config(
                name="qwen2/1.5B_lora_single_device",
                file_path="qwen2/1.5B_lora_single_device.yaml",
            ),
            Config(
                name="llama3_2_vision/11B_lora_single_device",
                file_path="llama3_2_vision/11B_lora_single_device.yaml",
            ),
            Config(
                name="llama3_2_vision/11B_qlora_single_device",
                file_path="llama3_2_vision/11B_qlora_single_device.yaml",
            ),
        ],
        supports_distributed=False,
    ),
    Recipe(
        name="lora_dpo_single_device",
        file_path="lora_dpo_single_device.py",
        configs=[
            Config(
                name="llama2/7B_lora_dpo_single_device",
                file_path="llama2/7B_lora_dpo_single_device.yaml",
            ),
        ],
        supports_distributed=False,
    ),
    Recipe(
        name="lora_dpo_distributed",
        file_path="lora_dpo_distributed.py",
        configs=[
            Config(
                name="llama2/7B_lora_dpo",
                file_path="llama2/7B_lora_dpo.yaml",
            ),
        ],
        supports_distributed=True,
    ),
    Recipe(
        name="ppo_full_finetune_single_device",
        file_path="ppo_full_finetune_single_device.py",
        configs=[
            Config(
                name="mistral/7B_full_ppo_low_memory",
                file_path="mistral/7B_full_ppo_low_memory.yaml",
            ),
        ],
        supports_distributed=False,
    ),
    Recipe(
        name="lora_finetune_distributed",
        file_path="lora_finetune_distributed.py",
        configs=[
            Config(name="llama2/7B_lora", file_path="llama2/7B_lora.yaml"),
            Config(name="llama2/13B_lora", file_path="llama2/13B_lora.yaml"),
            Config(name="llama2/70B_lora", file_path="llama2/70B_lora.yaml"),
            Config(
                name="llama2/7B_qlora",
                file_path="llama2/7B_qlora.yaml",
            ),
            Config(
                name="llama2/70B_qlora",
                file_path="llama2/70B_qlora.yaml",
            ),
            Config(name="llama3/8B_dora", file_path="llama3/8B_dora.yaml"),
            Config(name="llama3/70B_lora", file_path="llama3/70B_lora.yaml"),
            Config(name="llama3_1/70B_lora", file_path="llama3_1/70B_lora.yaml"),
            Config(name="llama3/8B_lora", file_path="llama3/8B_lora.yaml"),
            Config(name="llama3_1/8B_lora", file_path="llama3_1/8B_lora.yaml"),
            Config(name="llama3_2/1B_lora", file_path="llama3_2/1B_lora.yaml"),
            Config(name="llama3_2/3B_lora", file_path="llama3_2/3B_lora.yaml"),
            Config(
                name="llama3_1/405B_qlora",
                file_path="llama3_1/405B_qlora.yaml",
            ),
            Config(name="mistral/7B_lora", file_path="mistral/7B_lora.yaml"),
            Config(name="gemma/2B_lora", file_path="gemma/2B_lora.yaml"),
            Config(name="gemma/7B_lora", file_path="gemma/7B_lora.yaml"),
            Config(name="phi3/mini_lora", file_path="phi3/mini_lora.yaml"),
            Config(name="qwen2/7B_lora", file_path="qwen2/7B_lora.yaml"),
            Config(name="qwen2/0.5B_lora", file_path="qwen2/0.5B_lora.yaml"),
            Config(name="qwen2/1.5B_lora", file_path="qwen2/1.5B_lora.yaml"),
            Config(
                name="llama3_2_vision/11B_lora",
                file_path="llama3_2_vision/11B_lora.yaml",
            ),
            Config(
                name="llama3_2_vision/11B_qlora",
                file_path="llama3_2_vision/11B_qlora.yaml",
            ),
        ],
        supports_distributed=True,
    ),
    Recipe(
        name="generate",
        file_path="generate.py",
        configs=[
            Config(name="generation", file_path="generation.yaml"),
        ],
        supports_distributed=False,
    ),
    Recipe(
        name="dev/generate_v2",
        file_path="dev/generate_v2.py",
        configs=[
            Config(
                name="llama2/generation_v2",
                file_path="llama2/generation_v2.yaml",
            ),
            Config(
                name="llama3_2_vision/generation_v2",
                file_path="llama3_2_vision/generation_v2.yaml",
            ),
        ],
        supports_distributed=False,
    ),
    Recipe(
        name="eleuther_eval",
        file_path="eleuther_eval.py",
        configs=[
            Config(name="eleuther_evaluation", file_path="eleuther_evaluation.yaml"),
            Config(
                name="llama3_2_vision/evaluation",
                file_path="llama3_2_vision/evaluation.yaml",
            ),
            Config(
                name="qwen2/evaluation",
                file_path="qwen2/evaluation.yaml",
            ),
            Config(
                name="gemma/evaluation",
                file_path="gemma/evaluation.yaml",
            ),
            Config(
                name="phi3/evaluation",
                file_path="phi3/evaluation.yaml",
            ),
            Config(
                name="mistral/evaluation",
                file_path="mistral/evaluation.yaml",
            ),
        ],
        supports_distributed=False,
    ),
    Recipe(
        name="quantize",
        file_path="quantize.py",
        configs=[
            Config(name="quantization", file_path="quantization.yaml"),
        ],
        supports_distributed=False,
    ),
    Recipe(
        name="qat_distributed",
        file_path="qat_distributed.py",
        configs=[
            Config(name="llama2/7B_qat_full", file_path="llama2/7B_qat_full.yaml"),
            Config(name="llama3/8B_qat_full", file_path="llama3/8B_qat_full.yaml"),
        ],
        supports_distributed=True,
    ),
    Recipe(
        name="knowledge_distillation_single_device",
        file_path="knowledge_distillation_single_device.py",
        configs=[
            Config(
                name="qwen2/knowledge_distillation_single_device",
                file_path="qwen2/knowledge_distillation_single_device.yaml",
            ),
            Config(
                name="llama3_2/knowledge_distillation_single_device",
                file_path="llama3_2/knowledge_distillation_single_device.yaml",
            ),
        ],
        supports_distributed=False,
    ),
]


def get_all_recipes():
    """List of recipes available from the CLI."""
    return _ALL_RECIPES
