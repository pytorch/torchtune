# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
from omegaconf import DictConfig
from torchtune.config import instantiate
from torchtune.config._utils import _get_component_from_path
from torchtune.modules.peft import DoRALinear, LoRALinear


def classifier_model(num_classes, base_model: str, base_model_config):
    model = _get_component_from_path(base_model)(**base_model_config)
    del model.output.weight
    model.output = nn.Linear(model.head_dim * model.num_heads, num_classes, bias=False)
    return model


def lora_classifier_model(
    num_classes,
    apply_lora_to_output: bool = False,
    *,
    lora_rank: int = 8,
    lora_alpha: float = 16,
    lora_dropout: float = 0.0,
    use_dora: bool = False,
    **model_kwargs,
):
    if model_kwargs.get("apply_lora_to_output", False):
        model_kwargs.pop("apply_lora_to_output")
        print("Applying LoRA to new output layer!")
    base_model_path = model_kwargs.pop("base_model")
    model = _get_component_from_path(base_model_path)(
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        **model_kwargs,
    )
    adapter_cls = DoRALinear if use_dora else LoRALinear
    embed_dim = model.head_dim * model.num_heads
    model.output = (
        adapter_cls(
            embed_dim,
            num_classes,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
        )
        if apply_lora_to_output
        else nn.Linear(embed_dim, num_classes, bias=False)
    )
    return model

    # def lora_llama3_1_classifier(
    #     num_classes: int = 1,
    #     apply_lora_to_output: bool = False,
    #     *,
    #     embed_dim: int,
    #     lora_rank: int,
    #     lora_alpha: float,
    #     lora_dropout: float = 0.0,
    #     use_dora: bool = False,
    #     **kwargs,
    # ):
    #     model = lora_llama3_1(apply_lora_to_output=apply_lora_to_output, use_dora=use_dora, **kwargs)
    #     adapter_cls = DoRALinear if use_dora else LoRALinear
    model.output = (
        adapter_cls(
            embed_dim,
            num_classes,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
        )
        if apply_lora_to_output
        else nn.Linear(embed_dim, num_classes, bias=False)
    )


import torch
from omegaconf import DictConfig, OmegaConf
from torchtune import config
from torchtune.models import classifier_model, lora_classifier_model

conf = DictConfig(
    {
        "model": {
            "_component_": "torchtune.models.lora_classifier_model",
            "num_classes": 1,
            "base_model": "torchtune.models.llama3_2.lora_llama3_2_1b",
            "lora_attn_modules": ["q_proj"],
            "apply_lora_to_output": False,
            "lora_rank": 8,
            "lora_alpha": 16,
        }
    }
)

torch.set_default_device(torch.device("mps"))
torch.set_default_dtype(torch.bfloat16)
model = config.instantiate(conf["model"])
