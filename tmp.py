# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import matplotlib.pyplot as plt
import torch
from torchtune import utils
from torchtune.models import convert_weights

from torchtune.models.llama3 import llama3_8b
from torchtune.utils import FullModelMetaCheckpointer


CKPT_DIR = "/home/cpelletier/model/llama3-8b/original/"


base_model = llama3_8b()
checkpointer = FullModelMetaCheckpointer(
    checkpoint_dir=CKPT_DIR,
    checkpoint_files=["consolidated.00.pth"],
    recipe_checkpoint=None,
    output_dir=CKPT_DIR,
    model_type="LLAMA3",
    resume_from_checkpoint=False,
)
ckpt = checkpointer.load_checkpoint()
base_model.load_state_dict(ckpt[utils.MODEL_KEY], strict=False)
base_model.eval()

full_model = llama3_8b()
sd = torch.load(
    "/home/cpelletier/out/full/meta_model_0.pt",
    map_location="cpu",
    mmap=True,
    weights_only=True,
)
sd = convert_weights.meta_to_tune(sd)
full_model.load_state_dict(sd)
full_model.eval()

lora_model = llama3_8b()
sd = torch.load(
    "/home/cpelletier/out/lora/meta_model_0.pt",
    map_location="cpu",
    mmap=True,
    weights_only=True,
)
sd = convert_weights.meta_to_tune(sd)
lora_model.load_state_dict(sd)
lora_model.eval()

dora_model = llama3_8b()
sd = torch.load(
    "/home/cpelletier/out/dora/meta_model_0.pt",
    map_location="cpu",
    mmap=True,
    weights_only=True,
)
sd = convert_weights.meta_to_tune(sd)
dora_model.load_state_dict(sd)
dora_model.eval()


@torch.no_grad
def analyze(base_model, ft_model):
    mag_diffs = []
    dir_diffs = []

    cos = torch.nn.CosineSimilarity(dim=1)
    for base_layer, ft_layer in zip(base_model.layers, ft_model.layers):
        base_weight = base_layer.attn.q_proj.weight
        ft_weight = ft_layer.attn.q_proj.weight

        base_mag = torch.linalg.norm(base_weight, dim=1)
        ft_mag = torch.linalg.norm(ft_weight, dim=1)
        mag_diff = torch.mean(torch.abs(ft_mag - base_mag))
        mag_diffs.append(mag_diff.item())

        ft_dir = ft_weight / ft_mag.expand_as(ft_weight)
        base_dir = base_weight / base_mag.expand_as(base_weight)
        sim = cos(ft_dir, base_dir)
        dir_diff = torch.mean(1 - sim)
        dir_diffs.append(dir_diff.item())

    return dir_diffs, mag_diffs


dir_diffs, mag_diffs = analyze(base_model, full_model)
plt.scatter(dir_diffs, mag_diffs)
plt.savefig("/home/cpelletier/tmp/full.png")
plt.clf()

dir_diffs, mag_diffs = analyze(base_model, lora_model)
plt.scatter(dir_diffs, mag_diffs)
plt.savefig("/home/cpelletier/tmp/lora.png")
plt.clf()

dir_diffs, mag_diffs = analyze(base_model, dora_model)
plt.scatter(dir_diffs, mag_diffs)
plt.savefig("/home/cpelletier/tmp/dora.png")
plt.clf()
