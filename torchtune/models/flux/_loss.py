# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import nullcontext
from typing import Any, ContextManager, Dict

import torch
from torch import Tensor

from torchtune.models.flux._flow_model import FluxFlowModel
from torchtune.models.flux._utils import predict_noise


class FluxLossStep:
    def __init__(self, guidance: float):
        self._guidance = guidance

    def __call__(
        self,
        model: FluxFlowModel,
        batch: Dict[str, Any],
        model_ctx: ContextManager = nullcontext(),
    ) -> Tensor:
        latents = batch["img_encoding"]
        clip_encodings = batch["clip_text_encoding"]
        t5_encodings = batch["t5_text_encoding"]

        bsz = latents.shape[0]

        with torch.no_grad():
            noise = torch.randn_like(latents)
            timesteps = torch.rand((bsz,)).to(latents)
            sigmas = timesteps.view(-1, 1, 1, 1)
            noisy_latents = (1 - sigmas) * latents + sigmas * noise
            guidance = torch.full((bsz,), self._guidance).to(latents)

        pred = predict_noise(
            model,
            noisy_latents,
            clip_encodings,
            t5_encodings,
            timesteps,
            guidance,
            model_ctx,
        )

        target = noise - latents
        loss = torch.nn.functional.mse_loss(pred.float(), target.float().detach())
        return loss
