# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Protocol

import torch

from omegaconf import OmegaConf

from torchtune import config
from torchtune.utils import get_logger

logger = get_logger("DEBUG")


class _FederatorInterface(Protocol):
    """
    Interface implemented by Federators in torchtune.
    """

    def forward(self, models) -> None:
        ...

    def step(self) -> None:
        ...


class DiLoCoFederator(_FederatorInterface):
    """
    Federator for testing purposes.
    """

    def __init__(
        self,
        participants: List[str],
        optimizer: Dict[str, Any],
        model: torch.nn.Module,
        h: int = 1000,
        batch_size: int = 1,
    ) -> None:
        self._optimizer = config.instantiate(
            OmegaConf.create(optimizer), model.parameters()
        )
        self._model = model
        self._participants = participants.copy()
        self._h = h
        self._batch_size = batch_size

        self._zero_grad()

        logger.info("DiLiCoFederator initialized.")

    def _zero_grad(self):
        for param in self._model.parameters():
            param.grad = torch.zeros_like(param)

    """
    """

    def forward(self, models) -> None:
        for orig_param, *model_params in zip(
            self._model.parameters(), *[m.values() for m in models]
        ):
            diffs = [
                (orig_param.detach() - m_param.detach()) for m_param in model_params
            ]
            diff_sum = sum(diffs)
            avg_diff = diff_sum / len(self._participants)

            with torch.no_grad():
                orig_param.grad += avg_diff

    @torch.no_grad()
    def step(self) -> None:
        self._optimizer.step()
        self._zero_grad()
