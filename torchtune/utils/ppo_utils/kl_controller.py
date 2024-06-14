# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np


class AdaptiveKLController:
    """
    A class that implements an adaptive KL controller from https://arxiv.org/pdf/1909.08593.pdf.

    Attributes:
        init_kl_coef (float): The initial KL loss contribution coefficient value.
        target (float): The target KL value.
        horizon (int): The horizon for the update calculation.

    """

    def __init__(self, init_kl_coef: float, target: float, horizon: float):
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon

    def update(self, current: float, n_steps: int) -> None:
        """
        Updates the KL coefficient value based on the current KL value and the number of steps.

        Args:
            current (float): The current KL value.
            n_steps (int): The number of steps.
        """
        target = self.target
        proportional_error = np.clip(current / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """
    A class for using a fixed KL coefficient.

    Attributes:
        kl_coef (float): The fixed KL loss contribution coefficient value.

    """

    def __init__(self, kl_coef: float):
        self.value = kl_coef

    def update(self, current: float, n_steps: int) -> None:
        """
        Does nothing.
        """
        pass
