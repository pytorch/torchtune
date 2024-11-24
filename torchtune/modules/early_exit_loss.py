# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from enum import Enum

import numpy as np
import torch

from torchtune import utils

log = utils.get_logger("DEBUG")


class LossScaleType(str, Enum):
    ONE = "one"
    L = "l"
    SUM_L = "sum_l"
    INV_L = "inv_l"
    SQRT_L = "sqrt_l"
    INV_SQRT_L = "inv_sqrt_l"


# TODO: create docstring using other functions as template
# TODO: add assert on type of loss_fn
def early_exit_loss(
    model,
    hidden_states_dict,
    labels,
    loss_fn,
    e_scale: float = 1.0,
    loss_scale_type=LossScaleType.SUM_L,
):
    batch_loss_fn = copy.deepcopy(loss_fn)
    batch_loss_fn.reduction = "none"

    e = len(hidden_states_dict)
    # List of e tensors with shape [b, s, d]
    hidden_states = tuple(hidden_states_dict.values())
    hidden_layer_ids = tuple(hidden_states_dict.keys())
    # Shape: [e, b, s, d]
    hidden_states_stacked = torch.stack(hidden_states)
    # Shape: [e, b, s, out_dim]
    logits_early = model.unembed(hidden_states_stacked)
    # Shape: [e*b*s, out_dim]
    logits_early = logits_early.reshape(-1, logits_early.size(-1))
    logits_early = logits_early.contiguous()
    # Shape: [e*b*s]
    labels_repeated = labels.repeat(e, 1).reshape(-1)
    # Compute early losses: Shape: [e*b*s]
    losses_early = batch_loss_fn(logits_early, labels_repeated)
    # Shape: [e, b*s]
    losses_early = losses_early.view(e, -1)
    # Shape: [e]
    s_unpadded = (labels != loss_fn.ignore_index).sum()
    losses_early = losses_early.float().sum(-1) / s_unpadded
    # Shape: [e]
    losses_scales = layer_ids_to_loss_scales(
        torch.Tensor(hidden_layer_ids).to(losses_early),
        len(model.layers),
        loss_scale_type,
        e_scale,
    )

    return torch.sum(losses_scales * losses_early)


def layer_ids_to_loss_scales(
    layer_ids, n_layers, loss_scale_type: LossScaleType, e_scale: float
):
    if loss_scale_type == LossScaleType.ONE:
        loss_scales = torch.ones(len(layer_ids))
    elif loss_scale_type == LossScaleType.L:
        loss_scales = torch.Tensor(layer_ids + 1)
    elif loss_scale_type == LossScaleType.SUM_L:
        loss_scales = torch.cumsum(layer_ids + 1, dim=0)
    elif loss_scale_type == LossScaleType.SQRT_L:
        loss_scales = torch.sqrt(layer_ids + 1)
    elif loss_scale_type == LossScaleType.INV_L:
        loss_scales = 1.0 / (layer_ids + 1)
    elif loss_scale_type == LossScaleType.INV_SQRT_L:
        loss_scales = torch.reciprocal(torch.sqrt(layer_ids + 1))
    else:
        raise ValueError(f"Unsupported loss_scale type {loss_scale_type}")

    loss_scales = loss_scales * torch.where(layer_ids < n_layers - 1, e_scale, 1.0)
    # normalize loss scales to ensure that their sum is 1.0
    loss_scales = loss_scales / torch.sum(loss_scales)
    assert torch.isclose(torch.sum(loss_scales), torch.Tensor([1.0]).to(loss_scales))

    return loss_scales


class EarlyExitCurriculumType(str, Enum):
    NONE = "none"
    ROTATIONAL = "rot"
    GRADUAL = "gradual"


def setup_early_exit_loss_curriculum(
    early_exit_curriculum: EarlyExitCurriculumType, *args, **kwargs
):
    if early_exit_curriculum == EarlyExitCurriculumType.NONE:
        return None
    elif early_exit_curriculum == EarlyExitCurriculumType.ROTATIONAL:
        return RotationalEarlyExitCurriculum(*args, **kwargs)
    elif early_exit_curriculum == EarlyExitCurriculumType.GRADUAL:
        return GradualEarlyExitCurriculum(*args, **kwargs)
    else:
        raise ValueError(f"Unsupported early loss curriculum {early_exit_curriculum}.")


# TODO: create a base curriculum class that can be used for other aspects, e.g., dropout, datasets, etc.
class EarlyExitCurriculum:
    def __init__(
        self, do_output_hidden_states, max_steps, train_last_layer=True, verbose=False
    ):
        self._init_do_output_hidden_states = do_output_hidden_states
        self.do_output_hidden_states = do_output_hidden_states
        self.train_last_layer = train_last_layer
        self.verbose = verbose
        self.max_steps = max_steps

    def step(self):
        pass

    def get(self):
        do_output_hidden_states = np.copy(self.do_output_hidden_states)
        # Ensure last layer is trained
        if self.train_last_layer:
            do_output_hidden_states[-1] = True
        return do_output_hidden_states


class RotationalEarlyExitCurriculum(EarlyExitCurriculum):
    def __init__(
        self, do_output_hidden_states, max_steps, train_last_layer=True, verbose=False
    ):
        super().__init__(do_output_hidden_states, max_steps, train_last_layer, verbose)
        self._initial_do_output_hidden_states = np.copy(do_output_hidden_states)

    def step(self):
        # Rotate layer enablement one step forward
        self.do_output_hidden_states = np.roll(self.do_output_hidden_states, 1)

        if self.verbose:
            log.info(
                f"Updated self.output_hidden_states to {self.do_output_hidden_states}."
            )


class GradualEarlyExitCurriculum(EarlyExitCurriculum):
    def __init__(
        self,
        do_output_hidden_states,
        max_steps,
        train_last_layer=True,
        percent_scale=2,
        verbose=False,
    ):
        super().__init__(do_output_hidden_states, max_steps, train_last_layer, verbose)
        self._final_do_output_hidden_states = np.copy(do_output_hidden_states)
        self._step = 0
        self._percent_scale = percent_scale

        # Initialize all layers to False
        for i in range(len(self.do_output_hidden_states)):
            self.do_output_hidden_states[i] = False

    def step(self):
        percent_trained = self._step / self.max_steps
        n_layers = len(self.do_output_hidden_states)
        # Enable each layer based on proportion of completed training steps
        for layer_index in range(len(self.do_output_hidden_states)):
            should_train = (percent_trained * self._percent_scale) >= (
                n_layers - layer_index
            ) / n_layers
            self.do_output_hidden_states[layer_index] = should_train

        # Only enable layers that are set by the user
        self.do_output_hidden_states = np.logical_and(
            self.do_output_hidden_states, self._final_do_output_hidden_states
        )

        self._step += 1
        if self.verbose:
            log.info(
                f"Updated self.do_output_hidden_states to {self.do_output_hidden_states}."
            )
