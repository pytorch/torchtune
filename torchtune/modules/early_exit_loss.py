# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import torch

from torchtune import utils
from torchtune.modules.transformer import TransformerDecoder

log = utils.get_logger("DEBUG")


class LossScaleType(str, Enum):
    ONE = "one"
    L = "l"
    SUM_L = "sum_l"
    INV_L = "inv_l"
    SQRT_L = "sqrt_l"
    INV_SQRT_L = "inv_sqrt_l"


def early_exit_loss(
    model: TransformerDecoder,
    hidden_states_dict: Dict[int, torch.Tensor],
    labels: torch.Tensor,
    loss_fn: torch.nn.Module,
    e_scale: float = 1.0,
    loss_scale_type: LossScaleType = LossScaleType.SUM_L,
) -> torch.Tensor:
    """
    Compute the early exit loss for a given model and outputs of intermediate layers.
    This function takes in a model, a dictionary of hidden states, labels, a loss function,
    and optional parameters for scaling the loss. It computes the early exit loss by
    iterating over the hidden states, computing the logits and losses at each layer,
    and then scaling and summing these losses.
    Args:
        model (TransformerDecoder): The model to compute the early exit loss for.
        hidden_states_dict (Dict[int, torch.Tensor]): A dictionary of hidden states,
            where each key is a layer index and each value is a tensor of shape [b, s, d].
        labels (torch.Tensor): The labels for the input data.
        loss_fn (torch.nn.Module): The loss function to use (should be the same as the standard loss function for last layer).
        e_scale (float, optional): A scaling factor for the early exit losses. Defaults to 1.0.
        loss_scale_type (LossScaleType, optional): The type of loss scaling to use to determine
            scale of each layer's loss. Defaults to LossScaleType.SUM_L.
    Returns:
        torch.Tensor: The computed early exit loss.
    """
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
    layer_ids: torch.Tensor,
    n_layers: int,
    loss_scale_type: LossScaleType,
    e_scale: float,
) -> torch.Tensor:
    """
    Compute the loss scales for a given set of layer IDs and loss scale type.
    This function takes in a list of layer IDs, the total number of layers,
    a loss scale type, and an early exit scaling factor. It computes the loss
    scales based on the specified loss scale type and then normalizes them to
    ensure that their sum is 1.0.
    Args:
        layer_ids (torch.Tensor): A tensor of layer IDs.
        n_layers (int): The total number of layers.
        loss_scale_type (LossScaleType): The type of loss scaling to use.
        e_scale (float): An early exit scaling factor.
    Returns:
        torch.Tensor: The computed loss scales.
    Raises:
        ValueError: If the provided loss scale type is not supported.
        AssertionError: If the sum of the loss scales is not close to 1.0.
    Example:
        >>> layer_ids = [0, 1, 2]
        >>> n_layers = 3
        >>> loss_scale_type = LossScaleType.SUM_L
        >>> e_scale = 1.0
        >>> loss_scales = layer_ids_to_loss_scales(layer_ids, n_layers, loss_scale_type, e_scale)
    """
    if loss_scale_type == LossScaleType.ONE:
        loss_scales = torch.ones(len(layer_ids), device=layer_ids.device)
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


# TODO: create a base curriculum class that can be used for other aspects, e.g., dropout, datasets, etc.
class EarlyExitCurriculum:
    """
    A curriculum for early exit loss training, which controls which layers to use their hidden states
    during training.
    Args:
        do_output_hidden_states (List[bool]): A list indicating whether each layer's hidden state
            should be output to calculate their losses.
        max_steps (int): The maximum number of steps in the curriculum.
        train_last_layer (bool, optional): Whether to always calculate loss for the last layer. Defaults to True.
        last_step (Optional[int]): The last step the curriculum stopped at in a previous run. This is used when resuming training.
        verbose (bool, optional): Whether to print verbose logs. Defaults to False.
    """

    def __init__(
        self,
        do_output_hidden_states: List[bool],
        max_steps: int,
        train_last_layer: bool = True,
        last_step: Optional[int] = None,
        verbose: bool = False,
    ):
        self._init_do_output_hidden_states = do_output_hidden_states
        self._do_output_hidden_states = do_output_hidden_states
        self.train_last_layer = train_last_layer
        self.verbose = verbose
        self.max_steps = max_steps
        self._step = 0 if last_step is None else last_step

    def step(self) -> None:
        """
        Perform a step in the curriculum. Should be called at the end of each iteration during training.
        """
        pass

    def get(self) -> np.ndarray:
        """
        Get the current output hidden states.
        Returns:
            np.ndarray: A list indicating whether we should calculate loss for each layer.
        """
        do_output_hidden_states = np.copy(self._do_output_hidden_states)
        # Ensure last layer is trained
        if self.train_last_layer:
            do_output_hidden_states[-1] = True
        return do_output_hidden_states


class RotationalEarlyExitCurriculum(EarlyExitCurriculum):
    """
    A rotational early exit curriculum, which rotates the layer enablement one step forward
    at each step.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._initial_do_output_hidden_states = np.copy(self._do_output_hidden_states)

    def step(self):
        """
        Rotate the layer enablement one step forward.
        This method updates the `do_output_hidden_states` attribute by rotating it one position to the right.
        """
        # Rotate layer enablement one step forward
        self._do_output_hidden_states = np.roll(self._do_output_hidden_states, 1)

        self._step += 1
        if self.verbose:
            log.info(
                f"Updated self._do_output_hidden_states to {self._do_output_hidden_states}."
            )


class GradualEarlyExitCurriculum(EarlyExitCurriculum):
    """
    A gradual early exit curriculum, which gradually enables more layers (starting from the last layer) as training progresses.
    Args:
        *args: Positional arguments passed to the parent EarlyExitCurriculum class.
        percent_scale (float, optional): A scaling factor to determine at which percentage
            of steps, all the layers will be enabled. At `steps = max_steps / percent_scale`, all the layers will be enabled.
        **kwargs: Keyword arguments passed to the parent EarlyExitCurriculum class.
    """

    def __init__(
        self,
        *args,
        percent_scale: float = 2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._final_do_output_hidden_states = np.copy(self._do_output_hidden_states)
        self._step = 0
        self._percent_scale = percent_scale

        # Initialize all layers to False
        for i in range(len(self._do_output_hidden_states)):
            self._do_output_hidden_states[i] = False

    def step(self):
        """
        Perform a step in the curriculum.
        This method updates the `_do_output_hidden_states` attribute based on the current
            step and the percentage of completed training steps.
        """
        percent_trained = self._step / self.max_steps
        n_layers = len(self._do_output_hidden_states)
        # Enable each layer based on proportion of completed training steps
        for layer_index in range(len(self._do_output_hidden_states)):
            should_train = (percent_trained * self._percent_scale) >= (
                n_layers - layer_index
            ) / n_layers
            self._do_output_hidden_states[layer_index] = should_train

        # Only enable layers that are set by the user
        self._do_output_hidden_states = np.logical_and(
            self._do_output_hidden_states, self._final_do_output_hidden_states
        )

        self._step += 1
        if self.verbose:
            log.info(
                f"Updated self._do_output_hidden_states to {self._do_output_hidden_states}."
            )


def setup_early_exit_loss_curriculum(
    early_exit_curriculum: EarlyExitCurriculumType, *args, **kwargs
) -> Optional[EarlyExitCurriculum]:
    """
    Set up an early exit loss curriculum based on the provided type.
    This function takes in an early exit curriculum type and optional arguments.
    It returns an instance of the corresponding early exit curriculum class,
    or None if the curriculum type is NONE.
    Args:
        early_exit_curriculum (EarlyExitCurriculumType): The type of early exit curriculum to set up.
        *args: Optional positional arguments for the early exit curriculum constructor.
        **kwargs: Optional keyword arguments for the early exit curriculum constructor.
    Returns:
        Optional[EarlyExitCurriculum]:
            An instance of the corresponding early exit curriculum class, or None.
    Raises:
        ValueError: If the provided early exit curriculum type is not supported.
    """
    if early_exit_curriculum == EarlyExitCurriculumType.NONE:
        return None
    elif early_exit_curriculum == EarlyExitCurriculumType.ROTATIONAL:
        return RotationalEarlyExitCurriculum(*args, **kwargs)
    elif early_exit_curriculum == EarlyExitCurriculumType.GRADUAL:
        return GradualEarlyExitCurriculum(*args, **kwargs)
    else:
        raise ValueError(f"Unsupported early loss curriculum {early_exit_curriculum}.")