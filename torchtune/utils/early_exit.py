# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import numpy as np
import torch
from enum import Enum
from typing import List
import math

class LossScaleType(str, Enum):
    ONE = "one"
    L = "l"
    SUM_L = "sum_l"
    INV_L = "inv_l"
    SQRT_L = "sqrt_l"
    INV_SQRT_L = "inv_sqrt_l"

def early_exit_loss(model, hidden_states_dict, labels, loss_fn, e_scale: float=0.1, loss_scale_type=LossScaleType.SUM_L):
    batch_loss_fn = copy.deepcopy(loss_fn)
    batch_loss_fn.reduction = "none"

    e = len(hidden_states_dict)
    # List of e tensors with shape [b, s, d]
    hidden_states = tuple(hidden_states_dict.values())
    hidden_layer_ids = tuple(hidden_states_dict.keys())
    # Shape: [e, b, s, d]
    hidden_states_stacked = torch.stack(hidden_states)
    # Shape: [e, b, s, out_dim]
    logits_early = model.output(model.norm(hidden_states_stacked))
    logits_early = logits_early[..., :-1, :].contiguous()
    # Shape: [e*b, s, out_dim]
    logits_early = logits_early.flatten(0, 1)
    logits_early = logits_early.transpose(1, 2)
    # Shape: [e, b*s]
    labels_repeated = labels.repeat(e, 1)
    # Compute early losses: Shape: [e*b, s]
    losses_early = batch_loss_fn(logits_early, labels_repeated)
    # Shape: [e, b*s]
    losses_early = losses_early.view(e, -1)
    # Shape: [e]
    s_unpadded = (labels != loss_fn.ignore_index).sum()
    losses_early = losses_early.float().sum(-1) / s_unpadded
    # Shape: [e]
    # losses_scales = 0.1 * torch.Tensor(hidden_layer_ids).to(losses_early) / len(model.layers)
    losses_scales = layer_ids_to_loss_scales(torch.Tensor(hidden_layer_ids).to(losses_early), len(model.layers), loss_scale_type, e_scale)

    return torch.sum(losses_scales * losses_early)

def layer_ids_to_loss_scales(layer_ids, n_layers, loss_scale_type: LossScaleType, e_scale: float):
    match loss_scale_type:
        case LossScaleType.ONE:
            loss_scales = torch.ones(len(layer_ids))
        case LossScaleType.L:
            loss_scales = torch.Tensor(layer_ids+1)
        case LossScaleType.SUM_L:
            # TODO: should we change to sum 0:i ? Perhaps create a new scale_type
            loss_scales = torch.cumsum(layer_ids+1, dim=0)
        case LossScaleType.SQRT_L:
            loss_scales = torch.sqrt(layer_ids+1)
        case LossScaleType.INV_L:
            loss_scales = 1.0 / (layer_ids+1)
        case LossScaleType.INV_SQRT_L:
            loss_scales = 1.0 / torch.sqrt(layer_ids+1)
        case _:
            raise ValueError(f"Unsupported loss_scale type {loss_scale_type}")

    loss_scales = loss_scales * torch.where(loss_scales < n_layers - 1, e_scale, 1.0)

    # normalize loss scales to ensure that their sum is 1.0
    loss_scales = loss_scales / torch.sum(loss_scales)
    assert torch.isclose(torch.sum(loss_scales), torch.Tensor([1.0]).to(loss_scales))

    return loss_scales
