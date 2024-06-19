# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import torch

def early_exit_loss(model, hidden_states_dict, labels, loss_fn):
    # Pop last layer as we already calculated its loss
    if len(model.layers) - 1 in hidden_states_dict:
        hidden_states_dict.pop(len(model.layers) - 1)

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
    losses_scales = 0.1 * torch.Tensor(hidden_layer_ids).to(losses_early) / len(model.layers)

    return torch.sum(losses_scales * losses_early)
