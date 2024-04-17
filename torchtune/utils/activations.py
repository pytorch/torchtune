# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import gc

from collections import defaultdict

from typing import Any, Dict

import torch

from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
    CheckpointImpl,
)

from torch.utils.checkpoint import _pt2_selective_checkpoint_context_fn_gen, checkpoint

# for selective AC
no_recompute_list = {
    torch.ops.aten.mm.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops._c10d_functional.reduce_scatter_tensor.default,
}

# Uses PTD FSDP AC wrapper
# currently selective per op and per layer checkpointing are supported
def checkpoint_wrapper(module, ac_mode, ac_style):
    if ac_mode == "selective" and ac_style == "op":

        def _get_custom_policy(meta):
            def _custom_policy(mode, func, *args, **kwargs):
                mm_count_key = f"{mode}_mm_count"
                if func == torch.ops.aten.mm.default:
                    meta[mm_count_key] += 1
                # Saves output of all compute ops, except every second mm

                return func in no_recompute_list and not (
                    func == torch.ops.aten.mm.default and meta[mm_count_key] % 2 == 0
                )

            return _custom_policy

        def selective_checkpointing_context_fn():
            meta = defaultdict(int)
            return _pt2_selective_checkpoint_context_fn_gen(_get_custom_policy(meta))

        return ptd_checkpoint_wrapper(
            module,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            checkpoint_fn=checkpoint,
            context_fn=selective_checkpointing_context_fn,
            use_reentrant=False,
            preserve_rng_state=False,
        )
    elif ac_mode == "full":
        return ptd_checkpoint_wrapper(
            module,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            checkpoint_fn=checkpoint,
            use_reentrant=False,
            preserve_rng_state=False,
        )

    # selective layer checkpointing...some checks in case we receive '2' or 2...
    elif ac_mode == "selective" and (
        isinstance(ac_style, (int, float))
        or (isinstance(ac_style, str) and ac_style.isdigit())
    ):
        """enables selective checkpointing of candidate layers.
        Usage:
        'selective_ac_option' with a positive 'int' value in config controls which layers to checkpoint.
        1 == checkpointing every one (all).
        2 == checkpoint every 2nd one
        """
        every_x_layer = int(ac_style)
        assert (
            every_x_layer >= 0
        ), f"selective layer AC policy (every_x_layer) expects a positive integer, received {every_x_layer}"

        checkpoint_wrapper.__dict__.setdefault("_count", 0)

        checkpoint_wrapper._count += 1
        if not every_x_layer or checkpoint_wrapper._count % every_x_layer == 0:
            return ptd_checkpoint_wrapper(
                module,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                checkpoint_fn=checkpoint,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        # skip activation checkpointing and store activations for this layer
        else:
            return module

    else:
        raise NotImplementedError(
            "Unknown AC type or AC config. Only selective op and selective layer ac implemented currently."
        )

def set_selective_activation_checkpointing(
    model: nn.Module,
    ac_mode: str,
    ac_option: Optional[Union[int, str]],
) -> None:
    """Utility to setup activation checkpointing and wrap the model for checkpointing.

    Args:
        model (nn.Module): Model to setup activation checkpointing.
        ac_mode (str): Activation checkpointing mode. ['none', 'full', 'selective']
        ac_option (Optional[Union[int, str]]): Activation checkpointing option.
            - If ac_mode is 'selective', ac_option can be an integer or a string
              representing the number of layers to checkpoint.
            - If ac_mode is 'selective' and ac_option is 'op', then selective op ac is run.
            - If ac_mode is 'none' or 'full, ac_option is ignored.
    """

    for layer_id, transformer_block in enumerate(model.layers):
        # print(f"inside set act checkpoint: {type(transformer_block)=}) # , {transformer_block=}" )
        if ac_mode in ("full", "selective"):

            transformer_block = checkpoint_wrapper(
                transformer_block,
                ac_mode,
                ac_option,
            )
        model.layers[layer_id] = transformer_block
