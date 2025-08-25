# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared validation utilities for recipes."""

from typing import Optional, Set
from torchtune.modules.loss import LinearCrossEntropyLoss


def validate_custom_sharding_config(
    loss_fn,
    custom_sharded_layers: Optional[list[str]],
    required_layer: str = "output",
    parallelism_enabled: Optional[bool] = None,
    available_layers: Optional[Set[str]] = None,
) -> None:
    """
    Validates custom_sharded_layers configuration for specific loss functions.
    
    Args:
        loss_fn: The loss function instance
        custom_sharded_layers: List of layer names to shard, or None
        required_layer: The layer name that must be included (default: "output")
        parallelism_enabled: If False, skip validation (default: None)
        available_layers: Optional set of valid layer names for typo checking
        
    Raises:
        ValueError: If validation fails
    """
    # Skip when nothing to validate
    if not custom_sharded_layers:
        return
        
    # Skip validation if parallelism is explicitly disabled
    if parallelism_enabled is False:
        return
    
    # Only enforce when the loss needs the output projection
    needs_output_proj = isinstance(loss_fn, LinearCrossEntropyLoss)
    
    if needs_output_proj and required_layer not in custom_sharded_layers:
        raise ValueError(
            f"When using {type(loss_fn).__name__} with custom_sharded_layers, "
            f"'{required_layer}' must be included to ensure tensor compatibility. "
            f"Example: custom_sharded_layers = ['tok_embeddings', '{required_layer}']."
        )
    
    # Optional: catch typos early
    if available_layers is not None:
        unknown = set(custom_sharded_layers) - set(available_layers)
        if unknown:
            raise ValueError(
                f"Unknown layer(s) in custom_sharded_layers: {sorted(unknown)}"
            )
