# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtune.config._utils import _get_component_from_path

try:
    from torchao.prototype.low_bit_optim import (
        CPUOffloadOptimizer as _CPUOffloadOptimizer,
    )

    class CPUOffloadOptimizer(_CPUOffloadOptimizer):
        """Offload optimizer to CPU for single-GPU training. This will reduce GPU memory by the size of optimizer
        state. Optimizer step will be done on CPU.

        Args
            params: a list of parameters or parameter groups.
            optimizer_class: constructor of the base optimizer.
            offload_gradients: free GPU gradients once they are moved to CPU. Not compatible with gradient
                accumulation.
            kwargs: other keyword arguments to be passed to the base optimizer e.g. `lr`, `weight_decay`.

        NOTE: This is a light wrapper around `torchao.prototype.low_bit_optim.CPUOffloadOptimizer` that parses
        optimizer class string into a concrete optimizer class.
        """

        def __init__(
            self,
            params,
            optimizer_class: str,
            *,
            offload_gradients: bool = False,
            **kwargs,
        ) -> None:
            super().__init__(
                self,
                params,
                _get_component_from_path(optimizer_class),
                offload_gradients=offload_gradients,
                **kwargs,
            )

except ImportError:

    class CPUOffloadOptimizer:
        def __init__(
            self,
            params,
            optimizer_class: str,
            *,
            offload_gradients: bool = False,
            **kwargs,
        ) -> None:
            raise NotImplementedError(
                "CPU offload optimizer requires torchao>=0.5 or torchao-nightly. Try update torchao or install "
                "torchao from source: pip install git+https://github.com/pytorch/ao.git"
            )
