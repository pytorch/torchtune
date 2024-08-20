# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch.optim.optimizer import ParamsT
from torchtune.config._utils import _get_component_from_path

try:
    from torchao.prototype.low_bit_optim import (
        CPUOffloadOptimizer as _CPUOffloadOptimizer,
    )

    _CPU_OFFLOAD_OPTIM_AVAILABLE = True

    class CPUOffloadOptimizer(_CPUOffloadOptimizer):
        """Offload optimizer to CPU for single-GPU training. This will reduce GPU memory by the size of optimizer
        state. Optimizer step will be done on CPU.

        Args:
            params (ParamsT): a list of parameters or parameter groups.
            optimizer_class (str): name of the base optimizer.
            offload_gradients (bool): free GPU gradients once they are moved to CPU. Setting this to True will further
                reduce GPU memory by the size of gradients. This is not compatible with gradient accumulation.
            **kwargs: other keyword arguments to be passed to the base optimizer e.g. `lr`, `weight_decay`.

        Example:
            >>> from torchtune.modules.optim import CPUOffloadOptimizer
            >>> optimizer = CPUOffloadOptimizer(
            ...    params=model.parameters(),
            ...    optimizer_class='torch.optim.AdamW',
            ...    offload_gradients=False,
            ...    lr=2e-5
            ... )

        To use this optimizer from the command line, you can write
            optimizer._component_=torchtune.modules.optim.CPUOffloadOptimizer
            optimizer.optimizer_class=torch.optim.AdamW

        NOTE: This is a light wrapper around `torchao.prototype.low_bit_optim.CPUOffloadOptimizer` that parses
        optimizer class string into a concrete optimizer class. See the documentation here for caveats
        https://github.com/pytorch/ao/tree/main/torchao/prototype/low_bit_optim#optimizer-cpu-offload
        """

        def __init__(
            self,
            params: ParamsT,
            optimizer_class: str,
            *,
            offload_gradients: bool = False,
            **kwargs,
        ) -> None:
            super().__init__(
                params=params,
                optimizer_class=_get_component_from_path(optimizer_class),
                offload_gradients=offload_gradients,
                **kwargs,
            )

except ImportError:

    _CPU_OFFLOAD_OPTIM_AVAILABLE = False

    class CPUOffloadOptimizer:
        def __init__(
            self,
            params: ParamsT,
            optimizer_class: str,
            *,
            offload_gradients: bool = False,
            **kwargs,
        ) -> None:
            raise NotImplementedError(
                "CPU offload optimizer requires torchao>=0.5 or torchao-nightly. Try update torchao or install "
                "torchao from source: pip install git+https://github.com/pytorch/ao.git"
            )
