# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import Optional, Tuple, Union

from torch import nn
from torch.distributed.tensor import (
    DeviceMesh,
    distribute_module,
    distribute_tensor,
    DTensor,
    Partial,
    Replicate,
    Shard,
)
from torch.distributed.tensor.parallel import PrepareModuleInput, PrepareModuleOutput
from torch.distributed.tensor.parallel.style import ParallelStyle
from torch.distributed.tensor.placement_types import Placement


# implementation of Tensor Parallel on the non-shared experts in MoE
class ExpertTensorParallel(ParallelStyle):
    def __init__(
        self,
        *,
        input_layouts: Optional[Tuple[Optional[Placement]]] = None,
        output_layout: Optional[Placement] = None,
        use_local_output: bool = True,
    ):
        super().__init__()
        self.input_layouts = input_layouts or (Replicate(), None)
        self.output_layout = output_layout or Partial()
        self.desired_input_layouts = (Replicate(), None)
        self.use_local_output = use_local_output

    @staticmethod
    def _prepare_input_fn(
        input_layouts, desired_input_layouts, mod, inputs, device_mesh
    ):
        # TODO: figure out dynamo support for instance method and switch this to instance method

        # annotate module input placements/sharding with input_layouts
        input_tensor, input_layout, desired_input_layout = (
            inputs[0],
            input_layouts[0],
            desired_input_layouts[0],
        )
        if not isinstance(input_tensor, DTensor):
            input_tensor = DTensor.from_local(
                input_tensor, device_mesh, (input_layout,), run_check=False
            )

        if input_layouts != desired_input_layouts:
            input_tensor = input_tensor.redistribute(
                placements=(desired_input_layout,), async_op=True
            )
        return (input_tensor, *inputs[1:])

    def _partition_fn(self, name, module, device_mesh):
        module.register_parameter(
            "gate_proj",
            nn.Parameter(distribute_tensor(module.gate_proj, device_mesh, [Shard(2)])),
        )  # Column-wise sharding
        module.register_parameter(
            "down_proj",
            nn.Parameter(distribute_tensor(module.down_proj, device_mesh, [Shard(1)])),
        )  # Row-wise sharding
        module.register_parameter(
            "up_proj",
            nn.Parameter(distribute_tensor(module.up_proj, device_mesh, [Shard(2)])),
        )  # Column-wise sharding

    @staticmethod
    def _prepare_output_fn(output_layout, use_local_output, mod, outputs, device_mesh):
        if outputs.placements != (output_layout,):
            outputs = outputs.redistribute(placements=(output_layout,), async_op=True)
        # back to local tensor
        return outputs.to_local() if use_local_output else outputs

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            self._partition_fn,
            partial(
                self._prepare_input_fn, self.input_layouts, self.desired_input_layouts
            ),
            partial(self._prepare_output_fn, self.output_layout, self.use_local_output),
        )


# NOTE: This is to achieve replicate computation on the gate module in the MoE router.
# It does nothing other than (1) setting the module parameters as DTensors on the given mesh
# and (2) inserting hooks to module boundary to change torch.Tensor to DTensor and back.
# TODO: The reason we need this wrapping is to ensure all parameters are on the same 1D/2D mesh,
# which is assumed by (1) gradient norm clipping, and (2) optimizer fused implementation.
class NoParallel(ParallelStyle):
    def __init__(
        self,
        *,
        input_layout: Optional[Placement] = None,
        output_layout: Optional[Placement] = None,
        use_local_output: bool = True,
    ):
        super().__init__()
        self.input_layout = input_layout or Replicate()
        self.output_layout = output_layout or Replicate()
        self.desired_input_layout = Replicate()
        self.use_local_output = use_local_output

    @staticmethod
    def _prepare_input_fn(input_layout, desired_input_layout, mod, inputs, device_mesh):
        # annotate module input placements/sharding with input_layouts
        input_tensor = inputs[0]
        if not isinstance(input_tensor, DTensor):
            input_tensor = DTensor.from_local(
                input_tensor, device_mesh, (input_layout,), run_check=False
            )

        if input_layout != desired_input_layout:
            input_tensor = input_tensor.redistribute(
                placements=(desired_input_layout,), async_op=True
            )
        return (input_tensor, *inputs[1:])

    @staticmethod
    def _prepare_output_fn(output_layout, use_local_output, mod, outputs, device_mesh):
        if outputs.placements != (output_layout,):
            outputs = outputs.redistribute(placements=(output_layout,), async_op=True)
        # back to local tensor
        return outputs.to_local() if use_local_output else outputs

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            None,
            partial(
                self._prepare_input_fn, self.input_layout, self.desired_input_layout
            ),
            partial(self._prepare_output_fn, self.output_layout, self.use_local_output),
        )


# TODO: this is temporarily copied over from PyTorch core to enable Llama4 on stable PyTorch
# Once this API is in stable, we should migrate over to the PyTorch core one
class PrepareModuleInputOutput(ParallelStyle):
    """
    Configure the nn.Module's inputs (and outputs) to convert the input tensors (and output tensors, respectively) of the nn.Module
    to DTensors at runtime according to ``input_layouts`` (and output_layouts, respectively), and perform layout redistribution
    according to the ``desired_input_layouts`` (and ``desired_output_layouts``, respectively). This is a combination of
    :class:`PrepareModuleInput` and :class:`PrepareModuleOutput`.

    Keyword Args:
        input_layouts (Union[Placement, Tuple[Optional[Placement]]]):
            The DTensor layouts of input tensors for the nn.Module, this is used to convert the input tensors to
            DTensors. If some inputs are not torch.Tensor or no need to convert to DTensors, ``None`` need to be specified
            as a placeholder. default: None.
        desired_input_layouts (Union[Placement, Tuple[Optional[Placement]]]):
            The desired DTensor layout of input tensors for the nn.Module, this is used to ensure the inputs of the nn.Module
            have the desired DTensor layouts. This argument needs to have the same length with ``input_layouts``. default: None.
        input_kwarg_layouts (Dict[str, Placement]):
            The DTensor layouts of input kwargs for the nn.Module, this is used to convert the input kwarg tensors to DTensors.
            default: None
        desired_input_kwarg_layouts: (Dict[str, Placement]):
            The desired DTensor layout of input kwargs for the nn.Module, this is used to ensure the inputs of the nn.Module
            have the desired DTensor layouts. default: None.
        use_local_input (bool, optional):
            Whether to use local :class:`torch.Tensor` instead of :class:`DTensor` for the module inputs, default: False.
        output_layouts (Union[Placement, Tuple[Placement]]):
            The DTensor layouts of output tensors for the nn.Module, this is used to convert the output tensors to
            DTensors if they are :class:`torch.Tensor`. If some outputs are not torch.Tensor or no need to convert to DTensors,
            ``None`` need to be specified as a placeholder.
        desired_output_layouts (Union[Placement, Tuple[Placement]]):
            The desired DTensor layouts of output tensors for the nn.Module, this is used to ensure the outputs of the nn.Module
            have the desired DTensor layouts.
        use_local_output (bool, optional):
            Whether to use local :class:`torch.Tensor` instead of :class:`DTensor` for the module outputs, default: True.


    Example::
        >>> # xdoctest: +SKIP(failing)
        >>> from torch.distributed.tensor.parallel import parallelize_module, PrepareModuleInputOutput
        >>> from torch.distributed.device_mesh import init_device_mesh
        >>> ...
        >>> block = TransformerBlock(...)  # block is a nn.Module that contains an "attn" Attention submodule
        >>> tp_mesh = init_device_mesh("cuda", (8,))
        >>>
        >>> # According to the style specified below, the first input of attn will be annotated as Sharded DTensor
        >>> # and then redistributed to Replicated DTensor, and the output of the TransformerBlock will be annotated
        >>> # as Replicated DTensor and then redistributed to Sharded DTensor.
        >>> parallelize_module(
        >>>     block, # this can be a submodule or module
        >>>     tp_mesh,
        >>>     parallelize_plan={
        >>>         "attn": PrepareModuleInputOutput(
        >>>             input_layouts=(Shard(0), None, None, ...),
        >>>             desired_input_layouts=(Replicate(), None, None, ...),
        >>>             output_layouts=Replicate(),
        >>>             desired_output_layouts=Shard(0),
        >>>         ),
        >>>     }
        >>> )
    """

    def __init__(
        self,
        *,
        input_layouts: Optional[Union[Placement, tuple[Optional[Placement]]]] = None,
        desired_input_layouts: Optional[
            Union[Placement, tuple[Optional[Placement]]]
        ] = None,
        input_kwarg_layouts: Optional[dict[str, Placement]] = None,
        desired_input_kwarg_layouts: Optional[dict[str, Placement]] = None,
        use_local_input: bool = False,
        output_layouts: Union[Placement, tuple[Placement]],
        desired_output_layouts: Union[Placement, tuple[Placement]],
        use_local_output: bool = True,
    ):
        self.prepare_module_input = PrepareModuleInput(
            input_layouts=input_layouts,
            desired_input_layouts=desired_input_layouts,
            input_kwarg_layouts=input_kwarg_layouts,
            desired_input_kwarg_layouts=desired_input_kwarg_layouts,
            use_local_output=use_local_input,
        )
        self.prepare_module_output = PrepareModuleOutput(
            output_layouts=output_layouts,
            desired_output_layouts=desired_output_layouts,
            use_local_output=use_local_output,
        )

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        self.prepare_module_input._apply(module, device_mesh)
        self.prepare_module_output._apply(module, device_mesh)

        return module
