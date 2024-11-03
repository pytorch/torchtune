# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from tests.test_utils import gpu_test
from torch import nn
from torchtune.training import OffloadActivations


@gpu_test(gpu_count=1)
@pytest.mark.parametrize("use_streams", [True, False])
def test_offloading_is_same_as_without(use_streams) -> None:
    with torch.device("cuda"):
        torch.manual_seed(2024)
        model = nn.Sequential(
            nn.Linear(10, 10),
            nn.Linear(10, 10),
            nn.Linear(10, 10),
            nn.ReLU(),
        )
        torch.manual_seed(2024)
        model_c = nn.Sequential(
            nn.Linear(10, 10),
            nn.Linear(10, 10),
            nn.Linear(10, 10),
            nn.ReLU(),
        )

    inp = torch.randn((2, 10), device="cuda")
    loss = model(inp).sum()
    loss.backward()

    with OffloadActivations(use_streams=use_streams):
        loss_c = model_c(inp).sum()
    loss_c.backward()

    for param, param_c in zip(model.parameters(), model_c.parameters()):
        assert torch.equal(param.grad, param_c.grad)


@gpu_test(gpu_count=1)
def test_offloading_works_with_view_outputs() -> None:
    """
    This test is quite contrived but tests against a very obscure situation where
    any of the outputs of a backward node are a view of the unpacked tensor.

    We want to ensure that if an unpacked tensor may be used later that we do not
    free it too early.

    How did we contrive this test? We need the backward to execute as so:
    1. We first need a node that unpacks a tensor and returns a view of the tensor
    2. The next node just needs to pass that view along--this NoOp node is needed
        to bypass our heuristic where we delete the _previous_ node's stash after
        executing the current node.
    3. We need to allow the tensor to die to be contaminated with new info, and
        we need a way to look into the contents of the contaminated tensor. We
        separate these into two nodes (because having them in the same node does
        not properly let the tensor reference die as it is within scope.) The
        "Compute" Node queues up ~1 second of work on CUDA followed by a kernel
        evaluating whether dX is full of 1s. The next Node then inspects the
        earlier activation and asserts the result of dX == 1, which is a sync!

    Note that for the backward to execute in the above order, the fwd was made
    to execute in reverse order.
    """

    class BwdReturnsViewOfActivation(torch.autograd.Function):
        @staticmethod
        def forward(ctx, cloned_activation):
            cloned_activation = cloned_activation.t()
            ctx.save_for_backward(cloned_activation)
            return torch.rand(2, 4, device="cuda")

        @staticmethod
        def backward(ctx, dy):
            unpacked_activation = ctx.saved_tensors[0]
            return unpacked_activation.t()

    class NoOp(torch.autograd.Function):
        @staticmethod
        def forward(ctx, cloned_activation):
            ctx.save_for_backward(cloned_activation)
            return cloned_activation.clone()

        @staticmethod
        def backward(ctx, viewed_activation):
            rando_activation = ctx.saved_tensors[0]
            return viewed_activation

    class ComputeNode(torch.autograd.Function):
        @staticmethod
        def forward(ctx, activation):
            return activation.clone()

        @staticmethod
        def backward(ctx, viewed_activation):
            torch.cuda._sleep(2000000000)  # 2e9 is ~1s worth of GPU cycles
            return viewed_activation == 1

    class InspectEarlierActivation(torch.autograd.Function):
        @staticmethod
        def forward(ctx, activation):
            ctx.save_for_backward(torch.ones_like(activation) * 5)
            return activation

        @staticmethod
        def backward(ctx, viewed_activation_all_1):
            corrupter = ctx.saved_tensors[0]
            assert torch.all(
                viewed_activation_all_1
            )  # is the same as before (1s) and NOT W (5s)!!
            return corrupter

    def fwd(t):
        a = InspectEarlierActivation.apply(t)
        b = ComputeNode.apply(a)
        c = NoOp.apply(b)
        d = BwdReturnsViewOfActivation.apply(c)
        return d.sum()

    tensor_c = torch.ones(256, 1024, device="cuda", requires_grad=True)
    ctx = OffloadActivations(use_streams=True)
    with ctx:
        loss_c = fwd(tensor_c)
    # delete the fwd stash to avoid our peek-in-fwd-stash heuristic in the bwd
    ctx.fwd_stash = {}
    loss_c.backward()
