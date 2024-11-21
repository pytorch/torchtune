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

NUM_GPU_CYCLES_IN_ONE_SEC = 2000000000  # 2e9 is ~1s worth of GPU cycles


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
    any of the outputs of a backward node are a view of the unpacked tensor. (See
    the first line item under Note: [Track views of the unpacked]).

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
            torch.cuda._sleep(NUM_GPU_CYCLES_IN_ONE_SEC)
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


@gpu_test(gpu_count=1)
def test_offloading_works_with_view_ac_cached_buffers() -> None:
    """
    Similar to test_offloading_works_with_view_outputs, but for when AC stashes
    a view of the unpacked tensor. See the second line item under Note: [Track
    views of the unpacked].

    For details on how the following custom autograd function was contrived,
    please see the image attached to the PR description in #1936. The visual
    is more helpful than me trying to write a blob of text here.
    """

    class A(torch.autograd.Function):
        @staticmethod
        def forward(ctx, ones):
            ctx.save_for_backward(ones * 5)  # corruptedly saving 5s
            return ones

        @staticmethod
        def backward(ctx, activation_is_ones):
            fives = ctx.saved_tensors[0]
            assert torch.all(activation_is_ones)
            return activation_is_ones

    class B(torch.autograd.Function):
        @staticmethod
        def forward(ctx, ones):
            ctx.save_for_backward(ones.clone())
            return ones.clone()  # important, a view of 1s will be saved in C

        @staticmethod
        def backward(ctx, activation_is_ones):
            saved_tensor = ctx.saved_tensors[0]
            return activation_is_ones.clone()

    class C(torch.autograd.Function):
        @staticmethod
        def forward(ctx, ones):
            ctx.save_for_backward(ones.t().t())
            return ones.clone()

        @staticmethod
        def backward(ctx, grad):
            saved_tensor = ctx.saved_tensors[0]
            return saved_tensor == 1

    class D(torch.autograd.Function):
        @staticmethod
        def forward(ctx, ones):
            ctx.save_for_backward(torch.rand_like(ones))
            return torch.rand_like(ones)

        @staticmethod
        def backward(ctx, grad):
            saved_tensor = ctx.saved_tensors[0]
            torch.cuda._sleep(NUM_GPU_CYCLES_IN_ONE_SEC)
            return torch.rand_like(grad)

    class E(torch.autograd.Function):
        @staticmethod
        def forward(ctx, ones):
            ctx.save_for_backward(torch.rand_like(ones))
            return torch.rand_like(ones)

        @staticmethod
        def backward(ctx, grad):
            # It doesn't matter what E saves, but it needs to save something
            # just to trigger AC recompute to fill in this tensor.
            saved_tensor = ctx.saved_tensors[0]
            return torch.rand_like(grad)

    def checkpointed_region(b):
        c = C.apply(b)
        d = D.apply(c)
        return E.apply(d)

    def fwd(t):
        a = A.apply(t)
        b = B.apply(a)
        e = torch.utils.checkpoint.checkpoint(
            checkpointed_region, b, use_reentrant=False
        )
        return e.sum()

    tensor = torch.ones(256, 1024, device="cuda", requires_grad=True)
    ctx = OffloadActivations(use_streams=True)
    with ctx:
        loss = fwd(tensor)
    # delete the fwd stash to avoid our peek-in-fwd-stash heuristic in the bwd
    ctx.fwd_stash = {}
    loss.backward()
