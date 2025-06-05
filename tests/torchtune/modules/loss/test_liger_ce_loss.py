# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from tests.test_utils import assert_expected, fixed_init_model, gpu_test
from torch import nn
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.tensor import Replicate
from torch.distributed.tensor.parallel import ColwiseParallel, parallelize_module
from torch.optim import SGD
from torchtune.modules.loss import LigerLinearCrossEntropy
from torchtune.training import ParallelDims
from torchtune.training.seed import set_seed


BATCH_SIZE = 2
SEQ_LEN = 8
EMBED_DIM = 16
VOCAB_SIZE = 100
IGNORE_INDEX = -100
WORLD_SIZE = 4  # minimum to test FSDP + TP


class Model(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.layer = nn.Linear(embed_dim, embed_dim)
        self.output = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = F.relu(self.layer(x))
        return self.output(x)


@pytest.fixture(autouse=True)
def random():
    set_seed(42)


def get_open_port():
    """
    Finds and returns an open port on the system.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))  # Bind to all interfaces on a random available port
    sock.listen(1)  # Start listening for connections (doesn't matter for this purpose)
    port = sock.getsockname()[1]
    sock.close()
    return port


class TestLigerFusedCrossEntropyLoss:
    def loss_step(self, model, loss_fn, embedding, targets):
        opt = SGD(model.parameters(), lr=0.2)
        opt.zero_grad()

        # Forward pass
        hidden = model.layer(embedding)
        loss = loss_fn(hidden, targets)

        # Backward pass and optimizer step
        loss.backward()
        opt.step()
        return loss

    def compute_loss(self, fused_model, model, device, compile, rank=0):
        # Create dummy data
        gen = torch.Generator(device)
        gen.manual_seed(rank)
        embed = torch.randn(
            BATCH_SIZE,
            SEQ_LEN,
            EMBED_DIM,
            dtype=torch.float32,
            generator=gen,
            device=device,
        )

        targets = torch.randint(
            0,
            VOCAB_SIZE,
            (BATCH_SIZE, SEQ_LEN),
            dtype=torch.long,
            generator=gen,
            device=device,
        )

        # Add ignored indices
        mask = torch.rand((BATCH_SIZE, SEQ_LEN), generator=gen, device=device) < 0.2
        targets[mask] = IGNORE_INDEX

        # Test liger loss
        loss_fn = LigerLinearCrossEntropy(ignore_index=IGNORE_INDEX)
        loss_fn.set_model_output(fused_model)
        if compile:
            loss_fn.apply_compile_strategy()
        fused_loss = self.loss_step(fused_model, loss_fn, embed, targets)

        # Test standard loss
        def standard_loss_fn(hidden, targets):
            logits = model.output(hidden)
            logits = logits.reshape(-1, VOCAB_SIZE)
            targets = targets.reshape(-1)
            loss = F.cross_entropy(
                logits, targets, reduction="mean", ignore_index=IGNORE_INDEX
            )
            return loss

        if compile:
            torch.compile(standard_loss_fn)
        standard_loss = self.loss_step(model, standard_loss_fn, embed, targets)

        return fused_loss, standard_loss

    @gpu_test(gpu_count=1)
    @pytest.mark.parametrize("compile", [False, True])
    def test_liger_fused_cross_entropy(self, compile):
        """Test gradient flow through full forward/backward pass with optimizer step"""
        # Create model for fused loss
        fused_model = Model(VOCAB_SIZE, EMBED_DIM).cuda()
        fixed_init_model(fused_model, min_val=-0.1, max_val=0.1)

        # Create model for standard loss
        model = Model(VOCAB_SIZE, EMBED_DIM).cuda()
        fixed_init_model(model, min_val=-0.1, max_val=0.1)

        fused_loss, standard_loss = self.compute_loss(
            fused_model, model, "cuda", compile
        )

        # Verify:
        # 1. Validate the results are close enough
        assert_expected(fused_loss, standard_loss, rtol=1e-2, atol=1e-2)

        # 2. Validate the weight updates are close enough
        for p1, p2 in zip(fused_model.parameters(), model.parameters()):
            assert_expected(p1, p2, rtol=1e-2, atol=1e-2)

    def loss_worker(self, rank, compile, port):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(port)
        device = torch.device(f"cuda:{rank}")
        dist.init_process_group(
            backend="nccl", rank=rank, world_size=WORLD_SIZE, device_id=device
        )
        torch.cuda.set_device(device)
        mesh = ParallelDims(
            dp_replicate=1, dp_shard=WORLD_SIZE // 2, tp=2, world_size=WORLD_SIZE
        ).build_mesh("cuda")
        plan = {
            "output": ColwiseParallel(
                input_layouts=Replicate(), output_layouts=Replicate()
            )
        }

        # Create model for fused loss
        fused_model = Model(VOCAB_SIZE, EMBED_DIM).cuda()
        fixed_init_model(fused_model, min_val=-0.1, max_val=0.1)
        parallelize_module(fused_model, mesh["tp"], parallelize_plan=plan)
        for m in [fused_model.layer, fused_model.output, fused_model]:
            fully_shard(m, mesh=mesh["dp_shard"])

        # Create model for standard loss
        model = Model(VOCAB_SIZE, EMBED_DIM).cuda()
        fixed_init_model(model, min_val=-0.1, max_val=0.1)
        parallelize_module(model, mesh["tp"], parallelize_plan=plan)
        for m in [model.layer, model.output, model]:
            fully_shard(m, mesh=mesh["dp_shard"])

        fused_loss, standard_loss = self.compute_loss(
            fused_model, model, device, compile, mesh["dp_shard"].get_local_rank()
        )

        # Verify:
        # 1. Validate the results are close enough
        assert_expected(fused_loss, standard_loss, rtol=1e-2, atol=1e-2)

        # 2. Validate the weight updates are close enough
        for p1, p2 in zip(fused_model.parameters(), model.parameters()):
            assert_expected(p1, p2, rtol=1e-2, atol=1e-2)

        dist.destroy_process_group()

    @gpu_test(gpu_count=WORLD_SIZE)
    @pytest.mark.parametrize("compile", [False, True])
    def test_liger_fused_cross_entropy_distributed(self, compile):
        """Test gradient flow through full forward/backward pass with optimizer step"""
        port = get_open_port()
        mp.spawn(
            self.loss_worker,
            args=(compile, port),
            nprocs=WORLD_SIZE,
            join=True,
        )
