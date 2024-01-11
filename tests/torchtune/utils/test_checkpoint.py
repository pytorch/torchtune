# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import os
import tempfile

import numpy as np
import pytest
import torch

from tests.test_utils import get_pet_launch_config, skip_if_cuda_not_available
from torch.distributed import launcher
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torchtune.utils.checkpoint import load_checkpoint, save_checkpoint
from torchtune.utils.device import _get_device_from_env
from torchtune.utils.env import (
    _get_process_group_backend_from_device,
    init_from_env,
    seed,
)


class TestCheckpoint:
    def _save_and_load(self, checkpoint):
        with tempfile.NamedTemporaryFile() as f:
            save_checkpoint(ckpt_dict=checkpoint, output_loc=f.name)
            loaded_ckpt = torch.load(f.name)
        return loaded_ckpt

    def _get_model_and_optim(self, zero_model, fsdp):
        model = torch.nn.Linear(10, 10)
        if fsdp:
            model = FSDP(model)
        optim = torch.optim.SGD(model.parameters(), lr=0.01)
        return model, optim

    def _validate_dicts(self, d1, d2):
        assert len(d1) == len(d2)
        for k, v in d1.items():
            assert k in d2
            if isinstance(v, dict):
                self._validate_dicts(v, d2[k])
            else:
                if isinstance(v, torch.Tensor):
                    assert torch.allclose(v, d2[k])
                else:
                    assert v == d2[k]

    def test_local_checkpoint_save_load(self) -> None:
        model, optim = self._get_model_and_optim(zero_model=False, fsdp=False)
        # Create dummy optim states to verify they can be loaded.
        for p in model.parameters():
            p.grad = torch.rand_like(p)
        optim.step()
        checkpoint = {"model": model, "optimizer": optim, "lr": 0.01}
        loaded_ckpt = self._save_and_load(checkpoint)
        assert "lr" in loaded_ckpt and loaded_ckpt["lr"] == 0.01
        model_new, optim_new = self._get_model_and_optim(
            zero_model=True, fsdp=False
        )
        load_checkpoint(ckpt_dict=loaded_ckpt, model=model_new, optimizer=optim_new)
        # model_new and model params should match
        for p1, p2 in zip(model.parameters(), model_new.parameters()):
            assert torch.allclose(p1, p2)
        # optim state_dicts should match
        self._validate_dicts(optim.state_dict(), optim_new.state_dict())

    def _test_distributed_save_load(self) -> None:
        torch.distributed.init_process_group(backend="gloo")
        model, optim = self._get_model_and_optim(zero_model=False, fsdp=True)
        for p in model.parameters():
            p.grad = torch.rand_like(p)
        optim.step()
        checkpoint = {"model": model, "optimizer": optim, "lr": 0.01}
        loaded_ckpt = self._save_and_load(checkpoint)

        # Verify FSDP optim_state_dict matches as expected.
        fsdp_optim_state_dict = FSDP.optim_state_dict(model, optim)
        saved_fsdp_optim_state_dict = loaded_ckpt["optim"]
        self._validate_dicts(fsdp_optim_state_dict, saved_fsdp_optim_state_dict)

        assert "lr" in loaded_ckpt and loaded_ckpt["lr"] == 0.01

        # Verify model and optim state_dicts match after load.
        model_new, optim_new = self._get_model_and_optim(zero_model=True, fsdp=True)
        load_checkpoint(ckpt_dict=loaded_ckpt, model=model_new, optimizer=optim_new)
        # Verify  model
        with FSDP.summon_full_params(model_new):
            with FSDP.summon_full_params(model):
                for p1, p2 in zip(model.parameters(), model_new.parameters()):
                    assert torch.allclose(p1, p2)
        # Verify optim state_dicts
        self._validate_dicts(
            FSDP.optim_state_dict(model_new, optim_new), fsdp_optim_state_dict
        )

    def test_distributed_save_load(self) -> None:
        lc = get_pet_launch_config(nproc=4)
        launcher.elastic_launch(lc, entrypoint=self._test_distributed_save_load)()
