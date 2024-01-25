# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tempfile

import pytest
import torch
import torch.distributed as dist
from torch.distributed import launcher
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DistributedSampler
from torchtune.utils import CheckpointableDataLoader
from torchtune.utils.checkpoint import load_checkpoint, save_checkpoint
from torchtune.utils.distributed import get_world_size_and_rank

from tests.test_utils import (
    _IdentityMapDataset,
    get_pet_launch_config,
    skip_if_cuda_not_available,
)


class TestCheckpoint:
    def _save_and_load(self, checkpoint, model, optimizer, dataloader):
        with tempfile.NamedTemporaryFile() as f:
            save_checkpoint(ckpt_dict=checkpoint, output_loc=f.name)
            if torch.distributed.is_initialized():
                # All ranks wait for 0 to finish saving.
                dist.barrier()
                # Broadcast rank 0's saved filename so other ranks know
                # where to load from.
                file_list = [f.name if dist.get_rank() == 0 else None]
                dist.broadcast_object_list(file_list, src=0)
                file_name = file_list[0]
            else:
                file_name = f.name
            checkpoint = load_checkpoint(file_name, model, optimizer)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            dataloader.load_state_dict(checkpoint["dataloader"])
            # Have rank 0 wait for all ranks to finish loading before exiting
            # context manager which would trigger file destruction.
            if torch.distributed.is_initialized():
                dist.barrier()

    def _get_model_and_optim(self, zero_model, fsdp):
        model = torch.nn.Linear(10, 10)
        if zero_model:
            with torch.no_grad():
                for p in model.parameters():
                    p.zero_()
        if fsdp:
            model = FSDP(model, device_id=torch.cuda.current_device())
        optim = torch.optim.SGD(model.parameters(), lr=0.01)
        return model, optim

    def _get_dataloader(self):
        # Create a sampler and dataloader that we will checkpoint
        ws, rank = get_world_size_and_rank()
        dataset = _IdentityMapDataset(16 * ws)
        sampler = DistributedSampler(
            dataset, num_replicas=ws, rank=rank, shuffle=True, seed=5
        )
        dataloader = CheckpointableDataLoader(
            dataset,
            batch_size=1,
            shuffle=None,
            sampler=sampler,
        )
        return dataloader

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
        dataloader = self._get_dataloader()
        # Create dummy optim states to verify they can be loaded.
        for p in model.parameters():
            p.grad = torch.rand_like(p)
        optim.step()
        # Iter the dataloader a bit
        dl_iter = iter(dataloader)
        for _ in range(8):
            next(dl_iter)
        checkpoint = {
            "model": model,
            "optimizer": optim,
            "lr": 0.01,
            "dataloader": dataloader,
        }
        model_new, optim_new = self._get_model_and_optim(zero_model=True, fsdp=False)
        dl_new = self._get_dataloader()
        assert dl_new.state_dict() == {
            CheckpointableDataLoader._DISTRIBUTED_SAMPLER_SHUFFLE_SEED: 5,
            CheckpointableDataLoader._SKIP_INDEX_KEY: 0,
        }
        # Saves checkpoint, calls load_checkpoint and loads model + optim states into new model/optim.
        self._save_and_load(checkpoint, model_new, optim_new, dl_new)
        # model_new and model params should match
        for p1, p2 in zip(model.parameters(), model_new.parameters()):
            assert torch.allclose(p1, p2)
        # optim state_dicts should match
        self._validate_dicts(optim.state_dict(), optim_new.state_dict())
        # Dataloader should be forwarded.
        assert dl_new.state_dict() == {
            CheckpointableDataLoader._DISTRIBUTED_SAMPLER_SHUFFLE_SEED: 5,
            CheckpointableDataLoader._SKIP_INDEX_KEY: 8,
        }

    def test_no_model_key_save(self) -> None:
        checkpoint = {"lr": 0.03}
        with pytest.raises(
            RuntimeError,
            match="Expected `ckpt_dict` to contain a `model` key, but it does not.",
        ):
            with tempfile.NamedTemporaryFile() as f:
                save_checkpoint(checkpoint, f.name)

    def test_no_model_key_load(self) -> None:
        model = torch.nn.Linear(1, 1)
        with tempfile.NamedTemporaryFile() as f:
            torch.save({"lr": 0.01}, f.name)
            with pytest.raises(
                RuntimeError,
                match="Expected loaded checkpoint to contain a `model` key.*",
            ):
                load_checkpoint(f.name, model)

    def test_no_optim_key_load(self) -> None:
        model = torch.nn.Linear(1, 1)
        optim = torch.optim.SGD(model.parameters(), lr=0.01)
        with tempfile.NamedTemporaryFile() as f:
            save_checkpoint({"model": model}, f.name)
            with pytest.raises(
                RuntimeError,
                match="Expected loaded checkpoint to contain an `optimizer` key.*",
            ):
                load_checkpoint(f.name, model, optim)

    def _test_distributed_save_load(self) -> None:
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(torch.distributed.get_rank())
        torch.distributed.barrier()
        model, optim = self._get_model_and_optim(zero_model=False, fsdp=True)
        dataloader = self._get_dataloader()
        for p in model.parameters():
            p.grad = torch.rand_like(p)
        optim.step()
        # Iter the dataloader a bit
        dl_iter = iter(dataloader)
        for _ in range(8):
            next(dl_iter)
        checkpoint = {
            "model": model,
            "optimizer": optim,
            "lr": 0.01,
            "dataloader": dataloader,
        }
        model_new, optim_new = self._get_model_and_optim(zero_model=True, fsdp=True)
        dl_new = self._get_dataloader()
        assert dl_new.state_dict() == {
            CheckpointableDataLoader._DISTRIBUTED_SAMPLER_SHUFFLE_SEED: 5,
            CheckpointableDataLoader._SKIP_INDEX_KEY: 0,
        }
        # Saves checkpoint, calls load_checkpoint and loads model + optim states into new model/optim.
        self._save_and_load(checkpoint, model_new, optim_new, dl_new)

        # Verify  model
        with FSDP.summon_full_params(model_new):
            with FSDP.summon_full_params(model):
                for p1, p2 in zip(model.parameters(), model_new.parameters()):
                    assert torch.allclose(p1, p2)
        # Verify optim state_dicts
        self._validate_dicts(
            FSDP.optim_state_dict(model_new, optim_new),
            FSDP.optim_state_dict(model, optim),
        )
        # Dataloader should be forwarded.
        assert dl_new.state_dict() == {
            CheckpointableDataLoader._DISTRIBUTED_SAMPLER_SHUFFLE_SEED: 5,
            CheckpointableDataLoader._SKIP_INDEX_KEY: 8,
        }
        torch.distributed.barrier()

    @skip_if_cuda_not_available
    def test_distributed_save_load(self) -> None:
        lc = get_pet_launch_config(nproc=min(4, torch.cuda.device_count()))
        launcher.elastic_launch(lc, entrypoint=self._test_distributed_save_load)()
