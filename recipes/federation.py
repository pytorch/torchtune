# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict

import aiofiles

import torch

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from omegaconf import DictConfig

from torchtune import config, training, utils

logger = utils.get_logger("DEBUG")

app = FastAPI()


class FederationRecipe:
    """
    Recipe for merging federated Transformer-based LLM.

    Supported merge modes are:
    Average:
        torchtune.training.merge.AverageMerger
    """

    def is_participant(self, participant_id: str) -> bool:
        return participant_id in self._participants

    def add_checkpoint(self, participant_id: str, cfg: DictConfig):
        self._awaiting_participants.remove(participant_id)
        logger.info(f"Received checkpoint for step {self._current_step} {self._dtype}.")
        if len(self._awaiting_participants) == 0:
            loop = asyncio.get_running_loop()
            loop.run_in_executor(None, self.federate, cfg)

    def __init__(self, cfg: DictConfig) -> None:
        self._device = utils.get_device(device=cfg.device)
        self._dtype = training.get_dtype(dtype=cfg.dtype, device=self._device)
        self._participants = cfg.federation.participants
        self._awaiting_participants = cfg.federation.participants.copy()
        self._current_step = 0
        # self._merger = config.instantiate(cfg.merger)
        training.set_seed(
            seed=cfg.seed, debug_mode=cfg.get("cudnn_deterministic_mode", None)
        )

    def setup(self, cfg: DictConfig) -> None:
        self._checkpointer = config.instantiate(cfg.checkpointer)
        checkpoint_dict = self._checkpointer.load_checkpoint()
        self._model = checkpoint_dict[training.MODEL_KEY]

    def load_checkpoint(
        self, outdir: str, file: str, cfg: DictConfig
    ) -> Dict[str, Any]:
        checkpointer_cfg = cfg.checkpointer.copy()
        checkpointer_cfg.checkpoint_files = [file]
        checkpointer_cfg.weight_dir = outdir
        checkpointer = config.instantiate(checkpointer_cfg)
        checkpoint_dict = checkpointer.load_checkpoint()
        del checkpointer
        return checkpoint_dict

    def get_file_name(self, id: int, token: str | None, cfg: DictConfig):
        split = cfg.checkpointer.checkpoint_files[0].split(".")
        file_name = split[0]
        suffix = "." + split[-1]

        output_dir = Path(cfg.checkpointer.output_dir) / f"step-{id}"
        if token is not None:
            output_dir = output_dir / token
        output_dir.mkdir(exist_ok=True, parents=True)

        checkpoint_file = Path.joinpath(output_dir, f"{file_name}").with_suffix(suffix)

        return checkpoint_file

    def save_checkpoint(self, cfg: DictConfig):
        ckpt_dict = self._model
        split = cfg.checkpointer.checkpoint_files[0].split(".")
        file_name = split[0]
        suffix = "." + split[-1]

        output_dir = Path(cfg.checkpointer.output_dir) / f"step-{self._current_step}"
        output_dir.mkdir(exist_ok=True)
        checkpoint_file = Path.joinpath(output_dir, f"{file_name}").with_suffix(suffix)

        torch.save(ckpt_dict, checkpoint_file)
        logger.info(
            "Model checkpoint of size "
            f"{os.path.getsize(checkpoint_file) / 1024**3:.2f} GiB "
            f"saved to {checkpoint_file}"
        )

    @torch.no_grad()
    def federate(self, cfg: DictConfig):
        logger.info(f"Starting federated merge for step {self._current_step}.")

        # Prepare empty model for averaging
        for param_tensor in self._model:
            self._model[param_tensor] = torch.zeros_like(self._model[param_tensor]).to(
                self._device
            )

        for participant in self._participants:
            checkpoint_file = self.get_file_name(self._current_step, participant, cfg)
            if not os.path.exists(checkpoint_file):
                logger.error(f"Checkpoint file {checkpoint_file} not found.")
                return

            checkpoint = self.load_checkpoint(
                checkpoint_file.parent, checkpoint_file.name, cfg
            )[training.MODEL_KEY].to(self._device)

            for param_tensor in self._model:
                self._model[param_tensor] += checkpoint[param_tensor] / len(
                    self._participants
                )

        logger.info(f"Finished federated merge for step {self._current_step}.")
        self.save_checkpoint(cfg)
        self._current_step += 1

        self._awaiting_participants = cfg.federation.participants.copy()


async def authenticate(request: Request):
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    token = auth_header.split(" ", 1)[1]

    recipe = request.app.state.recipe
    if recipe is None:
        raise HTTPException(status_code=400, detail="Recipe not initialized")

    if not recipe.is_participant(token):
        raise HTTPException(status_code=401, detail="Unauthorized")

    cfg = request.app.state.cfg
    if cfg is None:
        raise HTTPException(status_code=500, detail="Configuration not available")

    return token, recipe, cfg


@app.put("/checkpoint")
async def put_checkpoint(request: Request, file: UploadFile = File(...)):
    token, recipe, cfg = await authenticate(request)
    checkpoint_file = recipe.get_file_name(recipe._current_step, token, cfg)

    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    f = await aiofiles.open(checkpoint_file, "wb")
    while chunk := await file.read(10 * 1024 * 1024):
        await f.write(chunk)

    logger.info(f"Received checkpoint data saved to {checkpoint_file}")

    try:
        recipe.add_checkpoint(token, cfg)
    except Exception as e:
        logger.error(f"Error when adding checkpoint: {e}")
        raise HTTPException(
            status_code=500, detail="Error when adding checkpoint"
        ) from None

    return {"detail": "Checkpoint saved", "id": recipe._current_step}


@app.get("/status")
async def get_status(request: Request):
    recipe = request.app.state.recipe
    if recipe is None:
        raise HTTPException(status_code=400, detail="Recipe not initialized")

    return {
        "step": recipe._current_step,
        "state": "ready" if len(recipe._awaiting_participants) > 0 else "busy",
    }


@app.get("/checkpoint/{checkpoint_id}")
async def get_checkpoint(request: Request, checkpoint_id: int):
    token, recipe, cfg = await authenticate(request)
    checkpoint_file = recipe.get_file_name(checkpoint_id, token, cfg)

    async def file_iterator(file_path: str, chunk_size: int = 10 * 1024 * 1024):
        async with aiofiles.open(file_path, "rb") as f:
            while True:
                chunk = await f.read(chunk_size)
                if not chunk:
                    break
                yield chunk

    return StreamingResponse(
        file_iterator(checkpoint_file), media_type="application/octet-stream"
    )


@config.parse
def main(cfg: DictConfig) -> None:
    config.log_config(recipe_name="FederationRecipe", cfg=cfg)
    recipe = FederationRecipe(cfg=cfg)
    recipe.setup(cfg=cfg)

    app.state.recipe = recipe
    app.state.cfg = cfg

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    sys.exit(main())
