# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import io
import os  # added for temporary directory handling
import time
from typing import Any, Dict, Protocol

import torch
from requests import get, put

from torchtune import training, utils
from torchtune.utils import get_logger

logger = get_logger("DEBUG")


class _ParticipantInterface(Protocol):
    """
    Interface implemented by Participants in torchtune.
    """

    def handshake(self) -> Dict[str, Any]:
        ...

    def synchronize(self, model: torch.nn.Module, global_step: int) -> bool:
        ...


class TuneParticipant(_ParticipantInterface):
    """
    Federator for testing purposes.
    """

    def __init__(
        self,
        endpoint: str,
        token: str,
        temporary_dir: str,
        device: torch.device,
        enable_cpu_offload: bool,
    ) -> None:
        self._endpoint = endpoint
        self._token = token
        self._headers = {"Authorization": f"Bearer {self._token}"}
        self._temporary_dir = temporary_dir
        self._device = device
        self._enable_cpu_offload = enable_cpu_offload

        if not os.path.exists(self._temporary_dir):
            os.makedirs(self._temporary_dir)  # create temporary dir if necessary

        _, self._rank = utils.get_world_size_and_rank()
        self._is_rank_zero = self._rank == 0

        logger.info("TuneParticipant initialized.")

    """
    Raises:
        PermissionError: If the federator rejects the authentication token.
        ConnectionError: If connection to the federator fails.
    """

    def handshake(self) -> Dict[str, Any]:
        handshake_endpoint = f"{self._endpoint}/handshake"
        try:
            response = get(handshake_endpoint, headers=self._headers)
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Handshake request failed: {e}")
            raise ConnectionError(f"Handshake request failed: {e}") from e
        try:
            result = response.json()
        except Exception as e:
            logger.error(f"Failed to parse handshake JSON: {e}")
            raise ConnectionError(f"Failed to parse handshake JSON: {e}") from e
        if "h" not in result:
            logger.error("Handshake response missing key 'h'")
            raise ConnectionError("Handshake response missing key 'h'") from e
        self._h = result["h"]
        logger.info(
            f"Federator setup successfully with {result.get('participant_count')} participants, "
            f"at step {result.get('step')}, h: {self._h}."
        )
        return result["config"]

    def _get_with_retry(
        self, url: str, description: str, max_attempts: int = 10, delay: int = 1
    ):
        attempt = 0
        while attempt < max_attempts:
            try:
                resp = get(url, headers=self._headers)
                resp.raise_for_status()
                return resp
            except Exception as e:
                logger.warning(f"{description} failed at attempt {attempt + 1}: {e}")
                attempt += 1
                time.sleep(delay)
        raise ConnectionError(f"{description} failed after {max_attempts} attempts.")

    def _put_with_retry(
        self, url: str, files, description: str, max_attempts: int = 10, delay: int = 1
    ):
        attempt = 0
        while attempt < max_attempts:
            try:
                resp = put(url, files=files, headers=self._headers)
                resp.raise_for_status()
                return resp
            except Exception as e:
                logger.warning(f"{description} failed at attempt {attempt + 1}: {e}")
                attempt += 1
                time.sleep(delay)
        raise ConnectionError(f"{description} failed after {max_attempts} attempts.")

    @torch.no_grad()
    def synchronize(self, model: torch.nn.Module, global_step: int) -> bool:
        if global_step % self._h != 0:
            return False

        if self._is_rank_zero:
            response = self._get_with_retry(
                f"{self._endpoint}/status", "Get federator status"
            )
            sync_state = response.json()

        if torch.distributed.is_initialized():
            model_state_dict = training.gather_cpu_state_dict(
                model,
                self._is_rank_zero,
                device="cpu",
            )
        else:
            model_state_dict = model.state_dict()

        if self._is_rank_zero:
            out = io.BytesIO()
            try:
                torch.save(model_state_dict, out)
            except Exception as e:
                logger.error(f"Failed to save model state to memory: {e}")
                raise RuntimeError(f"Failed to save model state to memory: {e}") from e
        del model_state_dict

        temp_checkpoint_file = os.path.join(self._temporary_dir, "checkpoint.pt")

        if self._is_rank_zero:
            out.seek(0)
            put_response = self._put_with_retry(
                f"{self._endpoint}/checkpoint", {"file": out}, "Put checkpoint"
            )
            del out

            # Wait for federator to merge with a retry loop
            while True:
                curr_response = self._get_with_retry(
                    f"{self._endpoint}/status",
                    "Get federator status in merge",
                    max_attempts=1,
                )
                current_state = curr_response.json()
                if current_state["step"] == sync_state["step"] + 1:
                    sync_state = current_state
                    break
                time.sleep(1)

            chkpt_response = self._get_with_retry(
                f"{self._endpoint}/checkpoint", "Get federator checkpoint"
            )

        if self._is_rank_zero:
            try:
                checkpoint = torch.load(
                    io.BytesIO(chkpt_response.content), weights_only=True
                )
            except Exception as e:
                logger.error(f"Failed to load checkpoint from memory: {e}")
                raise RuntimeError(f"Failed to load checkpoint from memory: {e}") from e
            try:
                with open(temp_checkpoint_file, "wb") as f:
                    f.write(chkpt_response.content)
            except Exception as e:
                logger.error(f"Failed to save checkpoint to disk: {e}")
                raise RuntimeError(f"Failed to save checkpoint to disk: {e}") from e

        if torch.distributed.is_initialized():
            torch.distributed.barrier()  # Wait for rank 0 to save to disk
            if not self._is_rank_zero:
                try:
                    checkpoint = torch.load(temp_checkpoint_file, weights_only=True)
                except Exception as e:
                    logger.error(f"Failed to load checkpoint from disk: {e}")
                    raise RuntimeError(
                        f"Failed to load checkpoint from disk: {e}"
                    ) from e
            try:
                training.load_from_full_model_state_dict(
                    model,
                    checkpoint,
                    self._device,
                    strict=True,
                    cpu_offload=self._enable_cpu_offload,
                )
            except Exception as e:
                logger.error(f"Failed to load model state dict: {e}")
                raise RuntimeError(f"Failed to load model state dict: {e}") from e
        else:
            try:
                model.load_state_dict(checkpoint)
            except Exception as e:
                logger.error(f"Failed to load model state dict locally: {e}")
                raise RuntimeError(
                    f"Failed to load model state dict locally: {e}"
                ) from e

        return True
