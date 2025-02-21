# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import io
import time
from typing import Any, Dict, Protocol

import torch
from requests import get, put

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

    def __init__(self, endpoint: str, token: str) -> None:
        self._endpoint = endpoint
        self._token = token
        self._headers = {"Authorization": f"Bearer {self._token}"}
        logger.info("TuneParticipant initialized.")

    """
    Raises:
        PermissionError: If the federator rejects the authentication token.
        ConnectionError: If connection to the federator fails.
    """

    def handshake(self) -> Dict[str, Any]:
        handshake_endpoint = f"{self._endpoint}/handshake"
        result = get(handshake_endpoint, headers=self._headers)
        if result.status_code == 403:
            logger.error(
                f"Federator Handshake with {self._endpoint} failed to authenticate."
            )
            raise PermissionError("The federator has rejected the token.")

        if result.status_code != 200:
            logger.error(
                f"Federator Handshake with {self._endpoint} failed with status code {result.status_code}."
            )
            raise ConnectionError(
                f"Handshake failed with status code {result.status_code}."
            )

        result = result.json()

        participants = result["participant_count"]
        step = result["step"]
        h = result["h"]
        logger.info(
            f"Federator setup successfully with {participant_count} participants, at step {step}, h: {h}."
        )

        self._h = result["h"]

        return result["config"]

    """
    """

    @torch.no_grad()
    def synchronize(self, model: torch.nn.Module, global_step: int) -> bool:
        if global_step % self._h != 0:
            return False

        result = get(f"{self._endpoint}/status", headers=self._headers)
        if result.status_code != 200:
            logger.error(
                f"Failed to get federator status, HTTP error {result.status_code}."
            )
            raise ConnectionError(
                f"Failed to get federator status, HTTP error {result.status_code}."
            )
        sync_state = result.json()

        out = io.BytesIO()
        torch.save(model.state_dict(), out)
        out.seek(0)
        result = put(
            f"{self._endpoint}/checkpoint", files={"file": out}, headers=self._headers
        )

        while True:
            current_state = get(f"{self._endpoint}/status", headers=self._headers)
            if current_state.status_code != 200:
                logger.warning(
                    f"Failed to get federator status, HTTP error {current_state.status_code}. Retrying..."
                )
            else:
                current_state = current_state.json()
                if current_state["step"] == sync_state["step"] + 1:
                    sync_state = current_state
                    break

            time.sleep(1)  # Give the federator some time to merge.

        result = get(f"{self._endpoint}/checkpoint", headers=self._headers)
        if result.status_code != 200:
            logger.error(
                f"Failed to get federator checkpoint, HTTP error {result.status_code}."
            )
            raise ConnectionError(
                f"Failed to get federator checkpoint, HTTP error {result.status_code}."
            )

        checkpoint = torch.load(io.BytesIO(result.content), weights_only=True)
        model.load_state_dict(checkpoint)

        return True
