# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import gc
import json
import os

from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

import torch
from torchtune import utils

from torchtune.models import convert_weights
from torchtune.utils._checkpointing._checkpointer_utils import (
    get_path,
    ModelType,
    safe_torch_load,
    save_config,
)
from torchtune.utils.logging import get_logger

logger = get_logger("DEBUG")


class _CheckpointerInterface(Protocol):
    """
    Interface implemented by Checkpointers in TorchTune.

    TorchTune checkpointers are designed to be composable components which can be plugged
    into any training recipe. Each checkpointer supports a specific set of models and training
    scenarios making these easy to understand, debug and extend. For example, the
    ``FullModelCheckpointer``s are used for loading and saving all of the model weights.
    This checkpointer can be used for Full-Finetuning scenarios or PEFT where the output is a
    merged checkpoint. In case the current suite of checkpointers are inadequate,
    users are encouraged to implement their own and contribute back to TorchTune.

    TorchTune is also designed to be "state-dict invariant". This means the checkpointer
    ensures that the output checkpoint has the same format as the original checkpoint i.e.
    the output checkpoint has the same keys split across the same number of files as the original
    checkpoint. Being "state-dict invariant" allows users to seamlessly use TorchTune checkpoints
    with their favorite post-training tools from the open-source ecosystem without writing
    TorchTune-specific convertors. To be "state-dict invariant", the ``load_checkpoint`` and
    ``save_checkpoint`` methods make use of the weight convertors available in
    ``torchtune/models/<model_folder>``.

    TorchTune Checkpointers support two checkpointing scenarios:
        * End-of-training Checkpointing. The model weights at the end of a completed training
            run are written out to file. The checkpointer ensures that the output checkpoint
            files have the same keys as the input checkpoint file used to begin training. The
            checkpointer also ensures that the keys are paritioned across the same number of
            files as the original checkpoint. This ensures that the original metadata files can
            be used as is, and the output checkpoint can be used with any tool that understands
            the original checkpoint format. This includes popular inference engines such as
            ``llama.cpp`` and ``gpt-fast``. The output state dict has the following format:
            {
                "key_1": weight
                ...
            }


        Mid-training Chekpointing. In addition to the model checkpoint files, we output an
            additional "recipe_state.pt" file for intermediate checkpoints. These are currently
            output at the end of each epoch, and contain information such as optimizer state,
            number of epochs completed etc which is needed to correctly resume a previously
            interrupted training run. The recipe is responsible for constructing the state dict
            with the information it needs. The checkpointer extracts the model state dict
            (key = "model") and writes everything else out to "recipe_state.pt". To prevent us
            from flooding ``output_dir`` with checkpoint files, the recipe state is overwritten
            at the end of each epoch. The output state dicts have the following formats:

            Model:
                {
                    "key_1": weight
                    ...
                }

            Recipe State:
                {
                    "optimizer": ...,
                    "epoch": ...,
                    ...
                }

    """

    def load_checkpoint(self, **kwargs) -> Dict[str, Any]:
        ...

    def save_checkpoint(self, state_dict: Dict[str, Any], **kwargs) -> None:
        ...


class FullModelTorchTuneCheckpointer(_CheckpointerInterface):
    """
    Checkpointer which reads and writes checkpoints in a format compatible with
    TorchTune. No conversion of weights is required.

    Currently this supports reading a single checkpoint file only. This will likely change as
    we add support for larger models.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        checkpoint_files: List[str],
        model_type: ModelType,
        output_dir: str,
        adapter_checkpoint: Optional[str] = None,
        recipe_checkpoint: Optional[str] = None,
        resume_from_checkpoint: bool = False,
    ) -> None:
        # Fail fast if ``checkpoint_files`` is invalid
        if len(checkpoint_files) != 1:
            raise ValueError(
                "Currently we only support reading from a single TorchTune checkpoint file. "
                f"Got {len(checkpoint_files)} files instead."
            )

        self._checkpoint_dir = Path(checkpoint_dir)
        self._checkpoint_path = get_path(self._checkpoint_dir, checkpoint_files[0])

        if not self._checkpoint_path.suffix == ".pt":
            raise ValueError(
                f"Checkpoint file {self._checkpoint_path} is not a valid checkpoint file. "
                "Checkpointer expects a valid .pt file."
            )

        self._adapter_checkpoint = (
            get_path(self._checkpoint_dir, adapter_checkpoint)
            if adapter_checkpoint
            else None
        )

        self._resume_from_checkpoint = resume_from_checkpoint
        self._model_type = model_type
        self._output_dir = Path(output_dir)

        # recipe_checkpoint contains the recipe state. This should be available if
        # resume_from_checkpoint is True
        self._recipe_checkpoint = None
        if self._resume_from_checkpoint:
            if recipe_checkpoint is None:
                raise ValueError(
                    "If resume_from_checkpoint is True, recipe_checkpoint file must be provided."
                )
            self._recipe_checkpoint = get_path(self._checkpoint_dir, recipe_checkpoint)

    def load_checkpoint(self, weights_only: bool = True) -> Dict[str, Any]:
        """
        Load TorchTune checkpoint from file. Currently only loading from a single file is supported.

        The output state_dict has the following format, with keys other than "model" only present if
        ``resume_from_checkpoint`` is True:
            {
                "model": {
                    "key_1": weight
                    ...
                },
                "optimizer": ...,
                ...
            }

        Args:
            weights_only (bool): flag passed down to torch.load. We expose this, because quantized models
                cannot be loaded with weights_only=True

        Returns:
            Dict[str, Any]: state_dict from the input checkpoint
        """
        state_dict: Dict[str:Any] = {}
        state_dict[utils.MODEL_KEY] = safe_torch_load(
            self._checkpoint_path, weights_only=weights_only
        )

        if self._adapter_checkpoint:
            adapter_state_dict = safe_torch_load(self._adapter_checkpoint)
            state_dict[utils.ADAPTER_KEY] = adapter_state_dict

        if self._resume_from_checkpoint:
            recipe_state = safe_torch_load(self._recipe_checkpoint)
            state_dict.update(recipe_state)
        return state_dict

    def save_checkpoint(
        self,
        state_dict: Dict[str, Any],
        epoch: int,
        intermediate_checkpoint: bool = False,
    ) -> None:
        """
        Save TorchTune checkpoint to file. If ``intermediate_checkpoint`` is True, an additional
        checkpoint file ``recipe_state.pt`` is created in ``_output_dir`` which contains the recipe
        state. The output state dicts have the following formats:

            Model:
                {
                    "key_1": weight
                    ...
                }

            Recipe State:
                {
                    "optimizer": ...,
                    "epoch": ...,
                    ...
                }

        Args:
            state_dict (Dict[str, Any]): State dict with model and (optionally) recipe state
            epoch (int): Current epoch number. This is added to the checkpoint file name to ensure
                we're not overwriting intermediate checkpoint files
            intermediate_checkpoint (bool): If True, save an additional checkpoint file with the
                recipe state
        """
        self._output_dir.mkdir(exist_ok=True)

        # Output file is always a .pt file with the epoch number in the name
        checkpoint_file = Path.joinpath(
            self._output_dir, f"torchtune_model_{epoch}"
        ).with_suffix(".pt")
        torch.save(state_dict[utils.MODEL_KEY], checkpoint_file)
        logger.info(
            "Model checkpoint of size "
            f"{os.path.getsize(checkpoint_file) / 1000**3:.2f} GB "
            f"saved to {checkpoint_file}"
        )

        if utils.ADAPTER_KEY in state_dict:
            output_path = Path.joinpath(
                self._output_dir, f"adapter_{epoch}"
            ).with_suffix(".pt")
            torch.save(state_dict[utils.ADAPTER_KEY], output_path)
            logger.info(
                "Adapter checkpoint of size "
                f"{os.path.getsize(output_path) / 1000**3:.2f} GB "
                f"saved to {output_path}"
            )

        # If the recipe state needs to be output, first remove the model state dict
        if intermediate_checkpoint:
            _ = state_dict.pop(utils.MODEL_KEY)
            _ = state_dict.pop(utils.ADAPTER_KEY, None)
            output_path = Path.joinpath(self._output_dir, "recipe_state.pt")
            torch.save(state_dict, output_path)
            logger.info(
                "Recipe checkpoint of size "
                f"{os.path.getsize(output_path) / 1000**3:.2f} GB "
                f"saved to {output_path}"
            )


class FullModelHFCheckpointer(_CheckpointerInterface):
    """
    Checkpointer which reads and writes checkpoints in HF's format. Example includes
    the Llama-2-7b-hf model from the meta-llama repo (https://huggingface.co/meta-llama/Llama-2-7b-hf)

    A few notes about the checkpoint reading logic:
    - HF checkpoint names usually ordered by ID (eg: 0001_of_0003, 0002_of_0003, etc.) To ensure
    we read the files in the right order, we sort the checkpoint file names before reading
    - Checkpoint conversion to and from HF's format requires access to model params which are
    read directly from the "config.json" file. This helps ensure we either load the weights
    correctly or error out in case of discrepancy between the HF checkpoint file and TorchTune's
    model implementations.

    Args:
        checkpoint_dir (str): Directory containing the checkpoint files
        checkpoint_files (List[str]): List of checkpoint files to load. Since the checkpointer takes care
            of sorting by file ID, the order in this list does not matter
        model_type (ModelType): Model type of the model for which the checkpointer is being loaded
        output_dir (str): Directory to save the checkpoint files
        adapter_checkpoint (Optional[str]): Path to the adapter weights. Default is None
        recipe_checkpoint (Optional[str]): Path to the recipe state checkpoint file. Default is None
        resume_from_checkpoint (bool): If True, the checkpointer will load the additional checkpoint files to
            resume training from a previous run. Default is False

    Raises:
        ValueError: If ``resume_from_checkpoint`` is True but ``recipe_checkpoint`` is None
    """

    def __init__(
        self,
        checkpoint_dir: str,
        checkpoint_files: List[str],
        model_type: ModelType,
        output_dir: str,
        adapter_checkpoint: Optional[str] = None,
        recipe_checkpoint: Optional[str] = None,
        resume_from_checkpoint: bool = False,
    ) -> None:
        self._checkpoint_dir = Path(checkpoint_dir)
        self._checkpoint_paths = self._validate_hf_checkpoint_files(checkpoint_files)
        self._adapter_checkpoint = (
            get_path(self._checkpoint_dir, adapter_checkpoint)
            if adapter_checkpoint
            else None
        )

        self._model_type = model_type
        self._output_dir = Path(output_dir)
        self._resume_from_checkpoint = resume_from_checkpoint

        # weight_map contains the state_dict key -> checkpoint file mapping so we can correctly
        # parition the state dict into output checkpoint files. This is updated during checkpoint
        # load
        self._weight_map: Dict[str, str] = None

        # the config.json file contains model params needed for state dict conversion
        self._config = json.loads(
            Path.joinpath(self._checkpoint_dir, "config.json").read_text()
        )

        # save config.json to output_dir
        save_config(self._output_dir, self._config)

        # recipe_checkpoint contains the recipe state. This should be available if
        # resume_from_checkpoint is True
        self._recipe_checkpoint = None
        if self._resume_from_checkpoint:
            if recipe_checkpoint is None:
                raise ValueError(
                    "If resume_from_checkpoint is True, recipe_checkpoint file must be provided."
                )
            self._recipe_checkpoint = get_path(self._checkpoint_dir, recipe_checkpoint)

    def _validate_hf_checkpoint_files(self, checkpoint_files: List[str]) -> List[Path]:
        """
        Validates that the checkpoint files exist and sorts based on ID.
        """
        checkpoint_paths: List[Path] = []
        for f in checkpoint_files:
            checkpoint_path = get_path(self._checkpoint_dir, f)
            checkpoint_paths.append(checkpoint_path)
        return sorted(checkpoint_paths)

    def load_checkpoint(self) -> Dict[str, Any]:
        """
        Load TorchTune checkpoint from file.

        The keys and weights from across all checkpoint files are merged into a single state_dict.
        We preserve the "state_dict key" <-> "checkpoint file mapping" in weight_map so we can
        write the state dict correctly in ``save_checkpoint``.

        Before returning, the model state dict is converted to a TorchTune compatible format using.

        Returns:
            state_dict (Dict[str, Any]): TorchTune checkpoint state dict

        Raises:
            ValueError: If the values in the input state_dict are not Tensors
        """
        self._weight_map = {}

        # merged state_dict contains keys and weights from all the checkpoint files
        merged_state_dict: Dict[str, torch.Tensor] = {}

        # converted_state_dict is the final state_dict passed to the recipe after the
        # keys are converted into the TorchTune format. This optionally also contains
        # the recipe state and adapter weights
        converted_state_dict: Dict[str, Dict[str, torch.Tensor]] = {}

        # _checkpoint_paths are already sorted so simply enumerate to generate the right id
        for cpt_idx, cpt_path in enumerate(self._checkpoint_paths):
            state_dict = safe_torch_load(cpt_path)
            for key, value in state_dict.items():
                # Ensure that the state dict is a flat dict of keys and tensors. Breaking this assumption
                # will break recipe code
                if not isinstance(value, torch.Tensor):
                    raise ValueError(
                        f"Expected all values in the state dict to be torch.Tensor. "
                        f"Found {type(value)} instead."
                    )
                # idx is written in the 4 digit format (eg: 0001, 0002, etc.)
                self._weight_map[key] = f"{cpt_idx+1:04}"
            merged_state_dict.update(state_dict)

            # delete the state_dict to free up memory; TODO check if this del is needed
            del state_dict
            gc.collect()

        converted_state_dict[utils.MODEL_KEY] = convert_weights.hf_to_tune(
            merged_state_dict,
            num_heads=self._config["num_attention_heads"],
            num_kv_heads=self._config["num_key_value_heads"],
            dim=self._config["hidden_size"],
        )

        if self._adapter_checkpoint:
            adapter_state_dict = safe_torch_load(self._adapter_checkpoint)
            converted_state_dict[utils.ADAPTER_KEY] = adapter_state_dict

        if self._resume_from_checkpoint:
            recipe_state = safe_torch_load(self._recipe_checkpoint)
            converted_state_dict.update(recipe_state)
        return converted_state_dict

    def save_checkpoint(
        self,
        state_dict: Dict[str, Any],
        epoch: int,
        intermediate_checkpoint: bool = False,
    ) -> None:
        """
        Save TorchTune checkpoint to file. If ``intermediate_checkpoint`` is True, an additional
        checkpoint file ``recipe_state.pt`` is created in ``_output_dir`` which contains the recipe
        state.

        The state_dict is first converted back to the HF format and then paritioned based on the
        ``_weight_map`` into separate checkpoint files.

        Args:
            state_dict (Dict[str, Any]): Checkpoint state dict to be written out to file
            epoch (int): Epoch number. Used to create the checkpoint file name
            intermediate_checkpoint (bool): If True, an additional checkpoint files for recipe state
                and (if applicable) adapter weights are created. Default is False
        """
        self._output_dir.mkdir(exist_ok=True)

        # convert the state_dict back to hf format; do this inplace
        state_dict[utils.MODEL_KEY] = convert_weights.tune_to_hf(
            state_dict[utils.MODEL_KEY],
            num_heads=self._config["num_attention_heads"],
            num_kv_heads=self._config["num_key_value_heads"],
            dim=self._config["hidden_size"],
        )

        # split the state_dict into separate dicts, one for each output checkpoint file
        split_state_dicts: Dict[str, Dict[str, torch.Tensor]] = {}
        for key, weight in state_dict[utils.MODEL_KEY].items():
            cpt_idx = self._weight_map[key]
            if cpt_idx not in split_state_dicts:
                split_state_dicts[cpt_idx] = {}
            split_state_dicts[cpt_idx].update({key: weight})

        # write the partitioned state dicts to the right checkpoint file
        for cpt_idx, model_state_dict in split_state_dicts.items():
            output_path = Path.joinpath(
                self._output_dir, f"hf_model_{cpt_idx}_{epoch}"
            ).with_suffix(".pt")
            torch.save(model_state_dict, output_path)
            logger.info(
                "Model checkpoint of size "
                f"{os.path.getsize(output_path) / 1000**3:.2f} GB "
                f"saved to {output_path}"
            )

        if utils.ADAPTER_KEY in state_dict:
            output_path = Path.joinpath(
                self._output_dir, f"adapter_{epoch}"
            ).with_suffix(".pt")
            torch.save(state_dict[utils.ADAPTER_KEY], output_path)
            logger.info(
                "Adapter checkpoint of size "
                f"{os.path.getsize(output_path) / 1000**3:.2f} GB "
                f"saved to {output_path}"
            )

        # If the recipe state needs to be output, first remove the model state dict
        # and if it exists, remove the adapter state dict as well
        if intermediate_checkpoint:
            _ = state_dict.pop(utils.MODEL_KEY)
            _ = state_dict.pop(utils.ADAPTER_KEY, None)
            output_path = Path.joinpath(self._output_dir, "recipe_state.pt")
            torch.save(state_dict, output_path)
            logger.info(
                "Recipe checkpoint of size "
                f"{os.path.getsize(output_path) / 1000**3:.2f} GB "
                f"saved to {output_path}"
            )


class FullModelMetaCheckpointer(_CheckpointerInterface):
    """
    Checkpointer which reads and writes checkpoints in Meta's format. Example includes
    the Llama-2-7b model from the meta-llama repo (https://huggingface.co/meta-llama/Llama-2-7b)

    Currently we support reading from a single checkpoint file only. Support for reading from
    sharded checkpoints is WIP.

    Args:
        checkpoint_dir (str): Directory containing the checkpoint files
        checkpoint_files (List[str]): List of checkpoint files to load. Currently this checkpointer only
            supports loading a single checkpoint file.
        model_type (ModelType): Model type of the model for which the checkpointer is being loaded
        output_dir (str): Directory to save the checkpoint files
        adapter_checkpoint (Optional[str]): Path to the adapter weights. Default is None
        recipe_checkpoint (Optional[str]): Path to the recipe state checkpoint file. Default is None
        resume_from_checkpoint (bool): If True, the checkpointer will load the additional checkpoint files to
            resume training from a previous run. Default is False

    Raises:
        ValueError: If ``checkpoint_files`` is not a list of length 1
        ValueError: If ``resume_from_checkpoint`` is True but ``recipe_checkpoint`` is None
    """

    def __init__(
        self,
        checkpoint_dir: str,
        checkpoint_files: List[str],
        model_type: ModelType,
        output_dir: str,
        adapter_checkpoint: Optional[str] = None,
        recipe_checkpoint: Optional[str] = None,
        resume_from_checkpoint: bool = False,
    ) -> None:
        # Fail fast if ``checkpoint_files`` is invalid
        if len(checkpoint_files) != 1:
            raise ValueError(
                "Currently we only support reading from a single TorchTune checkpoint file. "
                f"Got {len(checkpoint_files)} files instead."
            )

        self._checkpoint_dir = Path(checkpoint_dir)
        self._checkpoint_path = get_path(self._checkpoint_dir, checkpoint_files[0])

        self._adapter_checkpoint = (
            get_path(self._checkpoint_dir, adapter_checkpoint)
            if adapter_checkpoint
            else None
        )

        self._resume_from_checkpoint = resume_from_checkpoint
        self._model_type = model_type
        self._output_dir = Path(output_dir)

        # recipe_checkpoint contains the recipe state. This should be available if
        # resume_from_checkpoint is True
        self._recipe_checkpoint = None
        if self._resume_from_checkpoint:
            if recipe_checkpoint is None:
                raise ValueError(
                    "If resume_from_checkpoint is True, recipe_checkpoint file must be provided."
                )
            self._recipe_checkpoint = get_path(self._checkpoint_dir, recipe_checkpoint)

    def load_checkpoint(self) -> Dict[str, Any]:
        """
        Load TorchTune checkpoint from file. Currently only loading from a single file is supported.
        """
        state_dict: Dict[str:Any] = {}
        model_state_dict = safe_torch_load(self._checkpoint_path)
        state_dict[utils.MODEL_KEY] = convert_weights.meta_to_tune(model_state_dict)

        if self._adapter_checkpoint:
            adapter_state_dict = safe_torch_load(self._adapter_checkpoint)
            state_dict[utils.ADAPTER_KEY] = adapter_state_dict

        if self._resume_from_checkpoint:
            recipe_state = safe_torch_load(self._recipe_checkpoint)
            state_dict.update(recipe_state)
        return state_dict

    def save_checkpoint(
        self,
        state_dict: Dict[str, Any],
        epoch: int,
        intermediate_checkpoint: bool = False,
    ) -> None:
        """
        Save TorchTune checkpoint to file. If ``intermediate_checkpoint`` is True, an additional
        checkpoint file ``recipe_state.pt`` is created in ``_output_dir`` which contains the recipe
        state.

        Args:
            state_dict (Dict[str, Any]): Checkpoint state dict to be written out to file
            epoch (int): Epoch number. Used to create the checkpoint file name
            intermediate_checkpoint (bool): If True, an additional checkpoint files for recipe state
                and (if applicable) adapter weights are created. Default is False
        """
        self._output_dir.mkdir(exist_ok=True)
        model_state_dict = state_dict[utils.MODEL_KEY]
        state_dict[utils.MODEL_KEY] = convert_weights.tune_to_meta(model_state_dict)

        # Output file is always a .pt file with the epoch number in the name
        checkpoint_file = Path.joinpath(
            self._output_dir, f"meta_model_{epoch}"
        ).with_suffix(".pt")
        torch.save(state_dict[utils.MODEL_KEY], checkpoint_file)
        logger.info(
            "Model checkpoint of size "
            f"{os.path.getsize(checkpoint_file) / 1000**3:.2f} GB "
            f"saved to {checkpoint_file}"
        )

        if utils.ADAPTER_KEY in state_dict:
            output_path = Path.joinpath(
                self._output_dir, f"adapter_{epoch}"
            ).with_suffix(".pt")
            torch.save(state_dict[utils.ADAPTER_KEY], output_path)
            logger.info(
                "Adapter checkpoint of size "
                f"{os.path.getsize(output_path) / 1000**3:.2f} GB "
                f"saved to {output_path}"
            )

        # If the recipe state needs to be output, first remove the model state dict
        # and if it exists, remove the adapter state dict as well
        if intermediate_checkpoint:
            _ = state_dict.pop(utils.MODEL_KEY)
            _ = state_dict.pop(utils.ADAPTER_KEY, None)
            output_path = Path.joinpath(self._output_dir, "recipe_state.pt")
            torch.save(state_dict, output_path)
            logger.info(
                "Recipe checkpoint of size "
                f"{os.path.getsize(output_path) / 1000**3:.2f} GB "
                f"saved to {output_path}"
            )
