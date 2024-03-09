
import gc
import os
import torch

from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from torchtune.models import llama2
from torchtune.utils.logging import get_logger
from torchtune.utils.precision import PRECISION_DTYPE_TO_STR, PRECISION_STR_TO_DTYPE
from torchtune.utils._checkpointing._checkpointer_utils import (
    CheckpointFormat,
    ModelType,
    is_torchtune_checkpoint
)

logger = get_logger("DEBUG")


class _CheckpointerInterface(Protocol):
    """
    Interface implemented by Checkpointers in TorchTune.

    TorchTune checkpointers are designed to be composable components which can be plugged
    into any training recipe. Each checkpointer supports a specific set of models and training
    scenarios making these easy to understand, debug and extend. For example, the
    ``FullModelCheckpointer`` does not provide support for adapter-based or parameter
    efficient training methods. In case the current suite of checkpointers are inadequate,
    users can easily implement their own and contribute back to TorchTune.

    TorchTune is also designed to be "state-dict invariant". This means the checkpointer
    ensures that the output checkpoint has the same format as the original checkpoint i.e.
    the output checkpoint has the same keys, same dtype and is split across the same number
    of files as the original checkpoint. Being "state-dict invariant" allows users to
    seamlessly use TorchTune checkpoints with their favorite post-training tools from the
    open-source ecosystem without writing TorchTune-specific convertors. As a result, each
    checkpointer also provides methods for converting state dicts to and from an external
    format (eg: HuggingFace) to a TorchTune-compatible format (keys can be loaded into the
    torchtune model class).

    TorchTune Checkpointers support two checkpointing scenarios:
        * Mid-training Chekpointing. In this case the state-dict contains more information
        than just the model weights. It also contains the optimizer state dict and the training
        state of the recipe needed to correctly restart training. The construction of the
        state-dict, including figuring out what information is needed to correctly resume
        training, is handled by the recipe. The checkpointer doesn't know or care about how the
        state-dict is constructed. In this scenario, the checkpointer simply adds additional
        information about the original checkpoint (eg: format, weight map etc) to ensure the
        final checkpoint is constructured correctly in case the current training run fails and
        needs to be resumed. Intermediate checkpoints don't require any conversion since these
        are directly saved in the ``TORCHTUNE_FORMAT``. An intermediate checkpoint can have
        arbitrary keys:
            ```
            {
                "model": model_state_dict
                "optimizer: opt_state_dict
                "seed": seed_value
                ...
            }
            ```

        * End-of-Training Checkpointing. In this scenario, the state-dict only contains the
        model weights. The checkpointer is responsible for converting the state-dict into the
        same format as the original checkpoint. In case the original checkpoint is read from
        multiple files, the checkpointer will split the final state-dict and ensure the keys
        are written to the correct files based on the ``weight_map``. To prevent the original
        files from being overwritten, the checkpointer will prefix the checkpoint filename
        with ``torchtune``. We assume the state_dict has the following format:
            ```
            {
                "key_1": weight_tensor_1
                "key_2": weight_tensor_2
                ...
            }
            ```
    """

    def load_checkpoint(self, **kwargs) -> Dict[str, Any]:
        ...

    def save_checkpoint(self, state_dict: Dict[str, Any], **kwargs) -> None:
        ...

    def convert_to_torchtune_format(
        self,
        state_dict: Dict[str, Any],
        original_checkpoint_format: CheckpointFormat,
        **kwargs
    ) -> Dict[str, Any]:
        ...

    def convert_from_torchtune_format(
        self,
        state_dict: Dict[str, Any],
        final_ckpt_format: CheckpointFormat,
        **kwargs,
    ) -> Dict[str, Any]:
        ...


class FullModelCheckpointer(_CheckpointerInterface):

    def __init__(
        self,
        checkpoint_dir: Path,
        checkpoint_files: List[Path],
        checkpoint_format: CheckpointFormat,
        model_type: ModelType,
        output_dir: Path,
        resume_from_checkpoint: bool = False,
    ) -> None:
        # Fail fast if checkpoint files are invalid
        self._checkpoint_files: List[Path] = []
        for cpt_file in checkpoint_files:
            # Path ensures that ``joinpath`` does the right thing even if
            # cpt_file is actually a full path. No need to explicitly check this
            checkpoint_path = Path.joinpath(checkpoint_dir, cpt_file)
            if not checkpoint_path.exists() or not checkpoint_path.is_file():
                raise ValueError(f"Checkpoint file {checkpoint_path} is not a valid checkpoint file.")
            self._checkpoint_files.append(checkpoint_path)

        self._checkpoint_format = checkpoint_format
        self._model_type = model_type
        self._output_dir = output_dir
        self._resume_from_checkpoint = resume_from_checkpoint

        # these attributes will be updated when we load the checkpoints
        # ------------

        # different formats for the same model have different dtypes. Track this
        # to convert weights to the right format before writing to file
        self._checkpoint_dtype = None

        # dict which maps keys in the model state_dict to checkpoint file names,
        # similar to the index.json file for checkpoints from HF.
        # This is used to ensure the final checkpoint is written in the same
        # way as the original checkpoint
        self._weight_map: Dict[str, str] = None

    def load_checkpoint(self) -> Dict[str, Any]:
        """
        Loads the checkpoint from the checkpoint files.

        The state-dict structure depends on the checkpoint format. Knowledge of this
        structure is needed to ensure the internal state of the checkpointer is correctly
        updated and output checkpoints are correctly writen.

        Intermediate checkpoints from previous TorchTune runs contain more information than
        just the model weights. These checkpoints need to be handled differently.

        Returns:
            Dict[str, Any]: State dict of the model
        """
        state_dict: Dict[str, Any] = None

        if self._resume_from_checkpoint:
            state_dict = self._load_intermediate_checkpoint()
        else:
            # Explicitly branching based on format makes the code easier to read
            # and extend. Eg: New format becomes a new branch and changes to a given
            # format are contained to a single method
            if self._checkpoint_format == CheckpointFormat.TORCHTUNE_NEW:
                state_dict = self._load_initial_torchtune_checkpoint()
            elif self._checkpoint_format == CheckpointFormat.META:
                state_dict = self._load_initial_meta_checkpoint()
            elif self._checkpoint_format == CheckpointFormat.HF:
                state_dict = self._load_initial_hf_checkpoint()
            else:
                raise ValueError(
                    f"Checkpoint format {self._checkpoint_format} is not supported."
                )
        return state_dict

    def _load_intermediate_checkpoint(self) -> Dict[str, Any]:
        """
        Loads the intermediate checkpoint from the checkpoint files.
        """
        if self._checkpoint_format != CheckpointFormat.TORCHTUNE_RESTART:
            raise ValueError(
                f"Expected a checkpoint with format {CheckpointFormat.TORCHTUNE_RESTART}. "
                f"Got {self._checkpoint_format} instead."
            )
        if len(self._checkpoint_files) != 1:
            raise ValueError(
                "Expected a single checkpoint file. "
                f"Found {len(self._checkpoint_files)} files instead."
            )
        if self._checkpoint_files[0].suffix != ".pt":
            raise ValueError(
                f"Expected a single checkpoint .pt file. "
                f"Found {self._checkpoint_files[0].suffix}. instead."
            )

        state_dict = torch.load(self._checkpoint_files[0], map_location="cpu", mmap=True, weights_only=True)

        # Extract the checkpoint metadata and remove it from the dict. All of these keys are
        # expected to be in the checkpoint and so any errors should lead to a failure
        try:
            self._checkpoint_dtype = PRECISION_STR_TO_DTYPE[state_dict.pop("checkpoint_dtype")]
            self._weight_map = state_dict.pop("weight_map")
            self._checkpoint_format = getattr(CheckpointFormat, state_dict.pop("checkpoint_format"))
        except KeyError as e:
            raise ValueError(
                f"Checkpoint file {self._checkpoint_files[0]} is missing required metadata. "
                "Are you sure this is a valid TorchTune intermediate checkpoint?\n"
                "Metadata values: \n"
                f"checkpoint_dtype: {self._checkpoint_dtype}\n"
                f"weight_map: {self._weight_map}\n"
                f"checkpoint_format: {self._checkpoint_format}\n"
            ) from e

        return state_dict

    def _load_initial_torchtune_checkpoint(self) -> Dict[str, torch.Tensor]:
        """
        Loads the initial TorchTune checkpoint when kicking off a new training run.
        """
        if len(self._checkpoint_files) != 1:
            raise ValueError(
                "Expected a single checkpoint file for TorchTune checkpoints. "
                f"Found {len(self._checkpoint_files)} files instead."
            )
        if self._checkpoint_files[0].suffix != ".pt":
            raise ValueError(
                f"Expected a single checkpoint .pt file. "
                f"Found {self._checkpoint_files[0].suffix}. instead."
            )
        state_dict = torch.load(self._checkpoint_files[0], map_location="cpu", mmap=True, weights_only=True)

        # Update the metadata
        self._weight_map: Dict[str, str] = {}
        for key, value in state_dict.items():
            # Ensure that the state dict is a flat dict of keys and tensors. Breaking this assumption
            # will break recipe code
            if not isinstance(value, torch.Tensor):
                raise ValueError(
                    f"Expected all values in the state dict to be torch.Tensor. "
                    f"Found {type(value)} instead."
                )
            if self._checkpoint_dtype is None:
                self._checkpoint_dtype = value.dtype
            self._weight_map[key] = self._checkpoint_files[0].name

        return state_dict

    def _load_initial_meta_checkpoint(self) -> Dict[str, torch.Tensor]:
        """
        Loads the initial Meta-formatted checkpoint when kicking off a new training run.

        Currently, this function is identical to the HF counterpart. As we include support
        for other Meta'formatted checkpoints, this function will need to be updated.
        """
        self._weight_map: Dict[str, str] = {}
        merged_state_dict: Dict[str, torch.Tensor] = {}

        for cpt_file in self._checkpoint_files:
            state_dict = torch.load(cpt_file, map_location="cpu", mmap=True, weights_only=True)
            for key, value in state_dict.items():
                # Ensure that the state dict is a flat dict of keys and tensors. Breaking this assumption
                # will break recipe code
                if not isinstance(value, torch.Tensor):
                    raise ValueError(
                        f"Expected all values in the state dict to be torch.Tensor. "
                        f"Found {type(value)} instead."
                    )
                if self._checkpoint_dtype is None:
                    self._checkpoint_dtype = value.dtype
                self._weight_map[key] = cpt_file.name
            merged_state_dict.update(state_dict)

            # delete the state_dict to free up memory
            del state_dict
            gc.collect()

        return merged_state_dict

    def _load_initial_hf_checkpoint(self) -> Dict[str, torch.Tensor]:
        """
        Loads the initial HF-formatted checkpoint when kicking off a new training run.
        """
        self._weight_map: Dict[str, str] = {}
        merged_state_dict: Dict[str, torch.Tensor] = {}

        for cpt_file in self._checkpoint_files:
            state_dict = torch.load(cpt_file, map_location="cpu", mmap=True, weights_only=True)
            for key, value in state_dict.items():
                # Ensure that the state dict is a flat dict of keys and tensors. Breaking this assumption
                # will break recipe code
                if not isinstance(value, torch.Tensor):
                    raise ValueError(
                        f"Expected all values in the state dict to be torch.Tensor. "
                        f"Found {type(value)} instead."
                    )
                if self._checkpoint_dtype is None:
                    self._checkpoint_dtype = value.dtype
                self._weight_map[key] = cpt_file.name
            merged_state_dict.update(state_dict)

            # delete the state_dict to free up memory
            del state_dict
            gc.collect()

        return merged_state_dict

    def convert_to_torchtune_format(
        self,
        state_dict: Dict[str, torch.Tensor],
        num_heads: int = 32,
        dim: int = 4096,
    ) -> Dict[str, torch.Tensor]:
        """
        Convert an external checkpoint to a format that can be used by TorchTune.

        Args:
            state_dict (Dict[str, torch.Tensor]): State dict from an external checkpoint

        Returns:
            Dict[str, torch.Tensor]: State dict in TorchTune format

        Raises:
            NotImplementedError: If the model type or checkpoint format is not supported
        """
        converted_state_dict = {}
        if self._model_type == ModelType.LLAMA2:
            if self._checkpoint_format == CheckpointFormat.META:
                converted_state_dict = llama2.meta_to_tune_llama2_7b(state_dict)
            elif self._checkpoint_format== CheckpointFormat.HF:
                converted_state_dict = llama2.hf_to_tune_llama2_7b(state_dict, num_heads, dim)
            else:
                raise NotImplementedError(
                    f"Checkpoint Format {self._checkpoint_format} is not currently supported by "
                    "this checkpointer."
                )
        else:
            raise NotImplementedError(
                f"Model {model_type} is not currently supported by this checkpointer."
            )
        return converted_state_dict

    def convert_from_torchtune_format(
        self,
        state_dict: Dict[str, torch.Tensor],
        num_heads: int = 32,
        dim: int = 4096,
    ) -> Dict[str, torch.Tensor]:
        """
        Convert a TorchTune checkpoint to match the original checkpoint used for fine-tuning.

        Args:
            state_dict (Dict[str, torch.Tensor]): State dict from a TorchTune checkpoint

        Returns:
            Dict[str, torch.Tensor]: State dict in the original format

        Raises:
            NotImplementedError: If the model type or checkpoint format is not supported
        """
        converted_state_dict = {}
        if self._model_type == ModelType.LLAMA2:
            if self._checkpoint_format == CheckpointFormat.META:
                converted_state_dict = llama2.tune_to_meta_llama2_7b(state_dict)
            elif self._checkpoint_format== CheckpointFormat.HF:
                converted_state_dict = llama2.tune_to_hf_llama2_7b(state_dict, num_heads, dim)
            else:
                raise NotImplementedError(
                    f"Checkpoint Format {self._checkpoint_format} is not currently supported by "
                    "this checkpointer."
                )
        else:
            raise NotImplementedError(
                f"Model {model_type} is not currently supported by this checkpointer."
            )
        return converted_state_dict

    def save_checkpoint(
        self,
        checkpoint_dict: Dict[str, Any],
        intermediate_checkpoint: bool = False,
        intermediate_checkpoint_name: Optional[str] = None,
    ) -> None:
        """
        Saves the checkpoint to the output directory.

        The format of the output checkpoint depends on the ``intermediate_checkpoint`` flag.

            * When ``intermediate_checkpoint`` is True, the checkpoint is being written mid-training
            and contains more information that just the model weights, including information about the
            recipe state needed to correctly resume training in case of a failure. Currently we write
            to a single ".pt" file.

            * When ``intermediate_checkpoint`` is False, the checkpoint is being written at the end of
            training. The state-dict is optionally split across files based on the key-to-file mapping
            in ``_weight_map`` and is written out to ``len(checkpoint_files)`` files.

        Args:
            checkpoint_dict (Dict[str, Any]): State dict of the model
            intermediate_checkpoint (bool): Whether this is an intermediate checkpoint
            intermediate_checkpoint_name (str, Optional): Name for the intermediate checkpoint file
        """
        self._output_dir.mkdir(exist_ok=True)

        if intermediate_checkpoint:
            self._save_intermediate_checkpoint(checkpoint_dict, intermediate_checkpoint_name)
        else:
            # Explicitly branching based on format makes the code easier to read
            # and extend. Eg: New format becomes a new branch and changes to a given
            # format are contained to a single method
            if self._checkpoint_format == CheckpointFormat.TORCHTUNE_NEW:
                self._save_final_torchtune_checkpoint(checkpoint_dict)
            elif self._checkpoint_format == CheckpointFormat.META:
                self._save_final_meta_checkpoint(checkpoint_dict)
            elif self._checkpoint_format == CheckpointFormat.HF:
                self._save_final_hf_checkpoint(checkpoint_dict)
            else:
                raise ValueError(
                    f"Checkpoint format {self._checkpoint_format} is not supported."
                )

    def _save_intermediate_checkpoint(
        self,
        checkpoint_dict: Dict[str, Any],
        intermediate_checkpoint_name: str
    ) -> None:
        """
        Saves the intermediate checkpoint to a single file.
        """
        # if this is an intermediate checkpoint, just add the original
        # checkpoint information and write to file.
        # Currently we only support writing to single file
        checkpoint_dict["checkpoint_dtype"] = PRECISION_DTYPE_TO_STR[self._checkpoint_dtype]
        checkpoint_dict["weight_map"] = self._weight_map
        checkpoint_dict["checkpoint_format"] = self._checkpoint_format.name

        # We write to a single ".pt" file irrespective of the extension provided by the recipe
        output_path = Path.joinpath(self._output_dir, intermediate_checkpoint_name).with_suffix(".pt")
        torch.save(checkpoint_dict, output_path)
        logger.info(
            "Model checkpoint of size "
            f"{os.path.getsize(output_path) / 1000**3:.2f} GB "
            f"saved to {output_path}"
        )

    def _save_final_torchtune_checkpoint(self, checkpoint_dict: Dict[str, torch.Tensor]):
        """
        Saves a TorchTune-format checkpoint to the output directory. Currently only
        supports writing to a single file.
        """
        for key, _ in checkpoint_dict.items():
            checkpoint_dict[key] = checkpoint_dict[key].to(self._checkpoint_dtype)

        ckpt_file = self._checkpoint_files[0]
        output_path = Path.joinpath(self._output_dir, ('torchtune_' + filename.name))
        torch.save(checkpoint_dict, output_path)
        logger.info(
            "Model checkpoint of size "
            f"{os.path.getsize(output_path) / 1000**3:.2f} GB "
            f"saved to {output_path}"
        )

    def _save_final_meta_checkpoint(self, checkpoint_dict: Dict[str, torch.Tensor]):
        """
        Saves a Meta-format checkpoint to the output directory.

        Currently, this function is identical to the HF counterpart. As we include support
        for other Meta'formatted checkpoints, this function will need to be updated.
        """
        split_state_dicts: Dict[str, Dict[str, torch.Tensor]] = {}

        for key, weight in checkpoint_dict.items():
            filename = self._weight_map[key]
            if not filename in split_state_dicts:
                split_state_dicts[filename] = {}
            split_state_dicts[filename].update({key: weight.to(self._checkpoint_dtype)})

        for filename, state_dict in split_state_dicts.items():
            # Update the filename so we don't overwrite the original checkpoints
            # in case output_dir is the same as checkpoint_dir
            filename = Path(filename)
            output_path = Path.joinpath(self._output_dir, ('torchtune_' + filename.name))
            torch.save(state_dict, output_path)
            logger.info(
                "Model checkpoint of size "
                f"{os.path.getsize(output_path) / 1000**3:.2f} GB "
                f"saved to {output_path}"
            )

    def _save_final_hf_checkpoint(self, checkpoint_dict: Dict[str, torch.Tensor]):
        """
        Saves a Meta-format checkpoint to the output directory.
        """
        split_state_dicts: Dict[str, Dict[str, torch.Tensor]] = {}

        for key, weight in checkpoint_dict.items():
            filename = self._weight_map[key]
            if not filename in split_state_dicts:
                split_state_dicts[filename] = {}
            split_state_dicts[filename].update({key: weight.to(self._checkpoint_dtype)})

        for filename, state_dict in split_state_dicts.items():
            # Update the filename so we don't overwrite the original checkpoints
            # in case output_dir is the same as checkpoint_dir
            filename = Path(filename)
            output_path = Path.joinpath(self._output_dir, ('torchtune_' + filename.name))
            torch.save(state_dict, output_path)
            logger.info(
                "Model checkpoint of size "
                f"{os.path.getsize(output_path) / 1000**3:.2f} GB "
                f"saved to {output_path}"
            )
