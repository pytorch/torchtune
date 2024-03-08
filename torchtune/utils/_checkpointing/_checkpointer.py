
import gc
import os
import torch

from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from torchtune.models import llama2
from torchtune.utils.logging import get_logger
from torchtune.utils._checkpointing._checkpointer_utils import CheckpointFormat, ModelType

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
        are directly saved in the ``TORCHTUNE_FORMAT``.

        * End-of-Training Checkpointing. In this scenario, the state-dict only contains the
        model weights. The checkpointer is responsible for converting the state-dict into the
        same format as the original checkpoint. In case the original checkpoint is read from
        multiple files, the checkpointer will split the final state-dict and ensure the keys
        are written to the correct files based on the ``weight_map``. To prevent the original
        files from being overwritten, the checkpointer will prefix the checkpoint filename
        with ``torchtune``.
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
    """
    Checkpointer for Full-Finetuning.

    Models Supported:
        * Llama2-7B

    Checkpoint Sources/Formats Supported:
        * Meta
        * HuggingFace
        * TorchTune (in case of resuming a training run from a previous TorchTune checkpoint)

    Args:
        checkpoint_dir (Path): Path to the directory where the checkpoint files are stored
        checkpoint_files (List of Paths): List of checkpoint files to load. ``save_checkpoint``
            ensures that the output checkpoint is written into files with the same keys and
            name prefixed with "torchtune" as compared to the original checkpoint
        checkpoint_format (CheckpointFormat): Format of the checkpoint files. Currently
            official checkpoints from the original authors (eg: Meta for Llama2) and HuggingFace
            are supported. Other models from HF Hub which follow a similar format should work
            out-of-the-box as well
        model_type (ModelType): Type of the model supported by this checkpoint format
        output_dir (Path): Path to the directory where the final checkpoint is written
        resume_from_checkpoint (bool): Whether training is resumed from a previous TorchTune
            checkpoint
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        checkpoint_files: List[Path],
        checkpoint_format: CheckpointFormat,
        model_type: ModelType,
        output_dir: Path,
        resume_from_checkpoint: bool = False,
    )-> None:

        # Fail fast if checkpoint files are invalid
        self._checkpoint_files: List[Path] = []
        for cpt_file in checkpoint_files:
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

        # dict which maps keys to checkpoint files, similar to the index.json
        # file in checkpoints with HF_FORMAT. This is used to ensure the final
        # checkpoint is written in the same way as the original checkpoint
        self._weight_map: Dict[str, str] = None

    def load_checkpoint(self, num_heads: int = 32, dim: int = 4096) -> Dict[str, Any]:
        """
        Loads the checkpoint from the checkpoint files.

        The state-dict structure depends on the source of checkpoint. When ``resume_from_checkpoint``
        is True, we're loading a TorchTune checkpoint previously saved mid-training. In this case,
        no conversion of the state-dict is required.

        Returns:
            Dict[str, Any]: State dict of the model
        """
        if self._resume_from_checkpoint:
            state_dict = self._load_torchtune_checkpoint()
        else:
            state_dict = self._load_external_checkpoint()
            state_dict = self.convert_to_torchtune_format(state_dict, num_heads, dim)
        return state_dict

    def save_checkpoint(
        self,
        checkpoint_dict: Dict[str, Any],
        intermediate_checkpoint: bool = False,
        intermediate_checkpoint_name: Optional[str] = None,
        num_heads: int = 32, dim: int = 4096
    ) -> None:
        """
        Saves the checkpoint to the output directory.

        The format of the output checkpoint depends on the ``intermediate_checkpoint`` flag.

            * When ``intermediate_checkpoint`` is True, the checkpoint is being written mid-training
            and contains more information that just the model weights, including information about the
            recipe state needed to correctly resume training in case of a failure. In this case, no
            conversion of the state-dict is required. We simply add additional information associated with
            the original checkpoint (dtype, weight_map and format) and write to file. Currently we write
            to a single ".pt" file.

            * When ``intermediate_checkpoint`` is False, the checkpoint is being written at the end of
            training. The state-dict is converted to the right format and is optionally split across files
            based on the key-to-file mapping in ``_weight_map``.

        Args:
            checkpoint_dict (Dict[str, Any]): State dict of the model
            intermediate_checkpoint (bool): Whether this is an intermediate checkpoint
            intermediate_checkpoint_name
        """
        self._output_dir.mkdir(exist_ok=True)

        # if this is an intermediate checkpoint, just add the original
        # checkpoint information and write to file.
        # Currently we only support writing to single file
        if intermediate_checkpoint:
            # Add the relevant checkpoint format information to the state dict
            checkpoint_dict["checkpoint_dtype"] = self._checkpoint_dtype
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
        else:
            # while writing the final checkpoint, first conver the state_dict
            checkpoint_dict = self.convert_from_torchtune_format(checkpoint_dict, num_heads, dim)

            # optionally split the state-dict across files. This is also where the
            # dtype conversion happens if necessary
            split_state_dicts = self._split_state_dict(checkpoint_dict)

            for filename, state_dict in split_state_dicts.items():
                # Update the filename so we don't overwrite the original checkpoints
                # in case output_dir is the same as checkpoint_dir
                filename = Path(filename)
                output_path = Path.joinpath(self._output_dir, filename.parent, ('torchtune_' + filename.name))
                torch.save(state_dict, output_path)
                logger.info(
                    "Model checkpoint of size "
                    f"{os.path.getsize(output_path) / 1000**3:.2f} GB "
                    f"saved to {output_path}"
                )

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
        if self._model_type == ModelType.LLAMA2_7B:
            if self._checkpoint_format == CheckpointFormat.META_FORMAT:
                converted_state_dict = llama2.meta_to_tune_llama2_7b(state_dict)
            elif self._checkpoint_format== CheckpointFormat.HF_FORMAT:
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
        if self._model_type == ModelType.LLAMA2_7B:
            if self._checkpoint_format == CheckpointFormat.META_FORMAT:
                converted_state_dict = llama2.tune_to_meta_llama2_7b(state_dict)
            elif self._checkpoint_format== CheckpointFormat.HF_FORMAT:
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


    def _load_torchtune_checkpoint(self) -> Dict[str, Any]:
        """
        Load a TorchTune checkpoint to resume a TorchTune training run.
        """
        # TorchTune checkpoints should be ".pt" files. Given that this is a common
        # checkpoint file extension, make sure this checkpoint is indeed a TorchTune
        # format checkpoint
        if self._checkpoint_format != CheckpointFormat.TORCHTUNE_FORMAT:
            raise ValueError(
                "When resuming a TorchTune training run, the checkpoint format is expected to "
                f'be "torchtune". Got {ckpt_format} instead.'
            )

        # Currently we output a single checkpoint file. This will change as we support
        # larger models. For now, raise an exception if this isn't true
        if self._checkpoint_files.__len__() != 1:
            raise ValueError(
                "When resuming a TorchTune training run, the checkpoint path should "
                f"should point to a file. Got {ckpt_path} which is not a file."
            )

        # TorchTune checkpoints are expected to be ".pt" files
        checkpoint_file = self._checkpoint_files[0]
        if checkpoint_file.suffix != ".pt":
            raise ValueError(
                'When resuming a TorchTune training run, the checkpoint should be ".pt" file. '
                f'Got a "{ckpt_path.suffix}" file instead. Make sure you are loading a valid '
                "TorchTune checkpoint."
            )

        # Intermediate TorchTune checkpoints contain a lot more information than just the model weights
        # This includes recipe state and information associated with original checkpoints. Extract
        # the checkpoint information, remove it from the dict, and pass onto the recipe
        checkpoint_dict = torch.load(checkpoint_file, map_location="cpu", mmap=True, weights_only=True)

        self._checkpoint_dtype = checkpoint_dict.pop("checkpoint_dtype", None)
        self._weight_map = checkpoint_dict.pop("weight_map", None)

        if self._checkpoint_dtype is None or self._weight_map is None:
            logger.warning(
                f"Checkpoint: {checkpoint_file} does not contain information about the original "
                "checkpoint. Continuing to load the checkpoint, but the final checkpoint will not be "
                "in a standard format."
            )

        # Overwrite the format with that of the original checkpoint so ``save_checkpoint`` writes
        # the final checkpoint correctly
        self._checkpoint_format = getattr(CheckpointFormat, checkpoint_dict.pop("checkpoint_format"))

        return checkpoint_dict

    def _load_external_checkpoint(self) -> Dict[str, torch.Tensor]:
        """
        Loads a checkpoint created outside of TorchTune.

            * Merge the state_dicts from multiple files
            * Extract information about the dtype
            * Create an index which maps keys to checkpoint files. This is used by ``save_checkpoint``
        """
        self._weight_map: Dict[str, str] = {}
        merged_state_dict: Dict[str, torch.Tensor] = {}

        for ckpt_file in self._checkpoint_files:
            state_dict = torch.load(
                ckpt_file, map_location="cpu", mmap=True, weights_only=True
            )
            # Extract the dtype
            _, value = next(iter(state_dict.items()))
            self._checkpoint_dtype = value.dtype

            # Create the index - this can potentially be done more efficiently
            for key in state_dict.keys():
                self._weight_map[key] = ckpt_file.name
            merged_state_dict.update(state_dict)

            # Don't store multiple copies of the state dict in memory.
            del state_dict
            gc.collect()


        logger.info(
            "Done creating the index for checkpoint files. \n"
            f"Index contains {len(self._weight_map.keys())} keys. "
            f"State Dict contains {len(merged_state_dict.keys())} keys. "
        )
        return merged_state_dict

    def _split_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Split the state dict into multiple state dicts based on the weight map. If needed,
        convert the dtype of the weights to match the original checkpoint.
        """
        # split the state dicts based on the index
        split_state_dicts: Dict[str, Dict[str, torch.Tensor]] = {}
        for key, weight in state_dict.items():
            filename = self._weight_map[key]
            if not filename in split_state_dicts:
                split_state_dicts[filename] = {}
            split_state_dicts[filename].update({key: weight.to(self._checkpoint_dtype)})
        return split_state_dicts
