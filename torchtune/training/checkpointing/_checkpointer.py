# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import gc
import json
import os
import time
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Optional, Protocol, Union

import torch
import torch.distributed as dist
from fsspec.core import url_to_fs
from safetensors.torch import save as save_safetensors
from torch.distributed.checkpoint import (
    async_save,
    DefaultLoadPlanner,
    FileSystemReader,
    FileSystemWriter,
    load,
    save,
)
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict_from_keys

from torchtune import training
from torchtune.models import convert_weights
from torchtune.training.checkpointing._utils import (
    ADAPTER_CONFIG_FNAME,
    ADAPTER_MODEL_FNAME,
    check_outdir_not_in_ckptdir,
    copy_files,
    get_adapter_checkpoint_path,
    get_all_checkpoints_in_dir,
    get_model_checkpoint_path,
    get_most_recent_checkpoint,
    get_recipe_checkpoint_path,
    ModelType,
    prune_surplus_checkpoints,
    RECIPE_STATE_DIRNAME,
    REPO_ID_FNAME,
    safe_torch_load,
    SHARD_FNAME,
    SUFFIXES_TO_NOT_COPY,
)
from torchtune.utils import get_logger, get_world_size_and_rank, log_rank_zero

logger = get_logger("DEBUG")


class _CheckpointerInterface(Protocol):
    """
    Interface implemented by Checkpointers in torchtune.

    torchtune checkpointers are designed to be composable components which can be plugged
    into any training recipe. Each checkpointer supports a specific set of models and training
    scenarios making these easy to understand, debug and extend. For example, the
    ``FullModelCheckpointer``s are used for loading and saving all of the model weights.
    This checkpointer can be used for Full-Finetuning scenarios or PEFT where the output is a
    merged checkpoint. In case the current suite of checkpointers are inadequate,
    users are encouraged to implement their own and contribute back to torchtune.

    torchtune is also designed to be "state-dict invariant". This means the checkpointer
    ensures that the output checkpoint has the same format as the original checkpoint i.e.
    the output checkpoint has the same keys split across the same number of files as the original
    checkpoint. Being "state-dict invariant" allows users to seamlessly use torchtune checkpoints
    with their favorite post-training tools from the open-source ecosystem without writing
    torchtune-specific convertors. To be "state-dict invariant", the ``load_checkpoint`` and
    ``save_checkpoint`` methods make use of the weight convertors available in
    ``torchtune/models/<model_folder>``.

    torchtune Checkpointers support two checkpointing scenarios:
        * End-of-training Checkpointing. The model weights at the end of a completed training
            run are written out to file. The checkpointer ensures that the output checkpoint
            files have the same keys as the input checkpoint file used to begin training. The
            checkpointer also ensures that the keys are partitioned across the same number of
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

    def load_checkpoint(self, **kwargs) -> dict[str, Any]:
        ...

    def save_checkpoint(self, state_dict: dict[str, Any], **kwargs) -> None:
        ...


class FullModelTorchTuneCheckpointer(_CheckpointerInterface):
    """
    Checkpointer which reads and writes checkpoints in a format compatible with
    torchtune. No conversion of weights is required.

    Currently this supports reading a single checkpoint file only. This will likely change as
    we add support for larger models.

    Args:
        checkpoint_dir (str): Directory containing the checkpoint files
        checkpoint_files (list[str]): list of checkpoint files to load. Since the checkpointer takes care
            of sorting by file ID, the order in this list does not matter
        model_type (str): Model type of the model for which the checkpointer is being loaded, e.g. LLAMA3.
        output_dir (str): Directory to save the checkpoint files
        adapter_checkpoint (Optional[str]): Path to the adapter weights. If None,
            and `should_load_recipe_state=True`, then look for adapter_model.pt in output_dir/epoch_{largest_epoch}.
            Default is None.
        recipe_checkpoint (Optional[str]): Path to the recipe state checkpoint file. If None,
            and `should_load_recipe_state=True`, then look for recipe_state.pt in output_dir/RECIPE_STATE_DIRNAME.
            Default is None.
        resume_from_checkpoint (bool): If True, the checkpointer will load the additional checkpoint files corresponding to
            the recipe state from a previous run. Default is False. This flag is deprecated. Please use the
            should_load_recipe_state flag instead.
        should_load_recipe_state (bool): If True, the checkpointer will load the additional checkpoint files corresponding to
            the recipe state from a previous run. Default is False

    Raises:
        ValueError: If more than one checkpoint file is provided
    """

    def __init__(
        self,
        checkpoint_dir: str,
        checkpoint_files: list[str],
        model_type: str,
        output_dir: str,
        adapter_checkpoint: Optional[str] = None,
        recipe_checkpoint: Optional[str] = None,
        resume_from_checkpoint: bool = False,
        should_load_recipe_state: bool = False,
    ) -> None:
        # Fail fast if ``checkpoint_files`` is invalid
        # TODO: support loading more than one file
        if len(checkpoint_files) != 1:
            raise ValueError(
                "Currently we only support reading from a single checkpoint file. "
                f"Got {len(checkpoint_files)} files instead."
            )

        self._checkpoint_dir = Path(checkpoint_dir)
        self._should_load_recipe_state = should_load_recipe_state

        if resume_from_checkpoint:
            self._should_load_recipe_state = resume_from_checkpoint
            logger.warning(
                "*resume_from_checkpoint is deprecated. Please use the 'should_load_recipe_state' instead"
            )

        self._model_type = ModelType[model_type]
        self._output_dir = Path(output_dir)
        check_outdir_not_in_ckptdir(
            ckpt_dir=self._checkpoint_dir, out_dir=self._output_dir
        )
        self._output_dir.mkdir(parents=True, exist_ok=True)

        #  resume from adapter_model ckpt
        self._adapter_checkpoint = get_adapter_checkpoint_path(
            output_dir=self._output_dir,
            adapter_checkpoint=adapter_checkpoint,
            should_load_recipe_state=self._should_load_recipe_state,
            pattern=r"^step_(\d+)",
        )

        # resume recipe_state ckpt
        self._recipe_checkpoint = get_recipe_checkpoint_path(
            output_dir=self._output_dir,
            checkpoint_dir=self._checkpoint_dir,
            recipe_checkpoint=recipe_checkpoint,
            should_load_recipe_state=self._should_load_recipe_state,
        )

        # get ckpt paths
        self._checkpoint_paths = get_model_checkpoint_path(
            checkpoint_files=checkpoint_files,
            checkpoint_dir=self._checkpoint_dir,
            output_dir=self._output_dir,
            should_load_recipe_state=self._should_load_recipe_state,
            has_adapter_checkpoint=self._adapter_checkpoint is not None,
        )

        # we currently accept only a single file
        self._checkpoint_path = self._checkpoint_paths[0]

        if self._should_load_recipe_state:
            logger.info(
                "Loading the recipe state using: "
                f"\n\tcheckpoint_paths: {[str(path) for path in self._checkpoint_paths]}"
                f"\n\trecipe_checkpoint: {self._recipe_checkpoint}"
                f"\n\tadapter_checkpoint: {self._adapter_checkpoint}"
            )

    def load_checkpoint(self, weights_only: bool = True) -> dict[str, Any]:
        """
        Load torchtune checkpoint from file. Currently only loading from a single file is supported.

        The output state_dict has the following format, with keys other than "model" only present if
        ``should_load_recipe_state`` is True:

        >>>     {
        >>>         "model": {
        >>>             "key_1": weight
        >>>             ...
        >>>         },
        >>>         "optimizer": {...},
        >>>         ...
        >>>     }

        Args:
            weights_only (bool): flag passed down to torch.load. We expose this, because quantized models
                cannot be loaded with weights_only=True

        Returns:
            dict[str, Any]: state_dict from the input checkpoint
        """
        state_dict: dict[str:Any] = {}
        state_dict[training.MODEL_KEY] = safe_torch_load(
            self._checkpoint_path, weights_only=weights_only
        )

        if self._adapter_checkpoint:
            adapter_state_dict = safe_torch_load(self._adapter_checkpoint)
            state_dict[training.ADAPTER_KEY] = adapter_state_dict

        if self._should_load_recipe_state:
            recipe_state = safe_torch_load(self._recipe_checkpoint, mmap=False)
            state_dict.update(recipe_state)
        return state_dict

    def save_checkpoint(
        self,
        state_dict: dict[str, Any],
        epoch: int,
        intermediate_checkpoint: bool = False,
        adapter_only: bool = False,
        **kwargs,
    ) -> None:
        """
        Save torchtune checkpoint to file. If ``intermediate_checkpoint`` is True, an additional
        checkpoint file ``recipe_state.pt`` is created in ``_output_dir/RECIPE_STATE_DIRNAME``
        which contains the recipe state. The output state dicts have the following formats:

        >>> # Model
        >>> {
        >>>     "key_1": weight
        >>>     ...
        >>> }
        >>>
        >>> # Recipe state
        >>> {
        >>>     "optimizer": ...,
        >>>     "epoch": ...,
        >>>     ...
        >>> }

        Args:
            state_dict (dict[str, Any]): State dict with model and (optionally) recipe state
            epoch (int): Current epoch number. This is added to the checkpoint file name to ensure
                we're not overwriting intermediate checkpoint files
            intermediate_checkpoint (bool): If True, save an additional checkpoint file with the
                recipe state
            adapter_only (bool): If True, only save the adapter weights. Default is False
            **kwargs: Ignored keyword arguments to maintain compatibility with the Checkpointer interface

        Raises:
            ValueError: if ``adapter_only`` is True and adapter checkpoint not found in state_dict.
        """
        # Output file is always a .bin file with the epoch number in the name
        if not adapter_only:
            shard_name = SHARD_FNAME.format(
                cpt_idx="1".zfill(5), num_shards="1".zfill(5)
            )
            output_path = Path.joinpath(
                self._output_dir, f"epoch_{epoch}", shard_name
            ).with_suffix(".bin")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(state_dict[training.MODEL_KEY], output_path)
            logger.info(
                "Model checkpoint of size "
                f"{os.path.getsize(output_path) / 1024**3:.2f} GiB "
                f"saved to {output_path}"
            )

        if training.ADAPTER_KEY in state_dict:
            output_path = Path.joinpath(
                self._output_dir, f"epoch_{epoch}", ADAPTER_MODEL_FNAME
            ).with_suffix(".pt")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(state_dict[training.ADAPTER_KEY], output_path)
            logger.info(
                "Adapter checkpoint of size "
                f"{os.path.getsize(output_path) / 1024**3:.2f} GiB "
                f"saved to {output_path}"
            )

        elif adapter_only:
            raise ValueError(
                "Adapter checkpoint not found in state_dict. Please ensure that the state_dict contains adapter weights."
            )

        # Save all files in ckpt_dir, except model weights and mapping, to output_dir/epoch_{epoch}
        # So its easy to run inference with the model using this epoch's checkpoint
        copy_files(
            self._checkpoint_dir,
            Path.joinpath(self._output_dir, f"epoch_{epoch}"),
            ignore_suffixes=SUFFIXES_TO_NOT_COPY,
        )

        # If the recipe state needs to be output, first remove the model state dict
        if intermediate_checkpoint:
            _ = state_dict.pop(training.MODEL_KEY, None)
            _ = state_dict.pop(training.ADAPTER_KEY, None)
            _ = state_dict.pop(training.ADAPTER_CONFIG, None)
            output_path = Path.joinpath(
                self._output_dir, f"epoch_{epoch}", "recipe_state.pt"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(state_dict, output_path)
            logger.info(
                "Recipe checkpoint of size "
                f"{os.path.getsize(output_path) / 1024**3:.2f} GiB "
                f"saved to {output_path}"
            )
        else:
            logger.info("Saving final epoch checkpoint.")
            if adapter_only:
                logger.info(
                    "Please note that you have set adapter_only=True, so only adapter weights will be saved."
                    "You need to merge the adapter weights into your base model for further use. "
                    f"See {self.__class__.__name__}.save_checkpoint for more details."
                )
            else:
                logger.info(
                    "The full model checkpoint, including all weights and configurations, has been saved successfully."
                    "You can now use this checkpoint for further training or inference."
                )


class FullModelHFCheckpointer(_CheckpointerInterface):
    """
    Checkpointer which reads and writes checkpoints in HF's format. For LoRA models this includes
    saving checkpoints in a format that can be loaded into PEFT via e.g. ``from_pretrained``. Examples include
    the Llama-2-7b-hf model from the meta-llama repo (https://huggingface.co/meta-llama/Llama-2-7b-hf).

    Note:
        HF checkpoint names are usually ordered by ID (eg: 0001_of_0003, 0002_of_0003, etc.) To ensure \
        we read the files in the right order, we sort the checkpoint file names before reading.

    Note:
        Checkpoint conversion to and from HF's format requires access to model params which are \
        read directly from the ``config.json`` file. This helps ensure we either load the weights \
        correctly or error out in case of discrepancy between the HF checkpoint file and torchtune's \
        model implementations.

    Args:
        checkpoint_dir (str): Directory containing the checkpoint files
        checkpoint_files (Union[list[str], dict[str, str]]): list of checkpoint files to load or a dictionary
            containing the keys keys ["filename_format", "max_filename"]. Since the checkpointer takes care
            of sorting by file ID, the order in this list does not matter.
        model_type (str): Model type of the model for which the checkpointer is being loaded, e.g. LLAMA3.
        output_dir (Optional[str]): Directory to save the checkpoint files, default None.
        adapter_checkpoint (str): Path to the adapter weights. If None,
            and `should_load_recipe_state=True`, then look for adapter_model.pt in output_dir/epoch_{largest_epoch}.
            Default is "adapter_model.pt".
        recipe_checkpoint (str): Path to the recipe state checkpoint file. If None,
            and `should_load_recipe_state=True`, then look for recipe_state.pt in output_dir/RECIPE_STATE_DIRNAME.
            Default is "recipe_state.pt".
        resume_from_checkpoint (bool): If True, the checkpointer will load the additional checkpoint files corresponding to
            the receipe state from a previous run. Default is False. This flag is deprecated. Please use
            the should_load_recipe_state flag instead.
        safe_serialization (bool): If True, the checkpointer will save the checkpoint file using `safetensors`.
            Default is True.
        should_load_recipe_state (bool): If True, the checkpointer will load the additional checkpoint files corresponding to
            the receipe state from a previous run. Default is False
        keep_last_n_checkpoints (Optional[int]): How many checkpoints to keep. If None, all checkpoints are kept.
        enable_dcp (bool): If True, the checkpointer will load the checkpoint file using dcp checkpointing apis.
            This is currently an experimental feature.

    Raises:
        ValueError: If recipe_state cannot be found when should_load_recipe_state is True
    """

    def __init__(
        self,
        checkpoint_dir: str,
        checkpoint_files: Union[list[str], dict[str, str]],
        model_type: str,
        output_dir: Optional[str] = None,
        adapter_checkpoint: str = "adapter_model.pt",
        recipe_checkpoint: str = "recipe_state.pt",
        resume_from_checkpoint: bool = False,
        safe_serialization: bool = True,
        should_load_recipe_state: bool = False,
        *,
        keep_last_n_checkpoints: Optional[int] = None,
        enable_dcp: bool = False,
    ) -> None:
        self._checkpoint_dir = checkpoint_dir
        self._keep_last_n_checkpoints = keep_last_n_checkpoints
        self._safe_serialization = safe_serialization
        self._model_type = ModelType[model_type]
        self._enable_dcp = enable_dcp
        self._output_dir = output_dir
        self._should_load_recipe_state = should_load_recipe_state
        if resume_from_checkpoint:
            self._should_load_recipe_state = resume_from_checkpoint
            logger.warning(
                "*resume_from_checkpoint is deprecated. Please use the 'should_load_recipe_state' instead"
            )

        if recipe_checkpoint != "recipe_state.pt":
            # I don't want to log warning for None, b/c that's been the default for a long time
            if recipe_checkpoint is not None:
                logger.warning(
                    "recipe_checkpoint is deprecated. torchtune will always save the recipe state under "
                    "output_dir / epoch_x (or step_x). If you are trying to resume from a specific checkpoint, then "
                    "you can pass in checkpoint_dir=PATH/epoch_x (or step_x). We will then load PATH/epoch_1/recipe_state.pt"
                )
            recipe_checkpoint = "recipe_state.pt"

        # Create fsspec filesystem for the checkpoint directory
        self._input_fs, _ = url_to_fs(checkpoint_dir)

        # Initialize the output directory if one is specified
        if self._output_dir is not None:
            check_outdir_not_in_ckptdir(
                ckpt_dir=checkpoint_dir, out_dir=self._output_dir
            )
            self._output_fs, _ = url_to_fs(self._output_dir)
            self._output_fs.mkdirs(self._output_dir, exist_ok=True)

        # weight_map contains the state_dict key -> checkpoint file mapping so we can correctly
        # parition the state dict into output checkpoint files. This is updated during checkpoint load
        self._weight_map: dict[str, str] = {}

        # the config.json file contains model params needed for state dict conversion
        self._config = None
        with self._input_fs.open(
            os.path.join(checkpoint_dir, "config.json"), "r"
        ) as json_file:
            self._config = json.loads(json_file.read())

        # repo_id is necessary for when saving an adapter config, so its compatible with HF.
        # This json file is produced and saved in the download step.
        # contents are {"repo_id": "some_model/some_model_version"}
        repo_id_path = os.path.join(checkpoint_dir, REPO_ID_FNAME) + ".json"
        self.repo_id = None
        if self._input_fs.exists(repo_id_path):
            with self._input_fs.open(repo_id_path, "r") as json_file:
                data = json.load(json_file)
                self.repo_id = data.get("repo_id")

        if self._should_load_recipe_state:
            assert output_dir is not None
            if "step" in self._checkpoint_dir or "epoch" in self._checkpoint_dir:
                # If there's a step or epoch in the name, we assume it's loading from a predetermined ckpt
                checkpoint_dir_to_load_from = self._checkpoint_dir
            else:
                most_recent_checkpoint = get_most_recent_checkpoint(
                    dir=Path(output_dir)
                )
                if most_recent_checkpoint is None:
                    raise ValueError(
                        "Recipe state cannot be loaded because no checkpoints were found in the output directory."
                    )
                checkpoint_dir_to_load_from = most_recent_checkpoint

            self._recipe_checkpoint = os.path.join(
                checkpoint_dir_to_load_from, recipe_checkpoint
            )
            assert os.path.exists(
                self._recipe_checkpoint
            ), f"{recipe_checkpoint} not found in {checkpoint_dir_to_load_from}"

            self._adapter_checkpoint = os.path.join(
                checkpoint_dir_to_load_from, adapter_checkpoint
            )
            self._checkpoint_paths = get_model_checkpoint_path(
                checkpoint_files=checkpoint_files,
                checkpoint_dir=checkpoint_dir,
                output_dir=checkpoint_dir_to_load_from,
                should_load_recipe_state=True,
                has_adapter_checkpoint=os.path.exists(self._adapter_checkpoint),
            )
        else:
            assert output_dir is not None
            self._checkpoint_paths = get_model_checkpoint_path(
                checkpoint_files=checkpoint_files,
                checkpoint_dir=checkpoint_dir,
                output_dir=output_dir,
                should_load_recipe_state=False,
                has_adapter_checkpoint=False,
            )

    def _load_checkpoint_with_dcp_hf(self):
        from torch.distributed.checkpoint._hf_planner import _HuggingFaceLoadPlanner
        from torch.distributed.checkpoint._hf_storage import _HuggingFaceStorageReader
        from torch.distributed.checkpoint.state_dict_loader import load

        # DCP load using the storage reader
        hf_storage_reader = _HuggingFaceStorageReader(path=self._checkpoint_dir)
        metadata = hf_storage_reader.read_metadata()
        state_dict = {}
        for key in metadata.state_dict_metadata.keys():
            # arbitrary value to ensure that the state_dict is not empty
            state_dict[key] = torch.empty(1)

        weight_map = metadata.storage_data

        load(
            state_dict=state_dict,
            storage_reader=hf_storage_reader,
            planner=_HuggingFaceLoadPlanner(allow_tensor_resize=True),
        )

        return state_dict, weight_map

    def load_checkpoint(self) -> dict[str, Any]:
        """
        Load HF checkpoint from file.

        The keys and weights from across all checkpoint files are merged into a single state_dict.
        We preserve the "state_dict key" <-> "checkpoint file" mapping in weight_map so we can
        write the state dict correctly in ``save_checkpoint``.

        Before returning, the model state dict is converted to a torchtune-compatible format using
        the appropriate convert_weights function (depending on ``self._model_type``).

        Returns:
            state_dict (dict[str, Any]): torchtune checkpoint state dict

        Raises:
            ValueError: If the values in the input state_dict are not Tensors
        """
        if self._enable_dcp:
            from torch.distributed.checkpoint import HuggingFaceStorageReader

            # DCP load using the storage reader
            hf_storage_reader = HuggingFaceStorageReader(path=self._checkpoint_dir)

            # TODO: reading the metadata isn't the best way to do this because
            # DCP can change their metadata structure and we've already read in
            # the metadata when doing _load_state_dict_from_keys
            metadata = hf_storage_reader.read_metadata()
            self._weight_map = {
                key.fqn: os.path.basename(val.relative_path)
                for key, val in metadata.storage_data.items()
            }

            state_dict = _load_state_dict_from_keys(storage_reader=hf_storage_reader)

            merged_state_dict = state_dict
        else:
            merged_state_dict = {}
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
                    self._weight_map[key] = f"{cpt_idx + 1:04}"
                merged_state_dict.update(state_dict)

                # delete the state_dict to free up memory; TODO check if this del is needed
                del state_dict
                gc.collect()

        # converted_state_dict is the final state_dict passed to the recipe after the
        # keys are converted into the torchtune format. This optionally also contains
        # the recipe state and adapter weights
        converted_state_dict: dict[str, dict[str, torch.Tensor]] = {}

        if self._model_type in (ModelType.PHI3_MINI, ModelType.PHI4):
            log_rank_zero(
                logger=logger,
                msg="Converting Phi weights from HF format."
                "Note that conversion of adapter weights into PEFT format is not supported.",
            )
            from torchtune.models.phi3._convert_weights import phi3_hf_to_tune

            num_heads = self._config["num_attention_heads"]
            num_kv_heads = self._config["num_key_value_heads"]
            dim = self._config["hidden_size"]

            # Should only pass num_heads, num_kv_heads, dim for GQA
            if num_heads == num_kv_heads:
                num_heads, num_kv_heads, dim = None, None, None

            converted_state_dict[training.MODEL_KEY] = phi3_hf_to_tune(
                merged_state_dict,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                dim=dim,
            )
        elif self._model_type == ModelType.REWARD:
            from torchtune.rlhf.utils import reward_hf_to_tune

            converted_state_dict[training.MODEL_KEY] = reward_hf_to_tune(
                merged_state_dict,
                num_heads=self._config["num_attention_heads"],
                num_kv_heads=self._config["num_key_value_heads"],
                dim=self._config["hidden_size"],
            )
        elif self._model_type == ModelType.QWEN2:
            from torchtune.models.qwen2._convert_weights import qwen2_hf_to_tune

            converted_state_dict[training.MODEL_KEY] = qwen2_hf_to_tune(
                merged_state_dict,
                num_heads=self._config["num_attention_heads"],
                num_kv_heads=self._config["num_key_value_heads"],
                dim=self._config["hidden_size"],
                tie_word_embeddings=self._config["tie_word_embeddings"],
            )
        elif self._model_type == ModelType.QWEN3:
            from torchtune.models.qwen3._convert_weights import qwen3_hf_to_tune

            converted_state_dict[training.MODEL_KEY] = qwen3_hf_to_tune(
                merged_state_dict,
                num_heads=self._config["num_attention_heads"],
                num_kv_heads=self._config["num_key_value_heads"],
                dim=self._config["hidden_size"],
                tie_word_embeddings=self._config["tie_word_embeddings"],
            )
        elif self._model_type == ModelType.LLAMA3_VISION:
            from torchtune.models.llama3_2_vision._convert_weights import (
                llama3_vision_hf_to_tune,
            )

            text_config = self._config.get("text_config", {})
            vision_config = self._config.get("vision_config", {})
            converted_state_dict[training.MODEL_KEY] = llama3_vision_hf_to_tune(
                merged_state_dict,
                num_heads=text_config["num_attention_heads"],
                num_kv_heads=text_config["num_key_value_heads"],
                dim=text_config["hidden_size"],
                head_dim=text_config.get("head_dim", None),
                vocab_size=text_config["vocab_size"],
                cross_attention_layers=text_config.get("cross_attention_layers", None),
                encoder_dim=vision_config["hidden_size"],
                tile_size=vision_config["image_size"],
                num_tiles=vision_config["max_num_tiles"],
                supported_aspect_ratios=vision_config.get(
                    "supported_aspect_ratios", None
                ),
            )
        elif self._model_type == ModelType.CLIP_TEXT:
            from torchtune.models.clip._convert_weights import clip_text_hf_to_tune

            converted_state_dict[training.MODEL_KEY] = clip_text_hf_to_tune(
                merged_state_dict,
            )
        elif self._model_type == ModelType.GEMMA2:
            from torchtune.models.gemma2._convert_weights import gemma2_hf_to_tune

            converted_state_dict[training.MODEL_KEY] = gemma2_hf_to_tune(
                merged_state_dict,
                num_heads=self._config["num_attention_heads"],
                num_kv_heads=self._config["num_key_value_heads"],
                dim=self._config["hidden_size"],
                head_dim=self._config.get("head_dim", None),
            )
        elif self._model_type == ModelType.GEMMA3:
            from torchtune.models.gemma3._convert_weights import gemma3_hf_to_tune

            # Here it became little bit tricky:
            # For 1B we have only 1 modality and specific config.json.
            # For 4B, 12B, 27B we have 2 modailites and same config.json

            if "text_config" in self._config:
                converted_state_dict[training.MODEL_KEY] = gemma3_hf_to_tune(
                    merged_state_dict,
                    num_heads=self._config["text_config"].get("num_attention_heads", 8),
                    num_kv_heads=self._config["text_config"].get("num_key_value_heads", 4),
                    dim=self._config["text_config"].get("hidden_size", 2560),
                    head_dim=self._config["text_config"].get("head_dim", 256),
                )
            else:
                # We are in 1B model!
                converted_state_dict[training.MODEL_KEY] = gemma3_hf_to_tune(
                    merged_state_dict,
                    num_heads=self._config["num_attention_heads"],
                    num_kv_heads=self._config["num_key_value_heads"],
                    dim=self._config["hidden_size"],
                    head_dim=self._config.get("head_dim", None),
                )

        elif self._model_type == ModelType.T5_ENCODER:
            from torchtune.models.t5._convert_weights import t5_encoder_hf_to_tune

            converted_state_dict[training.MODEL_KEY] = t5_encoder_hf_to_tune(
                merged_state_dict,
            )
        elif self._model_type == ModelType.LLAMA4:
            from torchtune.models.llama4._convert_weights import llama4_hf_to_tune

            converted_state_dict[training.MODEL_KEY] = llama4_hf_to_tune(
                merged_state_dict,
            )
        else:
            converted_state_dict[training.MODEL_KEY] = convert_weights.hf_to_tune(
                merged_state_dict,
                num_heads=self._config["num_attention_heads"],
                num_kv_heads=self._config["num_key_value_heads"],
                dim=self._config["hidden_size"],
                head_dim=self._config.get("head_dim", None),
            )

        if self._should_load_recipe_state:
            if os.path.exists(self._adapter_checkpoint):
                adapter_state_dict = safe_torch_load(self._adapter_checkpoint)
                converted_state_dict[training.ADAPTER_KEY] = adapter_state_dict
            recipe_state = safe_torch_load(self._recipe_checkpoint, mmap=False)
            converted_state_dict.update(recipe_state)
            logger.info(
                "Loading the recipe state using: "
                f"\n\tcheckpoint_paths: {[str(path) for path in self._checkpoint_paths]}"
                f"\n\trecipe_checkpoint: {self._recipe_checkpoint}"
                f"\n\tadapter_checkpoint: {self._adapter_checkpoint}"
            )

        return converted_state_dict

    def save_checkpoint(
        self,
        state_dict: dict[str, Any],
        epoch: int,
        intermediate_checkpoint: bool = False,
        adapter_only: bool = False,
        *,
        step: Optional[int] = None,
        max_shard_size: str = "5GB",
        dir_prefix: str = "epoch",
    ) -> None:
        """
        Save HF checkpoint to file. If ``intermediate_checkpoint`` is True, an additional
        checkpoint file ``recipe_state.pt`` is created in ``_output_dir/RECIPE_STATE_DIRNAME``
        which contains the recipe state.

        The state_dict is first converted back to the HF format and then partitioned based on the
        ``_weight_map`` into separate checkpoint files.

        Args:
            state_dict (dict[str, Any]): Checkpoint state dict to be written out to file
            epoch (int): Epoch number. Used to create the checkpoint file name
            intermediate_checkpoint (bool): If True, an additional checkpoint files for recipe state
                and (if applicable) adapter weights are created. Default is False
            adapter_only (bool): If True, only save the adapter weights. Default is False
            step (Optional[int]): Step number. Used to create the checkpoint file name if ``dir_prefix`` is 'step'.
            max_shard_size (str): Maximum shard size for the checkpoint files. Default is '5GB'.
            dir_prefix (str): Prefix for the checkpoint directory. Default is 'epoch'.

        Raises:
            ValueError:
                If ``adapter_only`` is True and adapter checkpoint not found in state_dict.
                If ``output_dir`` is not specified.
                If ``dir_prefix`` is 'step' but ``step`` is None.
                If ``dir_prefix`` is not 'epoch' or 'step'.
        """
        if dir_prefix == "epoch":
            ckpt_save_dirname = f"epoch_{epoch}"
            ckpt_pattern = r"^epoch_(\d+)"
        elif dir_prefix == "step":
            if step is None:
                raise ValueError(
                    "Step number must be provided when dir_prefix is 'step'."
                )
            ckpt_save_dirname = f"step_{step}"
            ckpt_pattern = r"^step_(\d+)"
        else:
            raise ValueError(
                f"Invalid dir_prefix: {dir_prefix}. Expected 'epoch' or 'step'."
            )

        if self._output_dir is None:
            raise ValueError(
                "Output directory not specified. Please specify an output directory to save the checkpoint."
            )
        ckpt_output_dir = os.path.join(self._output_dir, ckpt_save_dirname)
        self._output_fs.mkdirs(ckpt_output_dir, exist_ok=True)

        # 1. Convert the model weights back to transformer format inplace
        if not adapter_only:
            if self._model_type in (ModelType.PHI3_MINI, ModelType.PHI4):
                from torchtune.models.phi3._convert_weights import phi3_tune_to_hf

                state_dict[training.MODEL_KEY] = phi3_tune_to_hf(
                    state_dict[training.MODEL_KEY]
                )
            elif self._model_type == ModelType.REWARD:
                from torchtune.rlhf.utils import reward_tune_to_hf

                state_dict[training.MODEL_KEY] = reward_tune_to_hf(
                    state_dict[training.MODEL_KEY],
                    num_heads=self._config["num_attention_heads"],
                    num_kv_heads=self._config["num_key_value_heads"],
                    dim=self._config["hidden_size"],
                )
            elif self._model_type == ModelType.QWEN2:
                from torchtune.models.qwen2._convert_weights import qwen2_tune_to_hf

                state_dict[training.MODEL_KEY] = qwen2_tune_to_hf(
                    state_dict[training.MODEL_KEY],
                    num_heads=self._config["num_attention_heads"],
                    num_kv_heads=self._config["num_key_value_heads"],
                    dim=self._config["hidden_size"],
                    tie_word_embeddings=self._config["tie_word_embeddings"],
                )
            elif self._model_type == ModelType.QWEN3:
                from torchtune.models.qwen3._convert_weights import qwen3_tune_to_hf

                state_dict[training.MODEL_KEY] = qwen3_tune_to_hf(
                    state_dict[training.MODEL_KEY],
                    num_heads=self._config["num_attention_heads"],
                    num_kv_heads=self._config["num_key_value_heads"],
                    dim=self._config["hidden_size"],
                    tie_word_embeddings=self._config["tie_word_embeddings"],
                )
            elif self._model_type == ModelType.LLAMA3_VISION:
                from torchtune.models.llama3_2_vision._convert_weights import (
                    llama3_vision_tune_to_hf,
                )

                text_config = self._config.get("text_config", {})
                vision_config = self._config.get("vision_config", {})
                state_dict[training.MODEL_KEY] = llama3_vision_tune_to_hf(
                    state_dict[training.MODEL_KEY],
                    num_heads=text_config["num_attention_heads"],
                    num_kv_heads=text_config["num_key_value_heads"],
                    dim=text_config["hidden_size"],
                    head_dim=text_config.get("head_dim", None),
                    vocab_size=text_config["vocab_size"],
                    cross_attention_layers=text_config.get(
                        "cross_attention_layers", None
                    ),
                    encoder_dim=vision_config["hidden_size"],
                    tile_size=vision_config["image_size"],
                    num_tiles=vision_config["max_num_tiles"],
                    supported_aspect_ratios=vision_config.get(
                        "supported_aspect_ratios", None
                    ),
                )
            elif self._model_type == ModelType.GEMMA2:
                from torchtune.models.gemma2._convert_weights import gemma2_tune_to_hf

                state_dict[training.MODEL_KEY] = gemma2_tune_to_hf(
                    state_dict[training.MODEL_KEY],
                    num_heads=self._config["num_attention_heads"],
                    num_kv_heads=self._config["num_key_value_heads"],
                    dim=self._config["hidden_size"],
                    head_dim=self._config.get("head_dim", None),
                )
            elif self._model_type == ModelType.LLAMA4:
                from torchtune.models.llama4._convert_weights import llama4_tune_to_hf

                state_dict[training.MODEL_KEY] = llama4_tune_to_hf(
                    state_dict[training.MODEL_KEY],
                )
            else:
                state_dict[training.MODEL_KEY] = convert_weights.tune_to_hf(
                    state_dict[training.MODEL_KEY],
                    num_heads=self._config["num_attention_heads"],
                    num_kv_heads=self._config["num_key_value_heads"],
                    dim=self._config["hidden_size"],
                    head_dim=self._config.get("head_dim", None),
                )

            # Here we actually save the model weights
            if self._enable_dcp:
                from torch.distributed.checkpoint import HuggingFaceStorageWriter

                # DCP save using the storage writer
                fqn_to_file_index_mapping = {}
                for fqn, filename in self._weight_map.items():
                    index = int(filename.split("-")[1])
                    fqn_to_file_index_mapping[fqn] = index
                storage_writer = HuggingFaceStorageWriter(
                    path=os.path.join(self._output_dir, ckpt_save_dirname),
                    fqn_to_index_mapping=fqn_to_file_index_mapping,
                )
                save(
                    state_dict=state_dict[training.MODEL_KEY],
                    storage_writer=storage_writer,
                    no_dist=True,
                )
            else:
                from huggingface_hub.serialization import save_torch_state_dict

                save_torch_state_dict(
                    state_dict[training.MODEL_KEY],
                    ckpt_output_dir,
                    safe_serialization=self._safe_serialization,
                    max_shard_size=max_shard_size,
                )
                logger.info(f"Model checkpoint saved to {ckpt_output_dir}")

        if training.ADAPTER_KEY in state_dict:
            # TODO: saving it "as is" is a requirement because, if we only save with
            # convert_weights.tune_to_peft_adapter_weights, we do NOT have a fn
            # convert_weights.peft_to_tune. The .pt format is not needed, but
            # it is an easy way to distinguish the adapters. Ideally we should save only one.
            output_path = os.path.join(ckpt_output_dir, ADAPTER_MODEL_FNAME) + ".pt"
            with self._output_fs.open(output_path, "wb") as f:
                torch.save(state_dict[training.ADAPTER_KEY], f)
            logger.info(
                "Adapter checkpoint of size "
                f"{self._output_fs.size(output_path) / 1024**3:.2f} GiB "
                f"saved to {output_path}"
            )
            logger.info(
                "Please note that you have set adapter_only=True, so only adapter weights will be saved."
                "You need to merge the adapter weights into your base model for further use. "
                f"See {self.__class__.__name__}.save_checkpoint for more details."
            )

            if self._model_type in (ModelType.PHI3_MINI, ModelType.PHI4):
                logger.warning(
                    "Saving Phi adapter weights to PEFT format is not supported, saving to torchtune format instead"
                )
            elif self._model_type == ModelType.LLAMA3_VISION:
                logger.warning(
                    "Saving Llama3.2 Vision adapter weights to PEFT format is not supported, saving to torchtune format instead"
                )
            elif self._model_type == ModelType.LLAMA4:
                logger.warning(
                    "Saving Llama4 adapter weights to PEFT format is not supported, saving to torchtune format instead"
                )
            else:
                config = (
                    self._config["text_config"]
                    if "text_config" in self._config
                    else self._config
                )
                state_dict[
                    training.ADAPTER_KEY
                ] = convert_weights.tune_to_peft_adapter_weights(
                    state_dict[training.ADAPTER_KEY],
                    num_heads=config["num_attention_heads"],
                    num_kv_heads=config["num_key_value_heads"],
                    dim=config["hidden_size"],
                    head_dim=config.get("head_dim", None),
                )
                output_path = os.path.join(ckpt_output_dir, ADAPTER_MODEL_FNAME)
                self._output_fs.mkdirs(output_path, exist_ok=True)
                if not self._safe_serialization:
                    output_path = output_path + ".bin"
                    with self._output_fs.open(output_path, "wb") as f:
                        torch.save(state_dict[training.ADAPTER_KEY], f)
                else:
                    output_path = output_path + ".safetensors"
                    with self._output_fs.open(output_path, "wb") as f:
                        save_bytes = save_safetensors(
                            state_dict[training.ADAPTER_KEY],
                            metadata={"format": "pt"},
                        )
                        f.write(save_bytes)
                logger.info(
                    "Adapter checkpoint of size "
                    f"{self._output_fs.size(output_path) / 1024**3:.2f} GiB "
                    f"saved to {output_path}"
                )
        elif adapter_only:
            raise ValueError(
                "Adapter checkpoint not found in state_dict. Please ensure that the state_dict contains adapter weights."
            )

        if training.ADAPTER_CONFIG in state_dict:
            if self._model_type in (ModelType.PHI3_MINI, ModelType.PHI4):
                logger.warning(
                    "PEFT integration for Phi is not supported, skipping adapter config save"
                )
            elif self._model_type == ModelType.LLAMA3_VISION:
                logger.warning(
                    "PEFT integration for Llama3.2 Vision is not supported, skipping adapter config save"
                )
            else:
                state_dict[
                    training.ADAPTER_CONFIG
                ] = convert_weights.tune_to_peft_adapter_config(
                    adapter_config=state_dict[training.ADAPTER_CONFIG],
                    base_model_name_or_path=self.repo_id,
                )
                output_path = (
                    os.path.join(ckpt_output_dir, ADAPTER_CONFIG_FNAME) + ".json"
                )
                with self._output_fs.open(output_path, "w") as f:
                    json.dump(state_dict[training.ADAPTER_CONFIG], f)
                logger.info(
                    "Adapter checkpoint of size "
                    f"{self._output_fs.size(output_path) / 1024**3:.2f} GiB "
                    f"saved to {output_path}"
                )

        # 2. Save all files in ckpt_dir, except model weights and mapping, to output_dir/epoch_{epoch}
        # or step_{step} directory so it's easy to run inference with the model using this checkpoint
        copy_files(
            input_dir=self._checkpoint_dir,
            output_dir=ckpt_output_dir,
            ignore_suffixes=SUFFIXES_TO_NOT_COPY,
        )

        # 3. Save the recipe state, excluding the model weights and (if applicable) adapter weights
        if intermediate_checkpoint:
            for key in [
                training.MODEL_KEY,
                training.ADAPTER_KEY,
                training.ADAPTER_CONFIG,
            ]:
                state_dict.pop(key, None)
            recipe_state_path = os.path.join(ckpt_output_dir, "recipe_state.pt")
            with self._output_fs.open(recipe_state_path, "wb") as f:
                torch.save(state_dict, f)
            logger.info(
                f"Recipe checkpoint of size {os.path.getsize(recipe_state_path) / 1024**3:.2f} GiB "
                f"saved to {recipe_state_path}"
            )

        # 4. If specified, prune the checkpoints in the output directory
        if self._keep_last_n_checkpoints is not None:
            all_current_checkpoints = get_all_checkpoints_in_dir(
                Path(self._output_dir), pattern=ckpt_pattern
            )
            prune_surplus_checkpoints(
                all_current_checkpoints,
                keep_last_n_checkpoints=self._keep_last_n_checkpoints,
            )


class FullModelMetaCheckpointer(_CheckpointerInterface):
    """
    Checkpointer which reads and writes checkpoints in Meta's format. Examples include
    the Llama-2-7b model from the meta-llama repo (https://huggingface.co/meta-llama/Llama-2-7b)

    Currently we support reading from a single checkpoint file only. Support for reading from
    sharded checkpoints is WIP.

    Args:
        checkpoint_dir (str): Directory containing the checkpoint files
        checkpoint_files (list[str]): list of checkpoint files to load. Currently this checkpointer only
            supports loading a single checkpoint file.
        model_type (str): Model type of the model for which the checkpointer is being loaded, e.g. LLAMA3.
        output_dir (str): Directory to save the checkpoint files
        adapter_checkpoint (Optional[str]): Path to the adapter weights. If None,
            and `should_load_recipe_state=True`, then look for adapter_model.pt in output_dir/epoch_{largest_epoch}.
            Default is None.
        recipe_checkpoint (Optional[str]): Path to the recipe state checkpoint file. If None,
            and `should_load_recipe_state=True`, then look for recipe_state.pt in output_dir/recipe_state.
            Default is None.
        resume_from_checkpoint (bool): If True, the checkpointer will load the additional checkpoint files corresponding to
                the recipe state from a previous run. Default is False. This flag is deprecated. Please use the
                should_load_recipe_state instead.
        should_load_recipe_state (bool): If True, the checkpointer will load the additional checkpoint files corresponding to
                the recipe state from a previous run. Default is False

    Raises:
        ValueError: If ``checkpoint_files`` is not a list of length 1
    """

    def __init__(
        self,
        checkpoint_dir: str,
        checkpoint_files: list[str],
        model_type: str,
        output_dir: str,
        adapter_checkpoint: Optional[str] = None,
        recipe_checkpoint: Optional[str] = None,
        resume_from_checkpoint: bool = False,
        should_load_recipe_state: bool = False,
    ) -> None:
        # Fail fast if ``checkpoint_files`` is invalid
        # TODO: support loading more than one file
        if len(checkpoint_files) != 1:
            raise ValueError(
                "Currently we only support reading from a single checkpoint file. "
                f"Got {len(checkpoint_files)} files instead."
            )

        self._checkpoint_dir = Path(checkpoint_dir)
        self._should_load_recipe_state = should_load_recipe_state
        if resume_from_checkpoint:
            self._should_load_recipe_state = resume_from_checkpoint
            logger.warning(
                "*resume_from_checkpoint is deprecated. Please use the 'should_load_recipe_state' instead"
            )
        self._model_type = ModelType[model_type]
        self._output_dir = Path(output_dir)
        check_outdir_not_in_ckptdir(
            ckpt_dir=self._checkpoint_dir, out_dir=self._output_dir
        )
        self._output_dir.mkdir(parents=True, exist_ok=True)

        #  resume from adapter_model ckpt
        self._adapter_checkpoint = get_adapter_checkpoint_path(
            output_dir=self._output_dir,
            adapter_checkpoint=adapter_checkpoint,
            should_load_recipe_state=self._should_load_recipe_state,
            pattern=r"^epoch_(\d+)",
        )

        # resume recipe_state ckpt
        self._recipe_checkpoint = get_recipe_checkpoint_path(
            output_dir=self._output_dir,
            checkpoint_dir=self._checkpoint_dir,
            recipe_checkpoint=recipe_checkpoint,
            should_load_recipe_state=self._should_load_recipe_state,
        )

        # get ckpt paths
        self._checkpoint_paths = get_model_checkpoint_path(
            checkpoint_files=checkpoint_files,
            checkpoint_dir=self._checkpoint_dir,
            output_dir=self._output_dir,
            should_load_recipe_state=self._should_load_recipe_state,
            has_adapter_checkpoint=self._adapter_checkpoint is not None,
        )

        # we currently accept only a single file
        self._checkpoint_path = self._checkpoint_paths[0]

        if self._should_load_recipe_state:
            logger.info(
                "Loading the recipe state using: "
                f"\n\tcheckpoint_paths: {[str(path) for path in self._checkpoint_paths]}"
                f"\n\trecipe_checkpoint: {self._recipe_checkpoint}"
                f"\n\tadapter_checkpoint: {self._adapter_checkpoint}"
            )

    def load_checkpoint(self) -> dict[str, Any]:
        """
        Load Meta checkpoint from file. Currently only loading from a single file is supported.
        """
        state_dict: dict[str:Any] = {}
        model_state_dict = safe_torch_load(self._checkpoint_path)
        if self._model_type == ModelType.LLAMA3_VISION:
            from torchtune.models.llama3_2_vision._convert_weights import (
                llama3_vision_meta_to_tune,
            )

            state_dict[training.MODEL_KEY] = llama3_vision_meta_to_tune(
                model_state_dict
            )
        elif self._model_type == ModelType.LLAMA4:
            from torchtune.models.llama4._convert_weights import llama4_meta_to_tune

            state_dict[training.MODEL_KEY] = llama4_meta_to_tune(model_state_dict)
        else:
            state_dict[training.MODEL_KEY] = convert_weights.meta_to_tune(
                model_state_dict
            )

        # llama3_2 has tied weights, so we need to remove the output.weight key
        if self._model_type == ModelType.LLAMA3_2:
            logger.info(
                "Identified model_type = Llama3_2. Ignoring output.weight in"
                " checkpoint in favor of the tok_embedding.weight"
                " tied weights."
            )
            state_dict[training.MODEL_KEY].pop("output.weight")

        if self._adapter_checkpoint:
            adapter_state_dict = safe_torch_load(self._adapter_checkpoint)
            state_dict[training.ADAPTER_KEY] = adapter_state_dict

        if self._should_load_recipe_state:
            recipe_state = safe_torch_load(self._recipe_checkpoint, mmap=False)
            state_dict.update(recipe_state)
        return state_dict

    def save_checkpoint(
        self,
        state_dict: dict[str, Any],
        epoch: int,
        intermediate_checkpoint: bool = False,
        adapter_only: bool = False,
        **kwargs,
    ) -> None:
        """
        Save Meta checkpoint to file. If ``intermediate_checkpoint`` is True, an additional
        checkpoint file ``recipe_state.pt`` is created in ``_output_dir/RECIPE_STATE_DIRNAME``
        which contains the recipe state.

        Args:
            state_dict (dict[str, Any]): Checkpoint state dict to be written out to file
            epoch (int): Epoch number. Used to create the checkpoint file name
            intermediate_checkpoint (bool): If True, an additional checkpoint files for recipe state
                and (if applicable) adapter weights are created. Default is False
            adapter_only (bool): If True, only save the adapter weights. Default is False
            **kwargs: Ignored keyword arguments to maintain compatibility with the Checkpointer interface

        Raises:
            ValueError: if ``adapter_only`` is True and adapter checkpoint not found in state_dict.
        """
        ckpt_save_dirname = f"epoch_{epoch}"

        if not adapter_only:
            model_state_dict = state_dict[training.MODEL_KEY]
            if self._model_type == ModelType.LLAMA3_VISION:
                from torchtune.models.llama3_2_vision._convert_weights import (
                    llama3_vision_tune_to_meta,
                )

                state_dict[training.MODEL_KEY] = llama3_vision_tune_to_meta(
                    model_state_dict
                )
            elif self._model_type == ModelType.LLAMA4:
                from torchtune.models.llama4._convert_weights import llama4_tune_to_meta

                state_dict[training.MODEL_KEY] = llama4_tune_to_meta(model_state_dict)
            else:
                # llama3_2 has tied weights, so we need to add the output.weight key
                if (
                    self._model_type == ModelType.LLAMA3_2
                    and "output.weight" not in model_state_dict
                ):
                    model_state_dict["output.weight"] = model_state_dict[
                        "tok_embeddings.weight"
                    ]

                state_dict[training.MODEL_KEY] = convert_weights.tune_to_meta(
                    model_state_dict
                )

            # TODO: We should consider adding adapter/model config, like we do for HF.
            model_filename = SHARD_FNAME.format(
                cpt_idx="1".zfill(5), num_shards="1".zfill(5)
            )
            checkpoint_file = Path.joinpath(
                self._output_dir, ckpt_save_dirname, model_filename
            ).with_suffix(".bin")
            checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

            torch.save(state_dict[training.MODEL_KEY], checkpoint_file)
            logger.info(
                "Model checkpoint of size "
                f"{os.path.getsize(checkpoint_file) / 1024**3:.2f} GiB "
                f"saved to {checkpoint_file}"
            )

        if training.ADAPTER_KEY in state_dict:
            output_path = Path.joinpath(
                self._output_dir, ckpt_save_dirname, ADAPTER_MODEL_FNAME
            ).with_suffix(".pt")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(state_dict[training.ADAPTER_KEY], output_path)
            logger.info(
                "Adapter checkpoint of size "
                f"{os.path.getsize(output_path) / 1024**3:.2f} GiB "
                f"saved to {output_path}"
            )

        elif adapter_only:
            raise ValueError(
                "Adapter checkpoint not found in state_dict. Please ensure that the state_dict contains adapter weights."
            )

        # Save all files in ckpt_dir, except model weights and mapping, to output_dir/epoch_{epoch}
        # So its easy to run inference with the model using this epoch's checkpoint
        copy_files(
            self._checkpoint_dir,
            Path.joinpath(self._output_dir, ckpt_save_dirname),
            ignore_suffixes=SUFFIXES_TO_NOT_COPY,
        )

        # If the recipe state needs to be output, first remove the model state dict
        # and if it exists, remove the adapter state dict as well
        if intermediate_checkpoint:
            _ = state_dict.pop(training.MODEL_KEY, None)
            _ = state_dict.pop(training.ADAPTER_KEY, None)
            _ = state_dict.pop(training.ADAPTER_CONFIG, None)
            output_path = Path.joinpath(
                self._output_dir, RECIPE_STATE_DIRNAME, "recipe_state.pt"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(state_dict, output_path)
            logger.info(
                "Recipe checkpoint of size "
                f"{os.path.getsize(output_path) / 1024**3:.2f} GiB "
                f"saved to {output_path}"
            )
        else:
            logger.info("Saving final epoch checkpoint.")
            if adapter_only:
                logger.info(
                    "Please note that you have set adapter_only=True, so only adapter weights will be saved."
                    "You need to merge the adapter weights into your base model for further use. "
                    f"See {self.__class__.__name__}.save_checkpoint for more details."
                )
            else:
                logger.info(
                    "The full model checkpoint, including all weights and configurations, has been saved successfully."
                    "You can now use this checkpoint for further training or inference."
                )


class DistributedCheckpointer(_CheckpointerInterface):
    """
    Checkpointer which reads and writes checkpoints in the DistributedCheckpointing format.

    Args:
        checkpoint_dir (str): Directory containing the checkpoint files
        output_dir (str): Directory to save the checkpoint files
        process_group (Optional[dist.ProcessGroup]): Optional process group to use
            for distributed saving/loading. If None, the default process group will be used.
            For checkpointing, gloo CPU-based backend is needed.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        output_dir: str,
        process_group: Optional[dist.ProcessGroup] = None,
    ) -> None:
        self._checkpoint_dir = Path(checkpoint_dir)
        self._output_dir = Path(output_dir)
        self._checkpoint_future = None
        self._metadata_file = ".metadata"
        self._adapter_dir = "adapter_model"
        _, self._rank = get_world_size_and_rank()
        self._process_group: Optional[dist.ProcessGroup] = process_group

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    def load_checkpoint(
        self,
        state_dict: dict[str, Any],
        adapter_only: bool = False,
    ) -> dict[str, Any]:
        """
        Load a Distributed checkpoint saved at the <checkpoint_path>
        If no path is provided, latest intermediate checkpoint is loaded.
        """
        checkpoint_path = get_most_recent_checkpoint(self.output_dir)
        if checkpoint_path is None:
            raise ValueError(
                "No intermediate checkpoint was found in the output directory."
                "Please ensure that a checkpoint exists to load."
            )

        if adapter_only:
            checkpoint_path = os.path.join(checkpoint_path, self._adapter_dir)

        log_rank_zero(logger, msg=f"Loading checkpoint from {checkpoint_path}")

        load(
            state_dict=state_dict,
            storage_reader=FileSystemReader(checkpoint_path),
            process_group=self._process_group,
            planner=DefaultLoadPlanner(
                allow_partial_load=True
            ),  # this is because we add initial_lr to the state dict, but not all recipes have it
        )

        return state_dict

    def save_checkpoint(
        self,
        state_dict: dict[str, Any],
        epoch: int,
        save_async: bool = False,
        adapter_only: bool = False,
        *,
        step: Optional[int] = None,
        dir_prefix: str = "epoch",
    ) -> None:
        """
        Save a distributed checkpoint to storage.
        If ``save_async`` is True, the save happens asynchronously unblocking the GPUs sooner. This
        should only be used for the intermediate checkpoints. Final checkpoint has to be a synchronous
        one as the finetuning job can not terminate until the checkpoint gets persisted.

        Args:
            state_dict (dict[str, Any]): Checkpoint state dict to be written out to file
            epoch (int): Epoch number. Used to create the checkpoint file name
            save_async (bool): If True, save the checkpoint asynchronously
            adapter_only (bool): If True, only adapter weights are being saved, which affects which path to save to.
            step (Optional[int]): Step number. Used to create the checkpoint file name if provided.
            dir_prefix (str): Prefix to use for the checkpoint directory name. Defaults to "epoch".

        Raises:
            ValueError:
                If ``dir_prefix`` is 'step' and ``step`` is None.
                If ``dir_prefix`` is not one of 'epoch' or 'step'.
        """
        if dir_prefix == "epoch":
            ckpt_save_dirname = f"epoch_{epoch}"
        elif dir_prefix == "step":
            if step is None:
                raise ValueError("Step number is required when dir_prefix is 'step'.")
            ckpt_save_dirname = f"step_{step}"
        else:
            raise ValueError(
                f"Invalid dir_prefix: {dir_prefix}. Must be one of 'epoch' or 'step'."
            )

        checkpoint_path = os.path.join(self._output_dir, ckpt_save_dirname)
        if adapter_only:
            checkpoint_path = os.path.join(checkpoint_path, self._adapter_dir)

        if self._checkpoint_future and not self._checkpoint_future.done():
            # Previous checkpoint needs to finish before saving the next one.
            wait_start = time.perf_counter()

            logger.info(
                f"Rank {self._rank}: previous checkpoint has not finished. Checkpointing frequency is too high. Waiting...",
            )

            self._checkpoint_future.result()

            logger.info(
                f"Rank {self._rank}: waited {time.perf_counter() - wait_start:.2f} seconds for previous checkpoint to finish",
            )
            self._checkpoint_future = None

        cp_start = time.perf_counter()

        if save_async:

            def callback(
                f: Future,
            ) -> None:
                if f.exception() is None:
                    logger.info(
                        f"Rank {self._rank}: Checkpoint is saved asynchronously to {checkpoint_path} successfully.",
                    )
                else:
                    logger.error(
                        f"Rank {self._rank}: Checkpoint failed to save asynchronously to {checkpoint_path} "
                        f"with the exception {f.exception()}"
                    )

            self._checkpoint_future = async_save(
                state_dict=state_dict,
                storage_writer=FileSystemWriter(
                    checkpoint_path,
                    thread_count=16,
                    single_file_per_rank=False,
                    sync_files=False,
                ),
                process_group=self._process_group,
            )

            logger.info(
                f"Rank {self._rank}: Trainer was blocked for {time.perf_counter() - cp_start:.2f} seconds "
                "for checkpointing to finish...",
            )

            self._checkpoint_future.add_done_callback(callback)
        else:
            log_rank_zero(
                logger,
                msg=f"Saving model checkpoint synchronously to {checkpoint_path}.",
            )

            save(
                state_dict=state_dict,
                storage_writer=FileSystemWriter(
                    checkpoint_path,
                    thread_count=16,
                    single_file_per_rank=False,
                    sync_files=False,
                    cache_staged_state_dict=True,
                ),
                process_group=self._process_group,
            )

        # TODO: prune old checkpoints
