# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import gc
import json
import os
import re
import time
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union

import torch
import torch.distributed as dist
from safetensors.torch import save_file
from torch.distributed.checkpoint import (
    async_save,
    FileSystemReader,
    FileSystemWriter,
    load,
    save,
)

from torchtune import training
from torchtune.models import convert_weights
from torchtune.training.checkpointing._utils import (
    ADAPTER_CONFIG_FNAME,
    ADAPTER_MODEL_FNAME,
    check_outdir_not_in_ckptdir,
    copy_files,
    get_adapter_checkpoint_path,
    get_model_checkpoint_path,
    get_recipe_checkpoint_path,
    ModelType,
    RECIPE_STATE_DIRNAME,
    REPO_ID_FNAME,
    safe_torch_load,
    SAFETENSOR_INDEX_FNAME,
    SHARD_FNAME,
    SUFFIXES_TO_NOT_COPY,
    TORCH_INDEX_FNAME,
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

    def load_checkpoint(self, **kwargs) -> Dict[str, Any]:
        ...

    def save_checkpoint(self, state_dict: Dict[str, Any], **kwargs) -> None:
        ...


class FullModelTorchTuneCheckpointer(_CheckpointerInterface):
    """
    Checkpointer which reads and writes checkpoints in a format compatible with
    torchtune. No conversion of weights is required.

    Currently this supports reading a single checkpoint file only. This will likely change as
    we add support for larger models.

    Args:
        checkpoint_dir (str): Directory containing the checkpoint files
        checkpoint_files (List[str]): List of checkpoint files to load. Since the checkpointer takes care
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
        checkpoint_files: List[str],
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

    def load_checkpoint(self, weights_only: bool = True) -> Dict[str, Any]:
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
            Dict[str, Any]: state_dict from the input checkpoint
        """
        state_dict: Dict[str:Any] = {}
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
        state_dict: Dict[str, Any],
        epoch: int,
        intermediate_checkpoint: bool = False,
        adapter_only: bool = False,
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
            state_dict (Dict[str, Any]): State dict with model and (optionally) recipe state
            epoch (int): Current epoch number. This is added to the checkpoint file name to ensure
                we're not overwriting intermediate checkpoint files
            intermediate_checkpoint (bool): If True, save an additional checkpoint file with the
                recipe state
            adapter_only (bool): If True, only save the adapter weights. Default is False


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
        checkpoint_files (Union[List[str], Dict[str, str]]): List of checkpoint files to load or a dictionary
            containing the keys keys ["filename_format", "max_filename"]. Since the checkpointer takes care
            of sorting by file ID, the order in this list does not matter.
        model_type (str): Model type of the model for which the checkpointer is being loaded, e.g. LLAMA3.
        output_dir (str): Directory to save the checkpoint files
        adapter_checkpoint (Optional[str]): Path to the adapter weights. If None,
            and `should_load_recipe_state=True`, then look for adapter_model.pt in output_dir/epoch_{largest_epoch}.
            Default is None.
        recipe_checkpoint (Optional[str]): Path to the recipe state checkpoint file. If None,
            and `should_load_recipe_state=True`, then look for recipe_state.pt in output_dir/RECIPE_STATE_DIRNAME.
            Default is None.
        resume_from_checkpoint (bool): If True, the checkpointer will load the additional checkpoint files corresponding to
            the receipe state from a previous run. Default is False. This flag is deprecated. Please use
            the should_load_recipe_state flag instead.
        safe_serialization (bool): If True, the checkpointer will save the checkpoint file using `safetensors`.
            Default is True.
        should_load_recipe_state (bool): If True, the checkpointer will load the additional checkpoint files corresponding to
            the receipe state from a previous run. Default is False
    """

    def __init__(
        self,
        checkpoint_dir: str,
        checkpoint_files: Union[List[str], Dict[str, str]],
        model_type: str,
        output_dir: str,
        adapter_checkpoint: Optional[str] = None,
        recipe_checkpoint: Optional[str] = None,
        resume_from_checkpoint: bool = False,
        safe_serialization: bool = True,
        should_load_recipe_state: bool = False,
    ) -> None:

        self._should_load_recipe_state = should_load_recipe_state
        if resume_from_checkpoint:
            self._should_load_recipe_state = resume_from_checkpoint
            logger.warning(
                "*resume_from_checkpoint is deprecated. Please use the 'should_load_recipe_state' instead"
            )

        self._safe_serialization = safe_serialization
        self._checkpoint_dir = Path(checkpoint_dir)
        self._model_type = ModelType[model_type]
        self._output_dir = Path(output_dir)
        check_outdir_not_in_ckptdir(
            ckpt_dir=self._checkpoint_dir, out_dir=self._output_dir
        )
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # weight_map contains the state_dict key -> checkpoint file mapping so we can correctly
        # parition the state dict into output checkpoint files. This is updated during checkpoint
        # load
        self._weight_map: Dict[str, str] = None

        # the config.json file contains model params needed for state dict conversion
        self._config = json.loads(
            Path.joinpath(self._checkpoint_dir, "config.json").read_text()
        )

        # repo_id is necessary for when saving an adapter config, so its compatible with HF.
        # This json file is produced and saved in the download step.
        # contents are {"repo_id": "some_model/some_model_version"}
        repo_id_path = Path.joinpath(self._checkpoint_dir, REPO_ID_FNAME).with_suffix(
            ".json"
        )
        self.repo_id = None
        if repo_id_path.exists():
            with open(repo_id_path, "r") as json_file:
                data = json.load(json_file)
                self.repo_id = data.get("repo_id")

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

        if self._should_load_recipe_state:
            logger.info(
                "Loading the recipe state using: "
                f"\n\tcheckpoint_paths: {[str(path) for path in self._checkpoint_paths]}"
                f"\n\trecipe_checkpoint: {self._recipe_checkpoint}"
                f"\n\tadapter_checkpoint: {self._adapter_checkpoint}"
            )

    def load_checkpoint(self) -> Dict[str, Any]:
        """
        Load HF checkpoint from file.

        The keys and weights from across all checkpoint files are merged into a single state_dict.
        We preserve the "state_dict key" <-> "checkpoint file" mapping in weight_map so we can
        write the state dict correctly in ``save_checkpoint``.

        Before returning, the model state dict is converted to a torchtune-compatible format using
        the appropriate convert_weights function (depending on ``self._model_type``).

        Returns:
            state_dict (Dict[str, Any]): torchtune checkpoint state dict

        Raises:
            ValueError: If the values in the input state_dict are not Tensors
        """

        self._weight_map = {}

        # merged state_dict contains keys and weights from all the checkpoint files
        merged_state_dict: Dict[str, torch.Tensor] = {}

        # converted_state_dict is the final state_dict passed to the recipe after the
        # keys are converted into the torchtune format. This optionally also contains
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
                self._weight_map[key] = f"{cpt_idx + 1:04}"
            merged_state_dict.update(state_dict)

            # delete the state_dict to free up memory; TODO check if this del is needed
            del state_dict
            gc.collect()
        if self._model_type == ModelType.PHI3_MINI:
            log_rank_zero(
                logger=logger,
                msg="Converting Phi-3 Mini weights from HF format."
                "Note that conversion of adapter weights into PEFT format is not supported.",
            )
            from torchtune.models.phi3._convert_weights import phi3_hf_to_tune

            converted_state_dict[training.MODEL_KEY] = phi3_hf_to_tune(
                merged_state_dict
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
        elif self._model_type == ModelType.T5_ENCODER:
            from torchtune.models.t5._convert_weights import t5_encoder_hf_to_tune

            converted_state_dict[training.MODEL_KEY] = t5_encoder_hf_to_tune(
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

        if self._adapter_checkpoint:
            adapter_state_dict = safe_torch_load(self._adapter_checkpoint)
            converted_state_dict[training.ADAPTER_KEY] = adapter_state_dict

        if self._should_load_recipe_state:
            recipe_state = safe_torch_load(self._recipe_checkpoint, mmap=False)
            converted_state_dict.update(recipe_state)

        return converted_state_dict

    def save_checkpoint(
        self,
        state_dict: Dict[str, Any],
        epoch: int,
        intermediate_checkpoint: bool = False,
        adapter_only: bool = False,
    ) -> None:
        """
        Save HF checkpoint to file. If ``intermediate_checkpoint`` is True, an additional
        checkpoint file ``recipe_state.pt`` is created in ``_output_dir/RECIPE_STATE_DIRNAME``
        which contains the recipe state.

        The state_dict is first converted back to the HF format and then partitioned based on the
        ``_weight_map`` into separate checkpoint files.

        Args:
            state_dict (Dict[str, Any]): Checkpoint state dict to be written out to file
            epoch (int): Epoch number. Used to create the checkpoint file name
            intermediate_checkpoint (bool): If True, an additional checkpoint files for recipe state
                and (if applicable) adapter weights are created. Default is False
            adapter_only (bool): If True, only save the adapter weights. Default is False

        Raises:
            ValueError: if ``adapter_only`` is True and adapter checkpoint not found in state_dict.
        """
        # convert the state_dict back to hf format; do this inplace
        if not adapter_only:
            if self._model_type == ModelType.PHI3_MINI:
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
            else:
                state_dict[training.MODEL_KEY] = convert_weights.tune_to_hf(
                    state_dict[training.MODEL_KEY],
                    num_heads=self._config["num_attention_heads"],
                    num_kv_heads=self._config["num_key_value_heads"],
                    dim=self._config["hidden_size"],
                    head_dim=self._config.get("head_dim", None),
                )

            # split the state_dict into separate dicts, one for each output checkpoint file
            # e.g. split_state_dicts= {
            #       "0001": {"key1": tensor1, "key2": tensor2},
            #       "0002": {"key3": tensor3}
            #       }
            split_state_dicts: Dict[str, Dict[str, torch.Tensor]] = {}
            total_size = 0
            for key, weight in state_dict[training.MODEL_KEY].items():
                cpt_idx = self._weight_map[key]

                # initialize dict
                if cpt_idx not in split_state_dicts:
                    split_state_dicts[cpt_idx] = {}

                split_state_dicts[cpt_idx].update({key: weight})
                total_size += weight.numel() * weight.element_size()

            # write the partitioned state dicts to the right checkpoint file
            # e.g. model-00001-of-00004.safetensors, model-00002-of-00004.safetensors, etc
            num_shards = len(split_state_dicts)
            map_original_name_to_new_name = {}
            for cpt_idx, model_state_dict in split_state_dicts.items():
                # TODO: We should probably use the original shard name and just add a prefix
                # however, having the SHARD_FNAME standardizes our checkpoints
                shard_name = SHARD_FNAME.format(
                    cpt_idx=f"{cpt_idx}".zfill(5), num_shards=f"{num_shards}".zfill(5)
                )
                map_original_name_to_new_name[cpt_idx] = shard_name
                output_path = Path.joinpath(
                    self._output_dir, f"epoch_{epoch}", shard_name
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                if not self._safe_serialization:
                    output_path = output_path.with_suffix(".bin")
                    torch.save(model_state_dict, output_path)
                else:
                    output_path = output_path.with_suffix(".safetensors")
                    save_file(model_state_dict, output_path, metadata={"format": "pt"})

                logger.info(
                    "Model checkpoint of size "
                    f"{os.path.getsize(output_path) / 1024**3:.2f} GiB "
                    f"saved to {output_path}"
                )

            # Save the appropriate index file based on serialization format
            # e.g. {metadata: {total_size: 1234}, weight_map: {"key1": "model_0001.safetensors", "key2": "model_0002.safetensors"}}
            if self._safe_serialization:
                weight_map = {
                    k: map_original_name_to_new_name[cpt_idx] + ".safetensors"
                    for k, cpt_idx in self._weight_map.items()
                }
                index_file_name = SAFETENSOR_INDEX_FNAME
            else:
                weight_map = {
                    k: map_original_name_to_new_name[cpt_idx] + ".bin"
                    for k, cpt_idx in self._weight_map.items()
                }
                index_file_name = TORCH_INDEX_FNAME

            index_path = Path.joinpath(
                self._output_dir, f"epoch_{epoch}", index_file_name
            )

            index_data = {
                "metadata": {"total_size": total_size},
                "weight_map": weight_map,
            }
            with open(index_path, "w") as f:
                json.dump(index_data, f, indent=2)

        if training.ADAPTER_KEY in state_dict:

            # TODO: saving it "as is" is a requirement because, if we only save with
            # convert_weights.tune_to_peft_adapter_weights, we do NOT have a fn
            # convert_weights.peft_to_tune. The .pt format is not needed, but
            # it is an easy way to distinguish the adapters. Ideally we should save only one.
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

            if self._model_type == ModelType.PHI3_MINI:
                logger.warning(
                    "Saving Phi-3 Mini adapter weights to PEFT format is not supported, saving to torchtune format instead"
                )
            elif self._model_type == ModelType.LLAMA3_VISION:
                logger.warning(
                    "Saving Llama3.2 Vision adapter weights to PEFT format is not supported, saving to torchtune format instead"
                )
            else:
                state_dict[
                    training.ADAPTER_KEY
                ] = convert_weights.tune_to_peft_adapter_weights(
                    state_dict[training.ADAPTER_KEY],
                    num_heads=self._config["num_attention_heads"],
                    num_kv_heads=self._config["num_key_value_heads"],
                    dim=self._config["hidden_size"],
                    head_dim=self._config.get("head_dim", None),
                )
                output_path = Path.joinpath(
                    self._output_dir, f"epoch_{epoch}", ADAPTER_MODEL_FNAME
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                if not self._safe_serialization:
                    output_path = output_path.with_suffix(".bin")
                    torch.save(state_dict[training.ADAPTER_KEY], output_path)
                else:
                    output_path = output_path.with_suffix(".safetensors")
                    save_file(
                        state_dict[training.ADAPTER_KEY],
                        output_path,
                        metadata={"format": "pt"},
                    )
                logger.info(
                    "Adapter checkpoint of size "
                    f"{os.path.getsize(output_path) / 1024**3:.2f} GiB "
                    f"saved to {output_path}"
                )
        elif adapter_only:
            raise ValueError(
                "Adapter checkpoint not found in state_dict. Please ensure that the state_dict contains adapter weights."
            )

        if training.ADAPTER_CONFIG in state_dict:
            if self._model_type == ModelType.PHI3_MINI:
                logger.warning(
                    "PEFT integration for Phi-3 Mini is not supported, skipping adapter config save"
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

                output_path = Path.joinpath(
                    self._output_dir, f"epoch_{epoch}", ADAPTER_CONFIG_FNAME
                ).with_suffix(".json")
                with open(output_path, "w") as f:
                    json.dump(state_dict[training.ADAPTER_CONFIG], f)
                logger.info(
                    "Adapter checkpoint of size "
                    f"{os.path.getsize(output_path) / 1024**3:.2f} GiB "
                    f"saved to {output_path}"
                )

        # Save all files in ckpt_dir, except model weights and mapping, to output_dir/epoch_{epoch}
        # So its easy to run inference with the model using this epoch's checkpoint
        copy_files(
            self._checkpoint_dir,
            Path.joinpath(self._output_dir, f"epoch_{epoch}"),
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


class FullModelMetaCheckpointer(_CheckpointerInterface):
    """
    Checkpointer which reads and writes checkpoints in Meta's format. Examples include
    the Llama-2-7b model from the meta-llama repo (https://huggingface.co/meta-llama/Llama-2-7b)

    Currently we support reading from a single checkpoint file only. Support for reading from
    sharded checkpoints is WIP.

    Args:
        checkpoint_dir (str): Directory containing the checkpoint files
        checkpoint_files (List[str]): List of checkpoint files to load. Currently this checkpointer only
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
        checkpoint_files: List[str],
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

    def load_checkpoint(self) -> Dict[str, Any]:
        """
        Load Meta checkpoint from file. Currently only loading from a single file is supported.
        """
        state_dict: Dict[str:Any] = {}
        model_state_dict = safe_torch_load(self._checkpoint_path)
        if self._model_type == ModelType.LLAMA3_VISION:
            from torchtune.models.llama3_2_vision._convert_weights import (
                llama3_vision_meta_to_tune,
            )

            state_dict[training.MODEL_KEY] = llama3_vision_meta_to_tune(
                model_state_dict
            )
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
        state_dict: Dict[str, Any],
        epoch: int,
        intermediate_checkpoint: bool = False,
        adapter_only: bool = False,
    ) -> None:
        """
        Save Meta checkpoint to file. If ``intermediate_checkpoint`` is True, an additional
        checkpoint file ``recipe_state.pt`` is created in ``_output_dir/RECIPE_STATE_DIRNAME``
        which contains the recipe state.

        Args:
            state_dict (Dict[str, Any]): Checkpoint state dict to be written out to file
            epoch (int): Epoch number. Used to create the checkpoint file name
            intermediate_checkpoint (bool): If True, an additional checkpoint files for recipe state
                and (if applicable) adapter weights are created. Default is False
            adapter_only (bool): If True, only save the adapter weights. Default is False

        Raises:
            ValueError: if ``adapter_only`` is True and adapter checkpoint not found in state_dict.
        """

        if not adapter_only:
            model_state_dict = state_dict[training.MODEL_KEY]
            if self._model_type == ModelType.LLAMA3_VISION:
                from torchtune.models.llama3_2_vision._convert_weights import (
                    llama3_vision_tune_to_meta,
                )

                state_dict[training.MODEL_KEY] = llama3_vision_tune_to_meta(
                    model_state_dict
                )
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
                self._output_dir, f"epoch_{epoch}", model_filename
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
        self._checkpoint_dir_prefix = "dist_epoch"
        self._metadata_file = ".metadata"
        _, self._rank = get_world_size_and_rank()
        self._process_group: Optional[dist.ProcessGroup] = process_group

    def _get_latest_intermediate_checkpoint(self) -> Optional[str]:
        """
        This method iterates over the available intermediate distributed checkpoints and
        finds the latest checkpoint to load.

        Returns:
            str: The fully qualified path of the checkpoint directory containing the latest and valid
            intermediate checkpoint. A valid checkpoint needs to have the metadata file.
        """

        checkpoint_dir_pattern = re.compile(f"{self._checkpoint_dir_prefix}_(\\d+)")
        checkpoint_paths = [
            name
            for name in os.listdir(self._output_dir)
            if re.match(checkpoint_dir_pattern, name)
            and os.path.isfile(
                os.path.join(self._output_dir, name, self._metadata_file)
            )
        ]

        if checkpoint_paths:
            latest_checkpoint_dir = sorted(
                checkpoint_paths, key=lambda x: int(x.split("_")[-1])
            )[-1]
            return os.path.join(self._output_dir, latest_checkpoint_dir)
        return None

    def load_checkpoint(
        self, state_dict: Dict[str, Any] = None, checkpoint_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load a Distributed checkpoint saved at the <checkpoint_path>
        If no path is provided, latest intermediate checkpoint is loaded.
        """

        if state_dict is None:
            raise ValueError(
                "State dict must be provided to load a distributed checkpoint."
            )

        # If no checkpoint path is provided, load the latest intermediate checkpoint.
        if checkpoint_path is None:
            checkpoint_path = self._get_latest_intermediate_checkpoint()

            if checkpoint_path is None:
                raise ValueError(
                    "No checkpoint path was provided."
                    "Also, No intermediate checkpoint was found in the output directory."
                    "Please ensure that a checkpoint exists to load."
                )

        log_rank_zero(logger, msg=f"Loading checkpoint from {checkpoint_path}")

        load(
            state_dict=state_dict,
            storage_reader=FileSystemReader(checkpoint_path),
            process_group=self._process_group,
        )

        return state_dict

    def save_checkpoint(
        self,
        state_dict: Dict[str, Any],
        epoch: int,
        save_async: bool = False,
    ) -> None:
        """
        Save a distributed checkpoint to storage.
        If ``save_async`` is True, the save happens asynchronously unblocking the GPUs sooner. This
        should only be used for the intermediate checkpoints. Final checkpoint has to be a synchronous
        one as the finetuning job can not terminate until the checkpoint gets persisted.

        Args:
            state_dict (Dict[str, Any]): Checkpoint state dict to be written out to file
            epoch (int): Epoch number. Used to create the checkpoint file name
            save_async (bool): If True, save the checkpoint asynchronously
        """

        log_rank_zero(
            logger,
            msg=f"DistributedCheckpointer is saving a checkpoint for the epoch {epoch}",
        )

        checkpoint_path = Path.joinpath(
            self._output_dir, f"{self._checkpoint_dir_prefix}_{epoch}"
        )

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
                ),
                process_group=self._process_group,
            )

        log_rank_zero(
            logger,
            msg="The full model checkpoint, including all the weights and configurations, has been saved successfully "
            "by the DistributedCheckpointer. You can now use this checkpoint for further training.",
        )
