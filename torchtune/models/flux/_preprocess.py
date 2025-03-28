# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import Dataset

from torchtune.datasets._text_to_image import TextToImageDataset
from torchtune.models.clip import clip_text_vit_large_patch14
from torchtune.models.clip._convert_weights import clip_text_hf_to_tune
from torchtune.models.flux import flux_1_autoencoder
from torchtune.models.flux._convert_weights import flux_ae_hf_to_tune
from torchtune.models.flux._utils import get_t5_max_seq_len
from torchtune.models.t5 import t5_v1_1_xxl_encoder
from torchtune.models.t5._convert_weights import t5_encoder_hf_to_tune
from torchtune.training.checkpointing._utils import safe_torch_load


class FluxEncodingsDataset(Dataset):
    """
    Dataset of preprocessed text-to-image data for finetuning Flux models.

    This dataset is created by `FluxPreprocessor.preprocess_dataset`.

    Args:
        encodings_dir (Path): path to folder where the preprocessed data is stored.
    """

    def __init__(self, encodings_dir: Path):
        self._encodings_paths = list(encodings_dir.iterdir())

    def __len__(self) -> int:
        return len(self._encodings_paths)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return torch.load(self._encodings_paths[index], weights_only=True)


class FluxPreprocessor:
    """
    Preprocess text-to-image data for finetuning Flux models.

    Args:
        autoencoder (Module): Flux image autoencoder
        clip_encoder (Module): CLIP text encoder
        t5_encoder (Module): T5 text encoder
        preprocessed_data_dir (str): folder where the preprocessed data will be stored
        preprocess_again_if_exists (bool): if false, data samples that already have preprocessed data will be skippped
        batch_size (int): batch size to use when preprocessing datasets
        device (torch.device): device to do preprocessing on
        dtype (torch.dtype): data type to do preprocessing in
    """

    def __init__(
        self,
        autoencoder: Module,
        clip_encoder: Module,
        t5_encoder: Module,
        preprocessed_data_dir: str,
        preprocess_again_if_exists: bool,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.autoencoder = autoencoder.to(device=device, dtype=dtype).eval()
        self._clip_encoder = clip_encoder.to(device=device, dtype=dtype).eval()
        self._t5_encoder = t5_encoder.to(device=device, dtype=dtype).eval()
        self._preprocess_again_if_exists = preprocess_again_if_exists
        self._batch_size = batch_size
        self._device = device
        self._dtype = dtype

        self._dir = Path(preprocessed_data_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad
    def preprocess_dataset(self, ds: TextToImageDataset) -> FluxEncodingsDataset:
        """
        Preprocess a text-to-image dataset into a Flux encodings dataset.

        Args:
            ds (TextToImageDataset): the raw dataset

        Returns:
            FluxEncodingsDataset: the preprocessed dataset
        """
        dataloader = torch.utils.data.DataLoader(
            dataset=ds,
            batch_size=self._batch_size,
            shuffle=False,
            drop_last=False,
        )

        for batch in dataloader:
            ids = batch["id"]
            if not self._should_preprocess(ids):
                continue

            images = batch["image"].to(device=self._device, dtype=self._dtype)
            clip_tokens = batch["clip_tokens"].to(device=self._device)
            t5_tokens = batch["t5_tokens"].to(device=self._device)

            img_encodings = self.autoencoder.encode(images)
            clip_text_encodings = self._clip_encoder(clip_tokens)
            t5_text_encodings = self._t5_encoder(t5_tokens)

            for id, img_encoding, clip_text_encoding, t5_text_encoding in zip(
                ids, img_encodings, clip_text_encodings, t5_text_encodings
            ):
                torch.save(
                    {
                        "img_encoding": img_encoding,
                        "clip_text_encoding": clip_text_encoding,
                        "t5_text_encoding": t5_text_encoding,
                    },
                    self._dir / f"{id}.pt",
                )

        return FluxEncodingsDataset(self._dir)

    @torch.no_grad
    def preprocess_text(
        self, prompts: List[str], tokenize: Callable[[str], Tuple[Tensor, Tensor]]
    ) -> Tuple[Tensor, Tensor]:
        """
        Preprocess a list of prompts into CLIP/T5 text encodings.

        Args:
            prompts (List[str]): a list of strings to preprocess
            tokenize (Callable[[str], Tuple[Tensor, Tensor]]): a callable that turns a string into CLIP and T5 tokens

        Returns:
            Tuple[Tensor, Tensor]]: a tuple of `(clip_encodings, t5_encodings)`
        """
        clip_encodings = []
        t5_encodings = []
        for prompt in prompts:
            clip_tokens, t5_tokens = tokenize(prompt)

            clip_tokens = clip_tokens.to(device=self._device).unsqueeze(0)
            t5_tokens = t5_tokens.to(device=self._device).unsqueeze(0)

            clip_encodings.append(self._clip_encoder(clip_tokens).cpu())
            t5_encodings.append(self._t5_encoder(t5_tokens).cpu())

        clip_encodings = torch.cat(clip_encodings, dim=0)
        t5_encodings = torch.cat(t5_encodings, dim=0)
        return clip_encodings, t5_encodings

    def _should_preprocess(self, ids):
        if self._preprocess_again_if_exists:
            return True

        for id in ids:
            path = self._dir / f"{id}.pt"
            if not path.exists():
                return True

        return False


def flux_preprocessor(
    # arguments from the recipe
    device: torch.device,
    dtype: torch.dtype,
    *,
    # arguments from the config
    autoencoder_path: str = "/tmp/flux/ae.safetensors",
    clip_text_encoder_path: str = "/tmp/clip/model.safetensors",
    t5_encoder_path: str = "/tmp/t5/pytorch_model.bin",
    preprocessed_data_dir: str,
    preprocess_again_if_exists: bool = False,
    batch_size: int = 1,
    flux_model_name: str = "FLUX.1-dev",
) -> FluxPreprocessor:
    """
    Preprocess text-to-image data for finetuning Flux models.

    Args:
        device (torch.device): device to do preprocessing on
        dtype (torch.dtype): data type to do preprocessing in
        autoencoder_path (str): path to the Flux image autoencoder "ae.safetensors" file.
            download with: `tune download black-forest-labs/FLUX.1-dev --output-dir /tmp/flux`
            default: "/tmp/flux"
        clip_text_encoder_path (str): CLIP text encoder
            download with: `tune download openai/clip-vit-large-patch14 --output-dir /tmp/clip`
            default: "/tmp/clip"
        t5_encoder_path (str): T5 text encoder
            download with: `tune download google/t5-v1_1-xxl --output-dir /tmp/t5`
            default: "/tmp/t5"
        preprocessed_data_dir (str): folder where the preprocessed data will be stored
        preprocess_again_if_exists (bool): if false, data samples that already have preprocessed data will be skippped
            default: False
        batch_size (int): batch size to use when preprocessing datasets
        flux_model_name (str): "FLUX.1-dev" or "FLUX.1-schnell" (affects the T5 encoder max seq len)
            default: "FLUX.1-dev"

    Returns:
        FluxPreprocessor
    """
    autoencoder = flux_1_autoencoder()
    autoencoder.load_state_dict(flux_ae_hf_to_tune(safe_torch_load(autoencoder_path)))

    clip_encoder = clip_text_vit_large_patch14()
    clip_encoder.load_state_dict(
        clip_text_hf_to_tune(safe_torch_load(clip_text_encoder_path))
    )

    t5_encoder = t5_v1_1_xxl_encoder(max_seq_len=get_t5_max_seq_len(flux_model_name))
    t5_encoder.load_state_dict(t5_encoder_hf_to_tune(safe_torch_load(t5_encoder_path)))

    return FluxPreprocessor(
        autoencoder=autoencoder,
        clip_encoder=clip_encoder,
        t5_encoder=t5_encoder,
        preprocessed_data_dir=preprocessed_data_dir,
        preprocess_again_if_exists=preprocess_again_if_exists,
        batch_size=batch_size,
        device=device,
        dtype=dtype,
    )
