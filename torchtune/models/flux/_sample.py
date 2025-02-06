import math
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
from PIL import Image
from torch import Tensor

from torchtune.models.flux._flow_model import FluxFlowModel
from torchtune.models.flux._preprocess import FluxPreprocessor
from torchtune.models.flux._util import generate_images

POSITION_DIM = 3


class FluxSampler:
    def __init__(
        self,
        # arguments from the recipe
        preprocessor: FluxPreprocessor,
        tokenize: Callable[[str], Tuple[Tensor, Tensor]],
        device: torch.device,
        dtype: torch.dtype,
        *,
        # arguments from the config
        samples_dir: str,
        width: int,
        height: int,
        seed: int,
        denoising_steps: int,
        guidance: float,
        batch_size: int,
        prompts: List[str],
        img_grid: Optional[Tuple[int, int]] = None,
    ):
        self._device = device
        self._dtype = dtype
        self._width = width
        self._height = height
        self._seed = seed
        self._denoising_steps = denoising_steps

        if img_grid is not None:
            if len(img_grid) != 2:
                raise ValueError("Sampler `img_grid` should be tuple of (rows, cols)")
            if math.prod(img_grid) != len(prompts):
                raise ValueError(
                    "Sampler `img_grid` rows*cols should equal the number of prompts"
                )
        self._img_grid = img_grid

        self._guidance = torch.full((batch_size,), guidance, device=device, dtype=dtype)

        self._samples_dir = Path(samples_dir)
        self._samples_dir.mkdir(parents=True, exist_ok=True)

        # preprocess the text prompts
        clip_encodings, t5_encodings = preprocessor.preprocess_text(prompts, tokenize)
        self._text_encodings = torch.utils.data.DataLoader(
            _TextEncodingsDataset(clip_encodings.cpu(), t5_encodings.cpu()),
            batch_size=batch_size,
            shuffle=False,
        )

        # keep the AE decoder for decoding generated images
        self._decoder = preprocessor.autoencoder.decoder

    @torch.no_grad
    def save_samples(
        self,
        model: FluxFlowModel,
        train_step: int,
    ):
        # generate images
        img_list = []
        for batch in self._text_encodings:
            clip_encodings = batch["clip_encoding"].to(
                device=self._device, dtype=self._dtype
            )
            t5_encodings = batch["t5_encoding"].to(
                device=self._device, dtype=self._dtype
            )
            imgs = generate_images(
                model=model,
                decoder=self._decoder,
                clip_encodings=clip_encodings,
                t5_encodings=t5_encodings,
                guidance=self._guidance,
                img_height=self._height,
                img_width=self._width,
                seed=self._seed,
                denoising_steps=self._denoising_steps,
                device=self._device,
                dtype=self._dtype,
            )
            for img in imgs:
                img_list.append(img)

        # save images
        if self._img_grid is None:
            # save individually
            for i, img in enumerate(img_list):
                _save_image(
                    img,
                    self._samples_dir / f"step{train_step:06d}_prompt{i:02d}.png",
                )
        else:
            # save as a grid of images
            rows, cols = self._img_grid
            img = _make_img_grid(img_list, rows, cols)
            _save_image(img, self._samples_dir / f"{train_step:06d}.png")


class _TextEncodingsDataset(torch.utils.data.Dataset):
    def __init__(self, clip_encodings, t5_encodings):
        super().__init__()
        assert len(clip_encodings) == len(t5_encodings)
        self._clip_encodings = clip_encodings
        self._t5_encodings = t5_encodings

    def __len__(self):
        return len(self._clip_encodings)

    def __item__(self, i):
        return {
            "clip_encoding": self._clip_encodings[i],
            "t5_encoding": self._t5_encodings[i],
        }


def _save_image(img, path):
    # c h w -> h w c
    img = img.permute(1, 2, 0)

    # [-1, 1] tensor -> [0, 255] numpy array
    img = img.clamp(-1, 1)
    img = (127.5 * (img + 1.0)).cpu().byte().numpy()

    # save to disk
    Image.fromarray(img).save(path)


def _make_img_grid(imgs, n_rows, n_cols):
    assert len(imgs) == n_rows * n_cols

    # concatenate horizontally
    rows = []
    for i in range(n_rows):
        row_images = imgs[i * n_cols : (i + 1) * n_cols]
        row = torch.cat(row_images, dim=2)
        rows.append(row)

    # concatenate vertically
    grid = torch.cat(rows, dim=1)

    return grid
