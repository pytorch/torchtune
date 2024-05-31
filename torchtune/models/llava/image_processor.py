import logging
from typing import List, Optional, Union

import numpy as np
import PIL

import torch
from torch import nn

from torchtune.utils.image_transforms import find_supported_resolutions, GetImagePatches
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F

"""
#TODO: llama and llava implemente different resize and pad stragies
llama pads using 0,0
llava padas using center + resize
"""
logger = logging.getLogger(__name__)

ImageInput = Union[
    "PIL.Image.Image", np.ndarray, "torch.Tensor", List["PIL.Image.Image"], List[np.ndarray], List["torch.Tensor"]
]  # noqa

class ImageProcessor(nn.Module):
    r"""
    Constructs a image processor for VLMs

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by
            `do_resize` in the `preprocess` method.
        size (`int`, defaults to 224):
            Size of the image after resizing. The shortest edge of the image is resized to size, with
            the longest edge resized to keep the input aspect ratio. Can be overridden by `size` in the `preprocess`
            method.
        possible_resolutions (`List[int]` *optional*):
            A list of possible resolutions to use for processing high resolution images. For example [[672, 336], [336, 672], [672, 672], [336, 1008], [1008, 336]].
            The best resolution is selected based on the original size of the image.
        max_num_chunks (`int`, *optional*, defaults to 4):
            The maximum number of chunks to use when processing high resolution images. Will be ignored if possible_resolutions
            is specified. Otherwise, will be used to calculate the best possible_resolutions.
        resample (`str`, *optional*, defaults to `"bicubic"`):
            The resampling filter to use when resizing the image.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by `do_rescale` in
            the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overridden by `rescale_factor` in the `preprocess`
            method.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by `do_normalize` in the `preprocess` method.
        image_mean (`float` or `List[float]`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image.
        image_std (`float` or `List[float]`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image.
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
        keep_original_and_resize (`bool`, *optional*, defaults to `True`):
            Whether to keep the original image in the output. In this case, it will be resized and added as another patch.
    """

    def __init__(
        self,
        patch_size: int = 224,
        possible_resolutions: Optional[List] = None,
        max_num_chunks: int = 4,
        resample:str = "bicubic",
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = True,
        keep_original_and_resize: bool = True
    ) -> None:
        super().__init__()
        # If possible_resolutions are not given, then calculate possible ones based on max_num_chunks
        if not possible_resolutions:
            possible_resolutions = find_supported_resolutions(
                max_num_chunks=max_num_chunks, patch_size=patch_size
            )
        else:
            possible_resolutions = possible_resolutions

        print('possible_resolutions', possible_resolutions)

        transforms = []
        if do_convert_rgb:
            transforms.append(v2.RGB()) # expects shape […, 1 or 3, H, W]

        transforms.append(
            GetImagePatches(
                            possible_resolutions = possible_resolutions,
                            patch_size = patch_size,
                            resample = resample,
                            keep_original_and_resize=keep_original_and_resize,
                            ))

        if do_rescale:
            transforms.append(v2.Normalize(mean=[0]*3, std=[1/rescale_factor]*3))

        if do_normalize:
            transforms.append(v2.Normalize(mean=image_mean, std=image_std))

        self.transforms = nn.Sequential(*transforms)

    def preprocess(
        self,
        images: ImageInput) -> list[torch.Tensor]:

        # always treat input as a list of images
        if not isinstance(images, list):
            images = [images]

        # make all images torch tensors.
        # TODO: to_image does not support torchscript.
        # can we increase self.transforms performance by not combining it
        # with to_image?
        images = [F.to_image(image) for image in images]

        return [self.transforms(image) for image in images]

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import requests

    image = "https://cdn.theatlantic.com/thumbor/W544GIT4l3z8SG-FMUoaKpFLaxE=/0x131:2555x1568/1600x900/media/img/mt/2017/06/shutterstock_319985324/original.jpg"

    # read image
    image = PIL.Image.open(requests.get(image, stream=True).raw)

    image_transforms = ImageProcessor(
        image_mean=[0, 0, 0], image_std=[1, 1, 1], rescale_factor=1
    )
    output = image_transforms.preprocess(image)

    print(output.shape)

    # print each rendered image. Shape is N, C, H, W
    for image_patches in output:
        for patch in image_patches:
            plt.imshow(patch.permute(1, 2, 0))
            plt.show()
