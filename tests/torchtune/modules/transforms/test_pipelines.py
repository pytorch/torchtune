import numpy as np
import PIL
import pytest

import torch

from torchtune.modules.transforms.pipelines import VariableImageSizeTransforms

@pytest.fixture(autouse=True)
def image_processor():

    IMAGE_RES = 224
    IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
    IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)

    image_processor = VariableImageSizeTransforms(
        image_mean=IMAGE_MEAN,
        image_std=IMAGE_STD,
        patch_size=IMAGE_RES,
        possible_resolutions=None,
        max_num_chunks=4,
        resample="bilinear",
        do_rescale=True,
        rescale_factor=1 / 255,
        do_normalize=True,
        limit_upscaling_to_patch_size=True,
    )
    return image_processor

class TestPipelines:
    @pytest.mark.parametrize(
        "params",
        [
            {"image_size": (100, 100, 3), "expected_shape": torch.Size([1, 3, 224, 224])},
            {"image_size": (200, 300), "expected_shape": torch.Size([2, 3, 224, 224])},
            {"image_size": (400, 400, 3), "expected_shape": torch.Size([4, 3, 224, 224])},
            {"image_size": (800, 600), "expected_shape": torch.Size([4, 3, 224, 224])},
        ],
    )
    def test_shapes_variable_image_size_transforms(self, params, image_processor):

        image_size = params["image_size"]

        # Create a random image
        image = (np.random.rand(*image_size) * 255).astype(np.uint8)
        image = PIL.Image.fromarray(image)

        output = image_processor(image)
        pixel_values = output["pixel_values"]

        assert pixel_values.shape == params["expected_shape"], f"Expected shape {params['expected_shape']} but got {pixel_values.shape}"
