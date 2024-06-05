import pytest
import unittest
from torchtune.models.llava_next.image_processor import GetImagePatches, ImageProcessor
import numpy as np
import PIL

class TestGetImagePatches(unittest.TestCase):
    def setUp(self):
        self.possible_resolutions = [
            [224, 896],
            [448, 448],
            [224, 224],
            [896, 224],
            [224, 672],
            [672, 224],
            [224, 448],
            [448, 224],
        ]

        self.image_processor = ImageProcessor(
        patch_size = 336,
        possible_resolutions = [
            (224, 896),
            (448, 448),
            (224, 224),
            (896, 224),
            (224, 672),
            (672, 224),
            (224, 448),
            (448, 224),
        ],
        max_num_chunks = 4,
        resample = "bicubic",
        do_rescale = True,
        rescale_factor = 1 / 255,
        do_normalize = True,
        image_mean = [
            0.48145466,
            0.4578275,
            0.40821073,
        ],
        image_std = [
            0.26862954,
            0.26130258,
            0.27577711,
        ],
    )

    @pytest.mark.parametrize("params", [
        {"image_size": [200, 300], "target_resolution": [450, 200], "expected": (134, 200)},
        {"image_size": [200, 300], "target_resolution": [100, 450], "expected": (100, 150)},
        {"image_size": [400, 400], "target_resolution": [400, 400], "expected": (400, 400)},
        {"image_size": [800, 600], "target_resolution": [1250, 350], "expected": (467, 350)},
        {"image_size": [800, 600], "target_resolution": [450, 1300], "expected": (450, 338)},
    ])
    def test_get_new_size_without_distortion(self, params):
        image_size = params['image_size']
        target_resolution = params['target_resolution']
        expected = params['expected']
        output_resolution = GetImagePatches._get_new_size_without_distortion(image_size=image_size, target_resolution=target_resolution)
        assert output_resolution == expected, f"Expected {expected} but got {output_resolution}"
   
    @pytest.mark.parametrize("params", [
        {"image": np.random.rand(3, 200, 300), "patch_size": 50, "expected_num_patches": 24},
        {"image": np.random.rand(3, 400, 400), "patch_size": 350, "expected_num_patches": 4},
        {"image": np.random.rand(3, 800, 600), "patch_size": 1000, "expected_num_patches": 1},
        {"image": np.random.rand(3, 400, 500), "patch_size": 450, "expected_num_patches": 2},
    ])
    def test_divide_to_patches(self, params):
        image = params['image']
        patch_size = params['patch_size']
        expected_num_patches = params['expected_num_patches']
        patches = GetImagePatches._divide_to_patches(image=image, patch_size=patch_size)
        assert len(patches) == expected_num_patches, f"Expected {expected_num_patches} patches but got {len(patches)}"
    
    @pytest.mark.parametrize("params", [
        {'original_size': (200, 300), 'possible_resolutions': [(100, 100), (200, 200), (300, 300), (400, 400)], 'expected': (300, 300)},
        {'original_size': (400, 400), 'possible_resolutions': [(350, 350), (450, 450), (550, 550)], 'expected': (450, 450)},
        {'original_size': (800, 600), 'possible_resolutions': [(800, 600), (1200, 800), (1600, 1200)], 'expected': (800, 600)},
        {'original_size': (800, 600), 'possible_resolutions': [(600, 800), (1600, 1200), (80, 60)], 'expected': (1600, 1200)}
    ])
    def test_select_best_resolution(self, params):
        original_size = params['original_size']
        expected = params['expected']
        possible_resolutions = params['possible_resolutions']
        best_resolution = GetImagePatches._select_best_resolution(original_size=original_size, possible_resolutions=possible_resolutions)
        assert best_resolution == expected, f"Expected {expected} but got {best_resolution}"

    @pytest.mark.parametrize("params", [
        {'image_size': (100, 100, 3), 'expected_shape': (2, 3, 336, 336)},
        {'image_size': (200, 300), 'expected_shape': (3, 3, 336, 336)},
        {'image_size': (400, 400, 3), 'expected_shape': (5, 3, 336, 336)},
        {'image_size': (800, 600), 'expected_shape': (5, 3, 336, 336)}
    ])
    def test_preprocess_output_shape(self, params):
        image_size = params['image_size']
        expected_shape = params['expected_shape']

        image = (np.random.rand(*image_size) * 255).astype(np.uint8)
        image = PIL.Image.fromarray(image)
        
        output = self.image_processor.preprocess(image)
        pixel_values = output["pixel_values"]

        assert pixel_values.shape == expected_shape, f"Expected shape {expected_shape} but got {pixel_values.shape} for image size {image_size}"
        assert output["image_size"] == (image_size[0], image_size[1]), f"Expected image size {image_size} but got {output['image_size']}"
