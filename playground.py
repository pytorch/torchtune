from torchvision.transforms import v2
import time
import math
from typing import Tuple

import torch


if __name__ == "__main__":

    class ImageProcessor:
        def __init__(self, size):
            self.size = size

        def find_closest_aspect_ratio(self, aspect_ratios, image_width, image_height):
            target_aspect_ratio = image_width / image_height
            closest_pair = None

            if target_aspect_ratio >= 1:
                # Handling landscape or square orientations
                closest_pair = min(
                    [ratio for ratio in aspect_ratios.keys() if ratio <= target_aspect_ratio],
                    key=lambda ratio: abs(ratio - target_aspect_ratio),
                )
                aspect_pairs = aspect_ratios[closest_pair]
                # Select the pair with the maximum width
                width_based_pairs = [(index, self.size * width) for index, (width, _) in enumerate(aspect_pairs)]
                target_index = max(width_based_pairs, key=lambda x: x[1])[0]
            else:
                # Handling portrait orientations
                closest_pair = min(
                    [ratio for ratio in aspect_ratios.keys() if ratio > target_aspect_ratio],
                    key=lambda ratio: abs(1 / ratio - 1 / target_aspect_ratio),
                )
                aspect_pairs = aspect_ratios[closest_pair]
                # Select the pair with the maximum height
                height_based_pairs = [(index, self.size * height) for index, (_, height) in enumerate(aspect_pairs)]
                target_index = max(height_based_pairs, key=lambda x: x[1])[0]
            selected_pair = aspect_pairs[target_index]
            return selected_pair

    def select_best_resolution(original_size, possible_resolutions):
        original_height, original_width = original_size
        best_fit = None
        max_effective_resolution = 0
        min_wasted_resolution = float("inf")

        for height, width in possible_resolutions:
            scale = min(width / original_width, height / original_height)
            downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
            effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
            wasted_resolution = (width * height) - effective_resolution

            if effective_resolution > max_effective_resolution or (
                effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution
            ):
                max_effective_resolution = effective_resolution
                min_wasted_resolution = wasted_resolution
                best_fit = (height, width)

        return best_fit

    # Example usage:
    size = 224
    processor = ImageProcessor(size=size)  # size is arbitrary as it's not used in aspect ratio calculation

    for original_size in [(600, 800), (1200, 1600), (200, 1000), (100,200), (300, 900)]:
        aspect_ratio_dict = {
                0.2: [(1, 5)],
                5.0: [(5, 1)],
                0.25: [(1, 4)],
                1.0: [(2, 2), (1, 1)],
                4.0: [(4, 1)],
                0.3333333333333333: [(1, 3)],
                3.0: [(3, 1)],
                0.5: [(1, 2)],
                2.0: [(2, 1)]
            }

        possible_resolutions = []
        for key, value in aspect_ratio_dict.items():
            for height, depth in value:
                possible_resolutions.append((height*size, depth*size))

        best_resolution = select_best_resolution(original_size, possible_resolutions)
        closest_aspect_ratio = processor.find_closest_aspect_ratio(aspect_ratio_dict, original_size[0], original_size[1])

        print('----')
        print("image input size", original_size)
        print("llava chosen aspect ratio:", best_resolution[0]/size, best_resolution[1]/size, "resolution:", best_resolution)
        print("llama chosen aspect ratio:", closest_aspect_ratio, "resolution:", closest_aspect_ratio[0]*size, closest_aspect_ratio[1]*size)
