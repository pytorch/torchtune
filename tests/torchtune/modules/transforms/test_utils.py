
import pytest

from torchtune.modules.transforms.utils import ( 
    find_supported_resolutions,
    GetBestResolution,
)

class TestUtils:
    @pytest.mark.parametrize(
        "params",
        [
            {
                "max_num_chunks": 1,
                "patch_size": 224,
                "expected_resolutions": [(224, 224)]
            },
            {
                "max_num_chunks": 2,
                "patch_size": 100,
                "expected_resolutions": [(100, 200), (200, 100), (100, 100)]
            },
            {
                "max_num_chunks": 3,
                "patch_size": 50,
                "expected_resolutions": [(50, 150), (150, 50), (50, 100), (100, 50), (50, 50)]
            },
            {
                "max_num_chunks": 4,
                "patch_size": 300,
                "expected_resolutions": [(300, 1200), (600, 600), (300, 300), (1200, 300), (300, 900), (900, 300), (300, 600), (600, 300)]
            }
        ],
    )
    def test_find_supported_resolutions(self, params):
        max_num_chunks = params["max_num_chunks"]
        patch_size = params["patch_size"]
        expected_resolutions = params["expected_resolutions"]
        resolutions = find_supported_resolutions(max_num_chunks, patch_size)

        assert len(set(resolutions)) == len(resolutions), "Resolutions should be unique"
        assert set(resolutions) == set(expected_resolutions), f"Expected resolutions {expected_resolutions} but got {resolutions}"

    @pytest.mark.parametrize(
        "params",
        [
            {
                "image_size": (800, 600),
                "patch_size": 224,
                "max_num_chunks": 4,
                "possible_resolutions": [(224, 896), (448, 448), (224, 224), (896, 224), (224, 672), (672, 224), (224, 448), (448, 224)],
                "expected_best_resolution": (448, 448)
            },
            {
                "image_size": (200, 300),
                "patch_size": 224,
                "max_num_chunks": 3,
                "possible_resolutions": [(224, 672), (672, 224), (224, 448), (448, 224), (224, 224)],
                "expected_best_resolution": (224, 448)
            },
            {
                "image_size": (500, 500),
                "patch_size": 100,
                "max_num_chunks": 3,
                "possible_resolutions": None,
                "expected_best_resolution": (100, 100)
            },
            {
                "image_size": (500, 500),
                "patch_size": 1000,
                "max_num_chunks": 4,
                "possible_resolutions": None, 
                "expected_best_resolution": (1000, 1000)
            },
            {
                "image_size": (600, 200),
                "patch_size": 300,
                "max_num_chunks": 4,
                "possible_resolutions": None,
                "expected_best_resolution": (900, 300)
            }
        ],
    )
    def test_get_best_resolution(self, params):
        image_size = params["image_size"]
        patch_size = params["patch_size"]
        max_num_chunks = params["max_num_chunks"]
        possible_resolutions = params["possible_resolutions"]
        expected_best_resolution = params["expected_best_resolution"]

        resolver = GetBestResolution(possible_resolutions, max_num_chunks, patch_size)
        best_resolution = resolver(image_size)

        assert tuple(best_resolution) == expected_best_resolution, f"Expected best resolution {expected_best_resolution} but got {best_resolution}"
