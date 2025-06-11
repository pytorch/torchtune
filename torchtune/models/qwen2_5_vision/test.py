from PIL import Image
from _transform import Qwen2_5_VLImageTransform
import numpy as np
import torch

# Try to import HuggingFace implementation for comparison
try:
    from transformers import Qwen2VLImageProcessor as HF_Qwen2_5_VLImageTransform
    HF_AVAILABLE = True
except ImportError:
    assert False, "HuggingFace transformers not available, skipping comparison"

# Create a test image
np.random.seed(42)  # For reproducible results
image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3)).astype(np.uint8))

print("=== Testing Qwen2_5_VLImageTransform ===")

# Test with default parameters
transform = Qwen2_5_VLImageTransform()
output = transform({"image": image})

print("Transform successful!")
print(f"pixel_values shape: {output['pixel_values'].shape}")
print(f"image_grid_thw: {output['image_grid_thw']}")

# Compare to HuggingFace implementation if available
if HF_AVAILABLE:
    print("\n=== Comparing with HuggingFace Implementation ===")
    hf_transform = HF_Qwen2_5_VLImageTransform()
    hf_output = hf_transform(image)
    
    print(f"HF pixel_values shape: {hf_output['pixel_values'].shape}")
    print(f"HF image_grid_thw shape: {hf_output['image_grid_thw'].shape}")
    print(f"HF image_grid_thw values: {hf_output['image_grid_thw']}")
    
    # Convert our output to numpy for comparison
    our_pixel_values = output["pixel_values"].detach().float().numpy()
    our_grid_thw = output["image_grid_thw"].detach().numpy()
    
    # Check shapes match
    shapes_match = (our_pixel_values.shape == hf_output["pixel_values"].shape and 
                   our_grid_thw.shape == hf_output["image_grid_thw"].shape)
    print(f"Shapes match: {shapes_match}")
    
    if shapes_match:
        # Check if grid_thw values match
        grid_values_match = np.array_equal(our_grid_thw, hf_output["image_grid_thw"])
        print(f"Grid THW values match: {grid_values_match}")
        
        # Check approximate pixel values (they might differ slightly due to dtype/precision)
        pixel_close = np.allclose(our_pixel_values, hf_output["pixel_values"], rtol=1e-4, atol=1e-6)
        print(f"Pixel values approximately match: {pixel_close}")
        
        if not pixel_close:
            diff_stats = np.abs(our_pixel_values - hf_output["pixel_values"])
            print(f"Max absolute difference: {np.max(diff_stats):.6f}")
            print(f"Mean absolute difference: {np.mean(diff_stats):.6f}")
    else:
        print("Cannot compare values due to shape mismatch")

# Test with custom parameters
print("\n=== Testing with custom parameters ===")
transform_custom = Qwen2_5_VLImageTransform(
    patch_size=14,
    merge_size=2,
    temporal_patch_size=2,
    min_pixels=1024,  # Smaller than default to test edge cases
    max_pixels=1003520,
    dtype=torch.float32
)

output_custom = transform_custom({"image": image})
print("Custom transform successful!")
print(f"pixel_values shape: {output_custom['pixel_values'].shape}")
print(f"image_grid_thw: {output_custom['image_grid_thw']}")

# Test with a smaller image
print("\n=== Testing with smaller image ===")
small_image = Image.fromarray(np.random.randint(0, 255, (28, 28, 3)).astype(np.uint8))
output_small = transform({"image": small_image})
print("Small image transform successful!")
print(f"pixel_values shape: {output_small['pixel_values'].shape}")
print(f"image_grid_thw: {output_small['image_grid_thw']}")

# Verify output dimensions make sense
grid_t, grid_h, grid_w = output["image_grid_thw"][0]  # Extract from [1, 3] shape
expected_patches = grid_t * grid_h * grid_w
actual_patches = output["pixel_values"].shape[0]
channels = 3
temporal_patch_size = 2
patch_size = 14

expected_feature_dim = channels * temporal_patch_size * patch_size * patch_size
actual_feature_dim = output["pixel_values"].shape[1]

print(f"\nValidation:")
print(f"Expected patches: {expected_patches}, Actual: {actual_patches}")
print(f"Expected feature dim: {expected_feature_dim}, Actual: {actual_feature_dim}")
print(f"Validation {'PASSED' if expected_patches == actual_patches and expected_feature_dim == actual_feature_dim else 'FAILED'}")

print("\nAll tests completed!")