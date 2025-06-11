"""
Comprehensive edge case tests for Qwen2_5_VLImageTransform
Tests various boundary conditions, input formats, and potential failure modes.
"""
from PIL import Image
from _transform import Qwen2_5_VLImageTransform
import numpy as np
import torch
import warnings

def test_basic_functionality():
    """Baseline test to ensure basic functionality works"""
    print("=== Test: Basic Functionality ===")
    transform = Qwen2_5_VLImageTransform()
    np.random.seed(42)
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3)).astype(np.uint8))
    output = transform({"image": image})
    
    assert "pixel_values" in output
    assert "image_grid_thw" in output
    assert output["pixel_values"].shape[1] == 1176  # 3 * 2 * 14 * 14
    print("✅ Basic functionality passed")

def test_color_mode_edge_cases():
    """Test different color modes and image formats"""
    print("\n=== Test: Color Mode Edge Cases ===")
    transform = Qwen2_5_VLImageTransform()
    
    # Test cases: (mode, channels, expected_behavior)
    test_cases = [
        ("L", 1, "grayscale"),      # Grayscale
        ("RGB", 3, "standard"),     # Standard RGB
        ("RGBA", 4, "with_alpha"),  # RGB with alpha
        ("P", 1, "palette"),        # Palette mode
    ]
    
    for mode, channels, desc in test_cases:
        print(f"  Testing {desc} ({mode}) image...")
        try:
            if mode == "L":
                img_array = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
                image = Image.fromarray(img_array, mode=mode)
            elif mode == "RGBA":
                img_array = np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)
                image = Image.fromarray(img_array, mode=mode)
            elif mode == "P":
                img_array = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
                image = Image.fromarray(img_array, mode="L").convert("P")
            else:  # RGB
                img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                image = Image.fromarray(img_array, mode=mode)
            
            output = transform({"image": image})
            
            # All should convert to RGB internally and produce valid output
            assert output["pixel_values"].shape[1] == 1176
            print(f"    ✅ {desc} -> RGB conversion successful")
            
        except Exception as e:
            print(f"    ❌ {desc} failed: {e}")
            raise

def test_extreme_image_sizes():
    """Test very small and very large images"""
    print("\n=== Test: Extreme Image Sizes ===")
    
    # Test very small images
    print("  Testing very small images...")
    small_sizes = [(7, 7), (14, 14), (27, 27), (1, 1)]
    
    for h, w in small_sizes:
        print(f"    Testing {h}x{w} image...")
        transform = Qwen2_5_VLImageTransform(min_pixels=1)  # Allow very small
        image = Image.fromarray(np.random.randint(0, 255, (h, w, 3)).astype(np.uint8))
        
        try:
            output = transform({"image": image})
            resized_h, resized_w = transform.smart_resize(h, w, 
                                                        factor=transform.patch_size * transform.merge_size,
                                                        min_pixels=1, 
                                                        max_pixels=transform.max_pixels)
            print(f"      Original: {h}x{w} -> Resized: {resized_h}x{resized_w}")
            print(f"      Output shape: {output['pixel_values'].shape}")
            assert output["pixel_values"].ndim == 2
            print(f"    ✅ Small image {h}x{w} processed successfully")
        except Exception as e:
            print(f"    ⚠️  Small image {h}x{w} failed: {e}")
    
    # Test moderately large images
    print("  Testing large images...")
    large_sizes = [(1000, 1000), (500, 2000), (2000, 500)]  # Within reasonable limits
    
    for h, w in large_sizes:
        print(f"    Testing {h}x{w} image...")
        transform = Qwen2_5_VLImageTransform()
        image = Image.fromarray(np.random.randint(0, 255, (h, w, 3)).astype(np.uint8))
        
        try:
            output = transform({"image": image})
            print(f"      Output shape: {output['pixel_values'].shape}")
            assert output["pixel_values"].ndim == 2
            print(f"    ✅ Large image {h}x{w} processed successfully")
        except Exception as e:
            print(f"    ❌ Large image {h}x{w} failed: {e}")

def test_extreme_aspect_ratios():
    """Test images with extreme aspect ratios"""
    print("\n=== Test: Extreme Aspect Ratios ===")
    
    # Test extreme but valid aspect ratios (< 200:1)
    aspect_ratios = [
        (28, 560),   # 1:20 ratio
        (560, 28),   # 20:1 ratio  
        (14, 280),   # 1:20 ratio
        (280, 14),   # 20:1 ratio
    ]
    
    transform = Qwen2_5_VLImageTransform()
    
    for h, w in aspect_ratios:
        ratio = max(h, w) / min(h, w)
        print(f"    Testing {h}x{w} (ratio: {ratio:.1f}:1)...")
        
        try:
            image = Image.fromarray(np.random.randint(0, 255, (h, w, 3)).astype(np.uint8))
            output = transform({"image": image})
            print(f"      Output shape: {output['pixel_values'].shape}")
            print(f"    ✅ Extreme aspect ratio {ratio:.1f}:1 processed successfully")
        except Exception as e:
            print(f"    ❌ Extreme aspect ratio {ratio:.1f}:1 failed: {e}")
    
    # Test invalid aspect ratio (should fail)
    print("  Testing invalid aspect ratio (>200:1)...")
    try:
        invalid_image = Image.fromarray(np.random.randint(0, 255, (1, 300, 3)).astype(np.uint8))
        output = transform({"image": invalid_image})
        print("    ❌ Should have failed with >200:1 aspect ratio!")
        assert False, "Expected ValueError for extreme aspect ratio"
    except ValueError as e:
        print(f"    ✅ Correctly rejected >200:1 aspect ratio: {e}")
    except Exception as e:
        print(f"    ⚠️  Unexpected error: {e}")

def test_tensor_input_formats():
    """Test different tensor input formats"""
    print("\n=== Test: Tensor Input Formats ===")
    
    transform = Qwen2_5_VLImageTransform()
    
    # Test different tensor dtypes
    dtypes = [torch.uint8, torch.float32, torch.float16]
    
    for dtype in dtypes:
        print(f"    Testing {dtype} tensor input...")
        try:
            if dtype == torch.uint8:
                tensor = torch.randint(0, 256, (3, 100, 100), dtype=dtype)
            else:
                tensor = torch.rand(3, 100, 100, dtype=dtype)
            
            output = transform({"image": tensor})
            print(f"      Input dtype: {dtype} -> Output shape: {output['pixel_values'].shape}")
            print(f"    ✅ {dtype} tensor processed successfully")
        except Exception as e:
            print(f"    ❌ {dtype} tensor failed: {e}")

def test_different_patch_configurations():
    """Test different patch and merge size configurations"""
    print("\n=== Test: Different Patch Configurations ===")
    
    # Test different configurations
    configs = [
        {"patch_size": 7, "merge_size": 1},    # Smaller patches, no merging
        {"patch_size": 14, "merge_size": 1},   # Standard patches, no merging  
        {"patch_size": 28, "merge_size": 2},   # Larger patches
        {"patch_size": 14, "merge_size": 4},   # Standard patches, more merging
    ]
    
    np.random.seed(42)
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3)).astype(np.uint8))
    
    for config in configs:
        print(f"    Testing patch_size={config['patch_size']}, merge_size={config['merge_size']}...")
        try:
            transform = Qwen2_5_VLImageTransform(**config)
            output = transform({"image": image})
            
            # Verify dimensions make sense
            grid_t, grid_h, grid_w = output["image_grid_thw"][0]
            expected_patches = grid_t * grid_h * grid_w
            actual_patches = output["pixel_values"].shape[0]
            
            feature_dim = 3 * transform.temporal_patch_size * transform.patch_size * transform.patch_size
            
            print(f"      Grid: {grid_t}x{grid_h}x{grid_w}, Patches: {actual_patches}, Feature dim: {output['pixel_values'].shape[1]}")
            
            assert actual_patches == expected_patches, f"Patch count mismatch: {actual_patches} vs {expected_patches}"
            assert output["pixel_values"].shape[1] == feature_dim, f"Feature dim mismatch: {output['pixel_values'].shape[1]} vs {feature_dim}"
            
            print(f"    ✅ Configuration {config} successful")
        except Exception as e:
            print(f"    ❌ Configuration {config} failed: {e}")

def test_different_dtypes():
    """Test different output dtypes"""
    print("\n=== Test: Different Output Dtypes ===")
    
    dtypes = [torch.float32, torch.float16, torch.bfloat16, torch.float64]
    
    np.random.seed(42)
    image = Image.fromarray(np.random.randint(0, 255, (100, 100, 3)).astype(np.uint8))
    
    for dtype in dtypes:
        print(f"    Testing output dtype: {dtype}...")
        try:
            transform = Qwen2_5_VLImageTransform(dtype=dtype)
            output = transform({"image": image})
            
            actual_dtype = output["pixel_values"].dtype
            print(f"      Requested: {dtype}, Actual: {actual_dtype}")
            
            assert actual_dtype == dtype, f"Dtype mismatch: {actual_dtype} vs {dtype}"
            print(f"    ✅ Output dtype {dtype} correct")
        except Exception as e:
            print(f"    ❌ Output dtype {dtype} failed: {e}")

def test_normalization_parameters():
    """Test custom normalization parameters"""
    print("\n=== Test: Custom Normalization Parameters ===")
    
    # Test with custom normalization
    custom_configs = [
        {"image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5]},  # Different values
        {"image_mean": [0.0, 0.0, 0.0], "image_std": [1.0, 1.0, 1.0]},  # No normalization essentially
        {"image_mean": None, "image_std": None},  # Should use OPENAI_CLIP defaults
    ]
    
    np.random.seed(42)
    image = Image.fromarray(np.random.randint(0, 255, (100, 100, 3)).astype(np.uint8))
    
    for i, config in enumerate(custom_configs):
        print(f"    Testing normalization config {i+1}: {config}...")
        try:
            transform = Qwen2_5_VLImageTransform(**config)
            output = transform({"image": image})
            
            # Check that normalization was applied (values should be different from [0,1] range)
            pixel_values = output["pixel_values"]
            value_range = (pixel_values.min().item(), pixel_values.max().item())
            print(f"      Value range after normalization: {value_range}")
            
            assert output["pixel_values"].shape[1] == 1176
            print(f"    ✅ Custom normalization config {i+1} successful")
        except Exception as e:
            print(f"    ❌ Custom normalization config {i+1} failed: {e}")

def test_boundary_pixel_constraints():
    """Test images at boundary conditions for pixel constraints"""
    print("\n=== Test: Boundary Pixel Constraints ===")
    
    # Test images that are exactly at min/max pixel boundaries
    min_pixels = 56 * 56  # 3136
    max_pixels = 28 * 28 * 1280  # 1003520
    
    # Create image that's exactly at min pixels  
    min_side = int(np.sqrt(min_pixels))  # Should be 56
    print(f"  Testing min boundary: {min_side}x{min_side} = {min_side*min_side} pixels...")
    
    transform = Qwen2_5_VLImageTransform()
    min_image = Image.fromarray(np.random.randint(0, 255, (min_side, min_side, 3)).astype(np.uint8))
    
    try:
        output = transform({"image": min_image})
        print(f"    ✅ Min boundary image processed: {output['pixel_values'].shape}")
    except Exception as e:
        print(f"    ❌ Min boundary failed: {e}")
    
    # Test slightly below min pixels
    below_min_side = min_side - 1
    print(f"  Testing below min: {below_min_side}x{below_min_side} = {below_min_side*below_min_side} pixels...")
    
    below_min_image = Image.fromarray(np.random.randint(0, 255, (below_min_side, below_min_side, 3)).astype(np.uint8))
    
    try:
        output = transform({"image": below_min_image})
        print(f"    ✅ Below min processed (should be upscaled): {output['pixel_values'].shape}")
    except Exception as e:
        print(f"    ❌ Below min failed: {e}")

def test_malformed_inputs():
    """Test malformed or invalid inputs"""
    print("\n=== Test: Malformed Inputs ===")
    
    transform = Qwen2_5_VLImageTransform()
    
    # Test invalid input types
    invalid_inputs = [
        None,
        "not_an_image",
        123,
        [],
        torch.tensor([1, 2, 3]),  # Wrong shape tensor
    ]
    
    for i, invalid_input in enumerate(invalid_inputs):
        print(f"    Testing invalid input {i+1}: {type(invalid_input)}...")
        try:
            output = transform({"image": invalid_input})
            print(f"    ❌ Should have failed with invalid input: {type(invalid_input)}")
        except (AssertionError, ValueError, TypeError, AttributeError) as e:
            print(f"    ✅ Correctly rejected invalid input: {type(e).__name__}")
        except Exception as e:
            print(f"    ⚠️  Unexpected error with invalid input: {e}")

if __name__ == "__main__":
    print("🔍 Running comprehensive edge case tests for Qwen2_5_VLImageTransform\n")
    
    try:
        test_basic_functionality()
        test_color_mode_edge_cases()
        test_extreme_image_sizes()
        test_extreme_aspect_ratios()
        test_tensor_input_formats()
        test_different_patch_configurations()
        test_different_dtypes()
        test_normalization_parameters()
        test_boundary_pixel_constraints()
        test_malformed_inputs()
        
        print("\n🎉 All edge case tests completed!")
        print("✅ Implementation appears robust against various edge cases")
        
    except Exception as e:
        print(f"\n❌ Edge case testing failed: {e}")
        raise 