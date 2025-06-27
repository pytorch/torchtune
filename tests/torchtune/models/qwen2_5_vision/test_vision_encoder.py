"""Test file for Qwen2.5-VL Vision Encoder component."""

import torch
import numpy as np
from PIL import Image
from torchtune.models.qwen2_5_vision import qwen2_5_vision_encoder
from torchtune.models.qwen2_5_vision._transform import Qwen2_5_VLImageTransform
from transformers import AutoProcessor, AutoModelForImageTextToText


def create_test_image(width: int = 224, height: int = 224) -> Image.Image:
    """Create a simple test image."""
    # Create a random RGB image
    image_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(image_array)


def load_hf_vision_model():
    """Load HuggingFace vision model for comparison."""
    hf_model_path = "/mnt/vast/share/inf2-training/models/open_source/Qwen2.5-VL-7B-Instruct"
    hf_processor = AutoProcessor.from_pretrained(hf_model_path)
    hf_model = AutoModelForImageTextToText.from_pretrained(hf_model_path)
    return hf_processor, hf_model.visual


def test_vision_encoder_basic():
    """Test basic vision encoder functionality."""
    print("Testing basic vision encoder functionality...")
    
    try:
        # Create the vision encoder
        vision_encoder = qwen2_5_vision_encoder()
        vision_encoder.eval()
        
        # Create test input
        batch_size = 2
        seq_len = 256  # Example sequence length after patching
        embed_dim = vision_encoder.patch_embed.embed_dim
        
        # Create random input tensor (simulating patched image embeddings)
        hidden_states = torch.randn(seq_len, embed_dim)
        
        # Create grid_thw (temporal, height, width grid info)
        # For a single image: T=1, H and W depend on image size and patch size
        grid_thw = torch.tensor([[1, 16, 16]])  # 1 temporal, 16x16 spatial grid
        
        # Forward pass
        with torch.no_grad():
            output = vision_encoder(hidden_states, grid_thw)
        
        # Check output properties
        assert isinstance(output, torch.Tensor), "Output should be a tensor"
        assert output.dim() == 2, "Output should be 2D tensor [seq_len, hidden_dim]"
        assert output.shape[0] <= seq_len, "Output sequence length should be <= input sequence length"
        
        print(f"✅ Vision encoder basic test passed!")
        print(f"   - Input shape: {hidden_states.shape}")
        print(f"   - Grid THW: {grid_thw}")
        print(f"   - Output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Vision encoder basic test failed: {e}")
        return False


def test_vision_encoder_with_image_transform():
    """Test vision encoder with actual image input through transform."""
    print("Testing vision encoder with image transform...")
    
    try:
        # Create image transform
        image_transform = Qwen2_5_VLImageTransform(
            patch_size=14,
            merge_size=2,
            temporal_patch_size=2,
            min_pixels=3136,  # 56*56
            max_pixels=1003520,  # 28*28*1280
        )
        
        # Create vision encoder
        vision_encoder = qwen2_5_vision_encoder()
        vision_encoder.eval()
        
        # Create test image
        test_image = create_test_image(448, 448)  # Larger image for more patches
        
        # Transform image
        sample = {"image": test_image}
        transformed = image_transform(sample)
        
        pixel_values = transformed["pixel_values"]  # Should be [num_patches, channels*temporal*patch*patch]
        image_grid_thw = transformed["image_grid_thw"]  # Should be [temporal, height, width]
        
        print(f"   - Pixel values shape: {pixel_values.shape}")
        print(f"   - Image grid THW: {image_grid_thw}")
        
        # Forward pass through vision encoder
        with torch.no_grad():
            vision_output = vision_encoder(pixel_values, image_grid_thw.unsqueeze(0))
        
        # Check output
        assert isinstance(vision_output, torch.Tensor), "Vision output should be a tensor"
        assert vision_output.dim() == 2, "Vision output should be 2D"
        
        print(f"✅ Vision encoder with image transform test passed!")
        print(f"   - Final vision output shape: {vision_output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Vision encoder with image transform test failed: {e}")
        return False


def test_vision_encoder_different_sizes():
    """Test vision encoder with different image sizes."""
    print("Testing vision encoder with different image sizes...")
    
    try:
        # Create image transform
        image_transform = Qwen2_5_VLImageTransform(
            patch_size=14,
            merge_size=2,
            temporal_patch_size=2,
        )
        
        # Create vision encoder
        vision_encoder = qwen2_5_vision_encoder()
        vision_encoder.eval()
        
        # Test different image sizes
        test_sizes = [(224, 224), (448, 224), (224, 448), (336, 336)]
        
        for width, height in test_sizes:
            print(f"   Testing size {width}x{height}...")
            
            # Create and transform image
            test_image = create_test_image(width, height)
            sample = {"image": test_image}
            transformed = image_transform(sample)
            
            pixel_values = transformed["pixel_values"]
            image_grid_thw = transformed["image_grid_thw"]
            
            # Forward pass
            with torch.no_grad():
                vision_output = vision_encoder(pixel_values, image_grid_thw.unsqueeze(0))
            
            # Check output
            assert isinstance(vision_output, torch.Tensor), f"Output should be tensor for size {width}x{height}"
            assert vision_output.dim() == 2, f"Output should be 2D for size {width}x{height}"
            
            print(f"     - Input: {pixel_values.shape}, Grid: {image_grid_thw}, Output: {vision_output.shape}")
        
        print(f"✅ Vision encoder different sizes test passed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Vision encoder different sizes test failed: {e}")
        return False


def run_all_tests():
    """Run all vision encoder tests."""
    print("=" * 50)
    print("Running Qwen2.5-VL Vision Encoder Tests")
    print("=" * 50)
    
    tests = [
        test_vision_encoder_basic,
        test_vision_encoder_with_image_transform,
        test_vision_encoder_different_sizes,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            results.append(False)
        print("-" * 30)
    
    # Summary
    passed = sum(results)
    total = len(results)
    print(f"Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed")
        
    return passed == total


if __name__ == "__main__":
    run_all_tests() 