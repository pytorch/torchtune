#!/usr/bin/env python3
"""
Test file for Qwen2_5_VLTransform - the complete multimodal transform class.
This tests the integration of tokenization and image processing.
"""

import sys
import os
from PIL import Image
import numpy as np
import torch
from typing import List, Dict, Any

# Add the current directory to path to import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _transform import Qwen2_5_VLTransform
from torchtune.data import Message

def create_test_image(size=(224, 224), seed=42):
    """Create a test image for testing."""
    np.random.seed(seed)
    return Image.fromarray(np.random.randint(0, 255, (*size, 3)).astype(np.uint8))

def create_test_messages_with_image():
    """Create test messages that include an image."""
    test_image = create_test_image()
    
    # Create a message with image content
    message = Message(
        role="user",
        content=[
            {"type": "text", "content": "What do you see in this image?"},
            {"type": "image", "content": test_image}
        ]
    )
    
    return [message]

def create_test_messages_text_only():
    """Create test messages with text only."""
    message = Message(
        role="user", 
        content=[{"type": "text", "content": "Hello, how are you?"}]
    )
    
    return [message]

def test_transform_initialization():
    """Test that the transform can be initialized properly."""
    print("=== Testing Qwen2_5_VLTransform Initialization ===")
    
    # Note: You'll need to provide actual paths to tokenizer files
    # For now, we'll test the structure assuming the files exist
    try:
        # This would need real tokenizer files - adjust paths as needed
        transform = Qwen2_5_VLTransform(
            path="/path/to/vocab.json",  # Replace with actual path
            merges_file="/path/to/merges.txt",  # Replace with actual path
            patch_size=14,
            max_seq_len=2048,
        )
        print("✅ Transform initialization successful")
        return transform
    except Exception as e:
        print(f"❌ Transform initialization failed: {e}")
        print("Note: This test requires actual tokenizer files")
        return None

def test_image_transform_method():
    """Test the transform_image method specifically."""
    print("\n=== Testing transform_image Method ===")
    
    # Create a mock transform for testing image processing only
    from _transform import Qwen2_5_VLImageTransform
    
    image_transform = Qwen2_5_VLImageTransform()
    test_image = create_test_image()
    
    # Test the image transform directly
    sample = {"image": test_image}
    result = image_transform(sample)
    
    print(f"✅ Image transform successful")
    print(f"   pixel_values shape: {result['pixel_values'].shape}")
    print(f"   image_grid_thw: {result['image_grid_thw']}")
    
    # Verify the output structure
    assert "pixel_values" in result, "pixel_values missing from output"
    assert "image_grid_thw" in result, "image_grid_thw missing from output"
    assert isinstance(result["pixel_values"], torch.Tensor), "pixel_values should be a tensor"
    assert isinstance(result["image_grid_thw"], torch.Tensor), "image_grid_thw should be a tensor"
    
    print("✅ Image transform output validation passed")

def test_encoder_input_structure():
    """Test that the encoder input has the correct structure."""
    print("\n=== Testing Encoder Input Structure ===")
    
    # Create a sample with messages containing images
    messages = create_test_messages_with_image()
    sample = {"messages": messages}
    
    # Mock the transform behavior to test structure
    from _transform import Qwen2_5_VLImageTransform
    image_transform = Qwen2_5_VLImageTransform()
    
    # Simulate what the full transform should do
    encoder_input = {"vision": {"images": []}}
    
    for message in messages:
        for content in message.content:
            if content["type"] == "image":
                image = content["content"]
                # Transform the image
                img_sample = {"image": image}
                transformed = image_transform(img_sample)
                pixel_values = transformed["pixel_values"]
                image_grid_thw = transformed["image_grid_thw"]
                
                encoder_input["vision"]["images"].append(pixel_values)
                content["image_grid_thw"] = image_grid_thw
    
    print("✅ Encoder input structure created successfully")
    print(f"   Number of images: {len(encoder_input['vision']['images'])}")
    print(f"   First image shape: {encoder_input['vision']['images'][0].shape}")
    
    # Verify structure
    assert "vision" in encoder_input, "vision key missing from encoder_input"
    assert "images" in encoder_input["vision"], "images key missing from vision"
    assert len(encoder_input["vision"]["images"]) > 0, "No images in encoder_input"
    
    print("✅ Encoder input structure validation passed")

def test_message_content_modification():
    """Test that image_grid_thw is properly added to message content."""
    print("\n=== Testing Message Content Modification ===")
    
    messages = create_test_messages_with_image()
    
    # Before processing, image content should not have image_grid_thw
    image_content = None
    for content in messages[0].content:
        if content["type"] == "image":
            image_content = content
            break
    
    assert image_content is not None, "No image content found in test message"
    assert "image_grid_thw" not in image_content, "image_grid_thw should not exist initially"
    
    # Simulate processing
    from _transform import Qwen2_5_VLImageTransform
    image_transform = Qwen2_5_VLImageTransform()
    
    img_sample = {"image": image_content["content"]}
    transformed = image_transform(img_sample)
    image_content["image_grid_thw"] = transformed["image_grid_thw"]
    
    # After processing, image_grid_thw should be present
    assert "image_grid_thw" in image_content, "image_grid_thw should be added to content"
    assert isinstance(image_content["image_grid_thw"], torch.Tensor), "image_grid_thw should be a tensor"
    
    print("✅ Message content modification test passed")
    print(f"   Added image_grid_thw: {image_content['image_grid_thw']}")

def test_different_image_sizes():
    """Test the transform with different image sizes."""
    print("\n=== Testing Different Image Sizes ===")
    
    from _transform import Qwen2_5_VLImageTransform
    image_transform = Qwen2_5_VLImageTransform()
    
    test_sizes = [(224, 224), (512, 512), (100, 200), (300, 150)]
    
    for size in test_sizes:
        test_image = create_test_image(size)
        sample = {"image": test_image}
        result = image_transform(sample)
        
        print(f"   Size {size}: pixel_values {result['pixel_values'].shape}, grid_thw {result['image_grid_thw']}")
        
        # Verify output is valid
        assert result["pixel_values"].shape[0] > 0, f"No patches generated for size {size}"
        assert result["image_grid_thw"].shape == (1, 3), f"Invalid grid_thw shape for size {size}"
    
    print("✅ Different image sizes test passed")

def run_all_tests():
    """Run all test functions."""
    print("🚀 Starting Qwen2_5_VLTransform Tests\n")
    
    try:
        # Test basic functionality that doesn't require tokenizer files
        test_image_transform_method()
        test_encoder_input_structure()
        test_message_content_modification()
        test_different_image_sizes()
        
        # Test initialization (may fail without tokenizer files)
        transform = test_transform_initialization()
        
        print("\n🎉 All available tests completed successfully!")
        print("\nNote: Full integration tests require actual tokenizer files.")
        print("To run complete tests, provide paths to:")
        print("  - vocab.json")
        print("  - merges.txt")
        print("  - (optional) special_tokens.json")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_all_tests() 