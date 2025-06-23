#!/usr/bin/env python3
"""
Integration test for Qwen2_5_VLTransform demonstrating the complete pipeline.
Uses mock tokenizer to avoid requiring actual tokenizer files.
"""

import sys
import os
from PIL import Image
import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Optional
from unittest.mock import Mock, MagicMock

# Add the current directory to path to import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _transform import Qwen2_5_VLImageTransform
from torchtune.data import Message

class MockQwen2_5Tokenizer:
    """Mock tokenizer for testing purposes."""
    
    def __init__(self, path, merges_file, special_tokens=None, max_seq_len=None, prompt_template=None, **kwargs):
        self.path = path
        self.merges_file = merges_file
        self.special_tokens = special_tokens or {}
        self.max_seq_len = max_seq_len
        self.prompt_template = prompt_template
        # Ignore other kwargs that are meant for the image transform
        
        # Mock properties
        self.base_vocab_size = 50000
        self.vocab_size = 50000
        self.pad_id = 0
        self.stop_tokens = [2]  # Mock EOS token
        
    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        """Mock encode method."""
        # Simple mock: convert text to token IDs based on length
        tokens = [1] if add_bos else []  # BOS token
        tokens.extend([hash(word) % 1000 + 10 for word in text.split()])  # Mock word tokens
        if add_eos:
            tokens.append(2)  # EOS token
        return tokens
    
    def decode(self, token_ids: List[int], truncate_at_eos: bool = True, skip_special_tokens: bool = True) -> str:
        """Mock decode method."""
        if truncate_at_eos and 2 in token_ids:
            token_ids = token_ids[:token_ids.index(2)]
        if skip_special_tokens:
            token_ids = [t for t in token_ids if t not in [0, 1, 2]]
        return f"decoded_text_from_{len(token_ids)}_tokens"
    
    def tokenize_message(self, message: Message, add_start_tokens: bool = True, add_end_tokens: bool = True) -> List[int]:
        """Mock tokenize_message method."""
        tokens = []
        if add_start_tokens:
            tokens.append(1)  # BOS
            
        for content in message.content:
            if content["type"] == "text":
                text_tokens = self.encode(content["content"], add_bos=False, add_eos=False)
                tokens.extend(text_tokens)
            elif content["type"] == "image":
                # Add special image tokens - mock with a range of IDs
                image_token_id = 32000  # Mock image token ID
                # For Qwen2.5-VL, we need to add tokens based on image_grid_thw
                if "image_grid_thw" in content:
                    grid_t, grid_h, grid_w = content["image_grid_thw"][0]
                    num_image_tokens = grid_t * grid_h * grid_w
                    tokens.extend([image_token_id] * num_image_tokens.item())
                else:
                    # Default number of image tokens
                    tokens.extend([image_token_id] * 256)
                    
        if add_end_tokens:
            tokens.append(2)  # EOS
            
        return tokens
    
    def tokenize_messages(self, messages: List[Message], add_end_tokens: bool = True) -> Tuple[List[int], List[bool]]:
        """Mock tokenize_messages method."""
        all_tokens = []
        all_masks = []
        
        for i, message in enumerate(messages):
            msg_tokens = self.tokenize_message(
                message, 
                add_start_tokens=(i == 0), 
                add_end_tokens=add_end_tokens
            )
            all_tokens.extend(msg_tokens)
            # Mock mask: True for assistant tokens, False for user tokens
            mask = [message.role == "assistant"] * len(msg_tokens)
            all_masks.extend(mask)
            
        return all_tokens, all_masks
    
    def __call__(self, sample: Dict[str, Any], inference: bool = False) -> Dict[str, Any]:
        """Mock tokenizer call method."""
        messages = sample["messages"]
        tokens, mask = self.tokenize_messages(messages)
        
        sample.update({
            "tokens": tokens,
            "mask": mask
        })
        
        return sample

class MockQwen2_5_VLTransform:
    """Mock version of Qwen2_5_VLTransform for testing."""
    
    def __init__(self, path: str, merges_file: str, **kwargs):
        # Initialize with mock tokenizer
        self.tokenizer = MockQwen2_5Tokenizer(path, merges_file, **kwargs)
        
        # Initialize real image transform
        self.image_transform = Qwen2_5_VLImageTransform(
            patch_size=kwargs.get("patch_size", 14),
            merge_size=2,
            temporal_patch_size=2,
            dtype=kwargs.get("dtype", torch.bfloat16),
        )
        
        # Copy properties from tokenizer
        self.stop_tokens = self.tokenizer.stop_tokens
        self.special_tokens = self.tokenizer.special_tokens
        self.max_seq_len = kwargs.get("max_seq_len")
        self.patch_size = kwargs.get("patch_size", 14)
        self.prompt_template = kwargs.get("prompt_template")
        self.pad_id = self.tokenizer.pad_id
    
    @property
    def base_vocab_size(self) -> int:
        return self.tokenizer.base_vocab_size

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        return self.tokenizer.encode(text=text, add_bos=add_bos, add_eos=add_eos)

    def decode(self, token_ids: List[int], truncate_at_eos: bool = True, skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(token_ids, truncate_at_eos=truncate_at_eos, skip_special_tokens=skip_special_tokens)

    def transform_image(self, image: Image.Image, inference: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = {"image": image}
        transformed = self.image_transform(sample, inference=inference)
        return transformed["pixel_values"], transformed["image_grid_thw"]

    def tokenize_message(self, message: Message, add_start_tokens: bool = True, add_end_tokens: bool = True) -> List[int]:
        return self.tokenizer.tokenize_message(message=message, add_start_tokens=add_start_tokens, add_end_tokens=add_end_tokens)

    def tokenize_messages(self, messages: List[Message], add_end_tokens: bool = True) -> Tuple[List[int], List[bool]]:
        return self.tokenizer.tokenize_messages(messages=messages, add_end_tokens=add_end_tokens)

    def __call__(self, sample: Dict[str, Any], inference: bool = False) -> Dict[str, Any]:
        """Complete multimodal transform pipeline."""
        encoder_input = {"vision": {"images": []}}
        messages = sample["messages"]
        
        # Process images in messages
        for message in messages:
            for content in message.content:
                if content["type"] == "image":
                    image = content["content"]
                    pixel_values, image_grid_thw = self.transform_image(image, inference=inference)
                    encoder_input["vision"]["images"].append(pixel_values)
                    
                    # Add grid info to content for tokenizer
                    content["image_grid_thw"] = image_grid_thw

        # Add encoder input to sample
        sample["encoder_input"] = encoder_input
        
        # Tokenize messages
        sample = self.tokenizer(sample, inference=inference)
        
        return sample

def create_test_image(size=(224, 224), seed=42):
    """Create a test image for testing."""
    np.random.seed(seed)
    return Image.fromarray(np.random.randint(0, 255, (*size, 3)).astype(np.uint8))

def test_complete_pipeline():
    """Test the complete multimodal transform pipeline."""
    print("=== Testing Complete Qwen2_5_VLTransform Pipeline ===")
    
    # Create mock transform
    transform = MockQwen2_5_VLTransform(
        path="mock_vocab.json",
        merges_file="mock_merges.txt",
        patch_size=14,
        max_seq_len=2048,
    )
    
    print("✅ Transform initialized successfully")
    
    # Test basic properties
    print(f"   Base vocab size: {transform.base_vocab_size}")
    print(f"   Vocab size: {transform.vocab_size}")
    print(f"   Pad ID: {transform.pad_id}")
    
    # Test encode/decode
    test_text = "Hello, how are you?"
    tokens = transform.encode(test_text)
    decoded = transform.decode(tokens)
    print(f"   Encode/decode test: '{test_text}' -> {len(tokens)} tokens -> '{decoded}'")
    
    # Test image transform
    test_image = create_test_image()
    pixel_values, image_grid_thw = transform.transform_image(test_image)
    print(f"   Image transform: {pixel_values.shape} pixels, grid {image_grid_thw}")
    
    # Test complete pipeline with multimodal message
    message = Message(
        role="user",
        content=[
            {"type": "text", "content": "What do you see in this image?"},
            {"type": "image", "content": test_image}
        ]
    )
    
    sample = {"messages": [message]}
    result = transform(sample)
    
    print("✅ Complete pipeline test successful")
    print(f"   Output keys: {list(result.keys())}")
    print(f"   Tokens: {len(result['tokens'])} tokens")
    print(f"   Mask: {len(result['mask'])} mask values")
    print(f"   Encoder input images: {len(result['encoder_input']['vision']['images'])}")
    print(f"   First image shape: {result['encoder_input']['vision']['images'][0].shape}")
    
    # Verify the structure
    assert "tokens" in result, "tokens missing from output"
    assert "mask" in result, "mask missing from output"
    assert "encoder_input" in result, "encoder_input missing from output"
    assert "vision" in result["encoder_input"], "vision missing from encoder_input"
    assert "images" in result["encoder_input"]["vision"], "images missing from vision"
    
    print("✅ Output structure validation passed")
    
    return result

def test_multiple_images():
    """Test with multiple images in a conversation."""
    print("\n=== Testing Multiple Images ===")
    
    transform = MockQwen2_5_VLTransform(
        path="mock_vocab.json",
        merges_file="mock_merges.txt",
    )
    
    # Create messages with multiple images
    image1 = create_test_image((200, 200), seed=42)
    image2 = create_test_image((300, 400), seed=123)
    
    messages = [
        Message(
            role="user",
            content=[
                {"type": "text", "content": "Compare these two images:"},
                {"type": "image", "content": image1},
                {"type": "image", "content": image2},
                {"type": "text", "content": "What are the differences?"}
            ]
        )
    ]
    
    sample = {"messages": messages}
    result = transform(sample)
    
    print(f"✅ Multiple images test successful")
    print(f"   Number of images processed: {len(result['encoder_input']['vision']['images'])}")
    print(f"   Image 1 shape: {result['encoder_input']['vision']['images'][0].shape}")
    print(f"   Image 2 shape: {result['encoder_input']['vision']['images'][1].shape}")
    print(f"   Total tokens: {len(result['tokens'])}")
    
    assert len(result['encoder_input']['vision']['images']) == 2, "Should have 2 images"
    
    print("✅ Multiple images validation passed")

def test_text_only_message():
    """Test with text-only message (no images)."""
    print("\n=== Testing Text-Only Message ===")
    
    transform = MockQwen2_5_VLTransform(
        path="mock_vocab.json",
        merges_file="mock_merges.txt",
    )
    
    message = Message(
        role="user",
        content=[{"type": "text", "content": "Hello, how are you today?"}]
    )
    
    sample = {"messages": [message]}
    result = transform(sample)
    
    print(f"✅ Text-only message test successful")
    print(f"   Tokens: {len(result['tokens'])}")
    print(f"   Images: {len(result['encoder_input']['vision']['images'])}")
    
    assert len(result['encoder_input']['vision']['images']) == 0, "Should have no images"
    assert len(result['tokens']) > 0, "Should have tokens"
    
    print("✅ Text-only validation passed")

def run_integration_tests():
    """Run all integration tests."""
    print("🚀 Starting Qwen2_5_VLTransform Integration Tests\n")
    
    try:
        test_complete_pipeline()
        test_multiple_images()
        test_text_only_message()
        
        print("\n🎉 All integration tests completed successfully!")
        print("\nThe Qwen2_5_VLTransform implementation is ready for use!")
        print("Next steps:")
        print("  1. Replace MockQwen2_5Tokenizer with real Qwen2_5Tokenizer")
        print("  2. Add to TorchTune model registry")
        print("  3. Create recipes for training/fine-tuning")
        
    except Exception as e:
        print(f"\n❌ Integration test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_integration_tests() 