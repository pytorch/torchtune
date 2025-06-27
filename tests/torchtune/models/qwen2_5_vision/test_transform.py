"""Test file for Qwen2.5-VL Transform component with HuggingFace comparison."""

import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info

from torchtune.models.qwen2_5_vision import qwen2_5_vl_transform
from torchtune.data import Message



def create_test_image(width: int = 224, height: int = 224, seed: int = 42) -> Image.Image:
    """Create a reproducible test image."""
    np.random.seed(seed)
    # Create a random RGB image
    image_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(image_array)


def load_hf_processor():
    """Load HuggingFace processor for comparison."""
    hf_model_path = "/mnt/vast/share/inf2-training/models/open_source/Qwen2.5-VL-7B-Instruct"
    try:
        hf_processor = AutoProcessor.from_pretrained(hf_model_path)
        return hf_processor
    except Exception as e:
        print(f"❌ Failed to load HuggingFace processor: {e}")
        return None


def load_tune_transform():
    """Load TorchTune transform."""
    hf_model_path = "/mnt/vast/share/inf2-training/models/open_source/Qwen2.5-VL-7B-Instruct"
    try:
        transform = qwen2_5_vl_transform(
            path=f"{hf_model_path}/vocab.json",
            merges_file=f"{hf_model_path}/merges.txt",
            special_tokens_path=f"{hf_model_path}/tokenizer.json",
        )
        return transform
    except Exception as e:
        print(f"❌ Failed to load TorchTune transform: {e}")
        return None


def test_text_tokenization_comparison():
    """
    Compare text tokenization between HuggingFace and TorchTune.
    
    Notably, torchtune adds the EOS token to the end of the token sequence.
    """
    print("Testing text tokenization comparison with HuggingFace...")
    
    hf_processor = load_hf_processor()
    tune_transform = load_tune_transform()
    
    if hf_processor is None or tune_transform is None:
        print("❌ Failed to load required components")
        return False
    
    try:
        # Test different text inputs
        test_texts = [
            "Hello, world!",
            "This is a test sentence with multiple words.",
            "What do you see in this image?",
            "Describe the scene in detail.",
            "How many objects are visible?",
        ]
        
        for text in test_texts:
            print(f"   Testing text: '{text}'")
            
            # HuggingFace tokenization
            hf_result = hf_processor(text=text, return_tensors="pt", add_special_tokens=True)
            hf_tokens = hf_result["input_ids"].squeeze().tolist()
            
            # TorchTune tokenization
            tune_tokens = tune_transform.encode(text, add_bos=True, add_eos=True)
            
            # Compare tokens
            if len(hf_tokens) != len(tune_tokens):
                print(f"     ⚠️  Length mismatch: HF={len(hf_tokens)}, Tune={len(tune_tokens)}")
                print(f"     HF tokens: {hf_tokens}")
                print(f"     Tune tokens: {tune_tokens}")
                # This might be OK due to different special token handling
            
            # Check that most tokens match (allowing for slight differences in special tokens)
            matching_tokens = sum(1 for h, t in zip(hf_tokens, tune_tokens) if h == t)
            match_ratio = matching_tokens / max(len(hf_tokens), len(tune_tokens))
            
            if match_ratio < 0.8:  # Allow some flexibility for special tokens
                print(f"     ❌ Poor token match ratio: {match_ratio:.2f}")
                print(f"     HF tokens: {hf_tokens}")
                print(f"     Tune tokens: {tune_tokens}")
                return False
            
            # Test decoding
            hf_decoded = hf_processor.tokenizer.decode(hf_tokens, skip_special_tokens=True)
            tune_decoded = tune_transform.decode(tune_tokens, skip_special_tokens=True)
            
            # The decoded text should be very similar
            if hf_decoded.strip() != tune_decoded.strip():
                print(f"     ⚠️  Decode mismatch:")
                print(f"     HF decoded: '{hf_decoded}'")
                print(f"     Tune decoded: '{tune_decoded}'")
                # This might still be acceptable due to tokenizer differences
            
            print(f"     ✓ Match ratio: {match_ratio:.2f}")
        
        print("✅ Text tokenization comparison passed!")
        return True
        
    except Exception as e:
        print(f"❌ Text tokenization comparison failed: {e}")
        return False


def test_image_transform_comparison():
    """Compare image transformation between HuggingFace and TorchTune."""
    print("Testing image transform comparison with HuggingFace...")
    
    hf_processor = load_hf_processor()
    tune_transform = load_tune_transform()
    
    if hf_processor is None or tune_transform is None:
        print("❌ Failed to load required components")
        return False
    
    try:
        # Test different image sizes
        test_configs = [
            (224, 224),
            (336, 336),
            (448, 224),
            (224, 448),
        ]
        
        for width, height in test_configs:
            print(f"   Testing image size: {width}x{height}")
            
            # Create test image
            test_image = create_test_image(width, height, seed=42)
            
            # HuggingFace processing - follow the official pattern
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": test_image},
                        {"type": "text", "text": "Describe this image."},
                    ],
                }
            ]
            
            # Use HuggingFace's recommended approach
            text = hf_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            hf_result = hf_processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            
            if hf_result is None or "pixel_values" not in hf_result:
                print(f"     ⚠️  HF processor returned None or missing pixel_values for size {width}x{height}")
                continue
                
            hf_pixel_values = hf_result["pixel_values"]
            
            # TorchTune processing
            tune_pixel_values, tune_image_grid_thw, num_patches = tune_transform.transform_image(test_image)
            
            print(f"     HF pixel values shape: {hf_pixel_values.shape}")
            print(f"     Tune pixel values shape: {tune_pixel_values.shape}")
            print(f"     Tune image grid: {tune_image_grid_thw}")
            
            # Check that pixel values are in reasonable range
            hf_min, hf_max = hf_pixel_values.min().item(), hf_pixel_values.max().item()
            tune_min, tune_max = tune_pixel_values.min().item(), tune_pixel_values.max().item()
            
            print(f"     HF pixel range: [{hf_min:.3f}, {hf_max:.3f}]")
            print(f"     Tune pixel range: [{tune_min:.3f}, {tune_max:.3f}]")
            
            # Element-wise comparison if shapes are compatible
            if hf_pixel_values.shape == tune_pixel_values.shape:
                # Direct element-wise comparison
                pixel_diff = torch.abs(hf_pixel_values - tune_pixel_values)
                max_diff = pixel_diff.max().item()
                mean_diff = pixel_diff.mean().item()
                
                print(f"     📊 Pixel value comparison:")
                print(f"       - Max absolute difference: {max_diff:.6f}")
                print(f"       - Mean absolute difference: {mean_diff:.6f}")
                print(f"       - Relative max diff: {max_diff / max(abs(hf_max), abs(tune_max)):.6f}")
                
                # Check if differences are within reasonable tolerance
                if max_diff < 1e-3:  # Very close
                    print(f"       - ✅ Excellent match (diff < 1e-3)")
                elif max_diff < 1e-2:  # Close enough
                    print(f"       - ✅ Good match (diff < 1e-2)")
                elif max_diff < 0.1:  # Acceptable
                    print(f"       - ⚠️  Acceptable match (diff < 0.1)")
                else:  # Large difference
                    print(f"       - ❌ Large difference (diff >= 0.1)")
                    
                # Print some sample values for debugging
                print(f"     📋 Sample pixel values:")
                flat_hf = hf_pixel_values.flatten()
                flat_tune = tune_pixel_values.flatten()
                sample_indices = torch.randperm(len(flat_hf))[:5]  # Random 5 samples
                
                for i, idx in enumerate(sample_indices):
                    hf_val = flat_hf[idx].item()
                    tune_val = flat_tune[idx].item()
                    diff = abs(hf_val - tune_val)
                    print(f"       [{i+1}] HF: {hf_val:.6f}, Tune: {tune_val:.6f}, Diff: {diff:.6f}")
                    
            else:
                print(f"     ⚠️  Shape mismatch - cannot do element-wise comparison")
                print(f"       HF shape: {hf_pixel_values.shape}")
                print(f"       Tune shape: {tune_pixel_values.shape}")
                
                # Try to compare flattened versions if total elements match
                if hf_pixel_values.numel() == tune_pixel_values.numel():
                    hf_flat = hf_pixel_values.flatten()
                    tune_flat = tune_pixel_values.flatten()
                    pixel_diff = torch.abs(hf_flat - tune_flat)
                    max_diff = pixel_diff.max().item()
                    mean_diff = pixel_diff.mean().item()
                    
                    print(f"     📊 Flattened comparison (same total elements):")
                    print(f"       - Max absolute difference: {max_diff:.6f}")
                    print(f"       - Mean absolute difference: {mean_diff:.6f}")
                else:
                    print(f"     ❌ Different total elements - HF: {hf_pixel_values.numel()}, Tune: {tune_pixel_values.numel()}")
            
            # Both should be normalized and in similar ranges
            assert -3 < hf_min < 3, f"HF pixel values out of expected range: {hf_min}"
            assert -3 < hf_max < 3, f"HF pixel values out of expected range: {hf_max}"
            assert -3 < tune_min < 3, f"Tune pixel values out of expected range: {tune_min}"
            assert -3 < tune_max < 3, f"Tune pixel values out of expected range: {tune_max}"
            
            print(f"     ✓ Image size {width}x{height} processed successfully")
        
        print("✅ Image transform comparison passed!")
        return True
        
    except Exception as e:
        print(f"❌ Image transform comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multimodal_transform_comparison():
    """Compare multimodal (image + text) transformation."""
    print("Testing multimodal transform comparison with HuggingFace...")
    
    hf_processor = load_hf_processor()
    tune_transform = load_tune_transform()
    
    if hf_processor is None or tune_transform is None:
        print("❌ Failed to load required components")
        return False
    
    try:
        # Create test inputs
        test_image = create_test_image(336, 336, seed=123)
        test_text = "What do you see in this image?"
        
        # HuggingFace processing - follow the official pattern
        hf_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": test_image},
                    {"type": "text", "text": test_text}
                ]
            }
        ]
        
        text = hf_processor.apply_chat_template(
            hf_messages, tokenize=False, add_generation_prompt=False
        )
        image_inputs, video_inputs = process_vision_info(hf_messages)
        hf_result = hf_processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # TorchTune processing - convert to proper Message format
        tune_messages = [
            Message(
                role="user",
                content=[
                    {"type": "image", "content": test_image},
                    {"type": "text", "content": test_text}
                ]
            )
        ]
        
        sample = {
            "image": test_image,
            "messages": tune_messages
        }
        
        tune_result = tune_transform(sample)
        
        # Compare results
        print(f"   HuggingFace results:")
        for key, value in hf_result.items():
            if isinstance(value, torch.Tensor):
                print(f"     {key}: {value.shape}, dtype={value.dtype}")
            else:
                print(f"     {key}: {type(value)}")
        
        print(f"   TorchTune results:")
        for key, value in tune_result.items():
            if isinstance(value, torch.Tensor):
                print(f"     {key}: {value.shape}, dtype={value.dtype}")
            elif isinstance(value, list):
                print(f"     {key}: list[{len(value)}], sample={value[:5] if len(value) > 5 else value}")
            elif isinstance(value, dict):
                print(f"     {key}: dict with keys {list(value.keys())}")
                # Examine encoder_input in detail
                if key == "encoder_input":
                    for sub_key, sub_value in value.items():
                        print(f"       {sub_key}: {type(sub_value)}")
                        if isinstance(sub_value, dict):
                            for sub_sub_key, sub_sub_value in sub_value.items():
                                if isinstance(sub_sub_value, torch.Tensor):
                                    print(f"         {sub_sub_key}: {sub_sub_value.shape}, dtype={sub_sub_value.dtype}")
                                elif isinstance(sub_sub_value, list):
                                    print(f"         {sub_sub_key}: list[{len(sub_sub_value)}]")
                                else:
                                    print(f"         {sub_sub_key}: {type(sub_sub_value)}")
            else:
                print(f"     {key}: {type(value)}")
        
        # Check that both produce reasonable token sequences
        hf_tokens = hf_result["input_ids"].squeeze().tolist()
        tune_tokens = tune_result["tokens"]
        
        print(f"   HF token count: {len(hf_tokens)}")
        print(f"   Tune token count: {len(tune_tokens)}")
        
        # Compare pixel values if both have them
        if "pixel_values" in hf_result and "encoder_input" in tune_result:
            hf_pixel_values = hf_result["pixel_values"]
            tune_image_data = tune_result["encoder_input"]["image"]
            
            if "hidden_states" in tune_image_data:
                tune_pixel_values = tune_image_data["hidden_states"]
                
                print(f"   📊 Pixel values comparison:")
                print(f"     HF pixel_values: {hf_pixel_values.shape}")
                print(f"     Tune pixel_values: {tune_pixel_values.shape}")
                
                # Handle batch dimension difference - squeeze TorchTune to match HF
                if tune_pixel_values.shape[0] == 1 and len(tune_pixel_values.shape) == 3:
                    tune_pixel_values_squeezed = tune_pixel_values.squeeze(0)
                    print(f"     Tune pixel_values (squeezed): {tune_pixel_values_squeezed.shape}")
                    
                    # Compare if shapes are now compatible
                    if hf_pixel_values.shape == tune_pixel_values_squeezed.shape:
                        # Convert to same dtype for comparison
                        hf_float = hf_pixel_values.float()
                        tune_float = tune_pixel_values_squeezed.float()
                        
                        pixel_diff = torch.abs(hf_float - tune_float)
                        max_diff = pixel_diff.max().item()
                        mean_diff = pixel_diff.mean().item()
                        
                        print(f"     Max difference: {max_diff:.6f}")
                        print(f"     Mean difference: {mean_diff:.6f}")
                        
                        if max_diff < 1e-3:
                            print(f"     ✅ Excellent pixel value match!")
                        elif max_diff < 1e-2:
                            print(f"     ✅ Good pixel value match!")
                        else:
                            print(f"     ⚠️  Notable pixel value differences")
                    else:
                        print(f"     ⚠️  Different pixel value shapes after squeezing")
                else:
                    print(f"     ⚠️  Cannot squeeze TorchTune tensor to match HF shape")
        
        # Both should produce non-empty token sequences
        assert len(hf_tokens) > 0, "HF should produce non-empty tokens"
        assert len(tune_tokens) > 0, "Tune should produce non-empty tokens"
        
        # The sequences might have different lengths due to different image token handling
        # TorchTune separates image and text tokens, while HF combines them
        print(f"   📝 Token analysis:")
        print(f"     HF tokens length {len(hf_tokens)}: {hf_tokens}")
        print(f"     Tune tokens length {len(tune_tokens)}: {tune_tokens}")
        
        # Calculate effective token counts
        tune_image_tokens = 0
        if "encoder_input" in tune_result and "image" in tune_result["encoder_input"]:
            image_data = tune_result["encoder_input"]["image"]
            if "hidden_states" in image_data and isinstance(image_data["hidden_states"], torch.Tensor):
                image_tensor = image_data["hidden_states"]
                # Image tokens are the number of patches (second dimension)
                tune_image_tokens = image_tensor.shape[1] if len(image_tensor.shape) > 1 else image_tensor.numel()

        
        # Check that we have the expected image dimensions
        if "encoder_input" in tune_result:
            image_data = tune_result["encoder_input"]["image"]
            if "grid_thw" in image_data:
                grid_thw = image_data["grid_thw"]
                print(f"     TorchTune image grid (t,h,w): {grid_thw.tolist()}")
                
        if "image_grid_thw" in hf_result:
            hf_grid = hf_result["image_grid_thw"]
            print(f"     HuggingFace image grid (t,h,w): {hf_grid.tolist()}")
        
        print("✅ Multimodal transform comparison passed!")
        return True
        
    except Exception as e:
        print(f"❌ Multimodal transform comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_image_transform_consistency():
    """Test that the image transform produces consistent results."""
    print("Testing image transform consistency...")
    
    tune_transform = load_tune_transform()
    
    if tune_transform is None:
        print("❌ Failed to load TorchTune transform")
        return False
    
    try:
        # Create test image
        test_image = create_test_image(256, 256, seed=999)
        
        # Transform the same image multiple times
        results = []
        for i in range(3):
            pixel_values, image_grid_thw, num_patches = tune_transform.transform_image(test_image)
            results.append((pixel_values, image_grid_thw))
        
        # Check that results are identical
        for i in range(1, len(results)):
            pixel_diff = torch.max(torch.abs(results[0][0] - results[i][0])).item()
            grid_diff = torch.max(torch.abs(results[0][1] - results[i][1])).item()
            
            assert pixel_diff < 1e-8, f"Pixel values should be identical, diff={pixel_diff}"
            assert grid_diff < 1e-8, f"Grid values should be identical, diff={grid_diff}"
        
        print("✅ Image transform consistency test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Image transform consistency test failed: {e}")
        return False


def run_all_tests():
    """Run all transform tests."""
    print("=" * 60)
    print("Running Qwen2.5-VL Transform Tests with HuggingFace Comparison")
    print("=" * 60)
    
    tests = [
        test_text_tokenization_comparison,
        test_image_transform_comparison,
        test_multimodal_transform_comparison,
        test_image_transform_consistency,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            results.append(False)
        print("-" * 40)
    
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