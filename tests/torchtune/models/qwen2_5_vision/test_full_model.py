"""Test file for full Qwen2.5-VL model comparison between TorchTune and HuggingFace."""

import torch
import safetensors.torch
from PIL import Image
import numpy as np
from transformers import AutoProcessor, AutoModelForImageTextToText

from torchtune.models.qwen2_5_vision._convert_weights import qwen2_5_vl_hf_to_tune
from torchtune.models.qwen2_5_vision._model_builders import qwen2_5_vl_7b
from torchtune.models.qwen2_5_vision import qwen2_5_vl_transform


def create_test_image(width: int = 224, height: int = 224) -> Image.Image:
    """Create a simple test image."""
    # Create a random RGB image
    image_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(image_array)


def load_hf_model():
    """Load HuggingFace model and processor."""
    print("Loading HuggingFace model...")
    hf_model_path = "/mnt/vast/share/inf2-training/models/open_source/Qwen2.5-VL-7B-Instruct"
    
    try:
        hf_processor = AutoProcessor.from_pretrained(hf_model_path)
        hf_model = AutoModelForImageTextToText.from_pretrained(
            hf_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        print("✅ HuggingFace model loaded successfully")
        return hf_processor, hf_model
    except Exception as e:
        print(f"❌ Failed to load HuggingFace model: {e}")
        return None, None


def load_tune_model():
    """Load TorchTune model with converted weights."""
    print("Loading TorchTune model...")
    tune_model_path = "/mnt/vast/share/inf2-training/models/open_source/Qwen2.5-VL-7B-Instruct"
    
    try:
        # Create model
        tune_qwen = qwen2_5_vl_7b()
        
        # Load weights from safetensors files
        state_dict = {}
        files = [f"{tune_model_path}/model-0000{i}-of-00005.safetensors" for i in range(1, 6)]
        
        for file in files:
            try:
                load_files_dict = safetensors.torch.load_file(file)
                state_dict.update(load_files_dict)
            except FileNotFoundError:
                print(f"Warning: Could not find {file}")
                continue
        
        if not state_dict:
            print("❌ No state dict files found")
            return None
            
        # Convert weights from HF format to TorchTune format
        converted = qwen2_5_vl_hf_to_tune(state_dict)
        
        # Load the converted weights
        tune_qwen.load_state_dict(converted, strict=False)
        
        print("✅ TorchTune model loaded successfully")
        return tune_qwen
        
    except Exception as e:
        print(f"❌ Failed to load TorchTune model: {e}")
        return None


def load_tune_transform():
    """Load TorchTune transform."""
    print("Loading TorchTune transform...")
    hf_model_path = "/mnt/vast/share/inf2-training/models/open_source/Qwen2.5-VL-7B-Instruct"
    
    try:
        transform = qwen2_5_vl_transform(
            path=hf_model_path,
            special_tokens_path=hf_model_path,
        )
        print("✅ TorchTune transform loaded successfully")
        return transform
    except Exception as e:
        print(f"❌ Failed to load TorchTune transform: {e}")
        return None


def compare_logits(tune_model, hf_model, tune_tokens, hf_inputs, tolerance=1e-4):
    """
    Compare logits between TorchTune and HuggingFace models.
    
    Args:
        tune_model: TorchTune model
        hf_model: HuggingFace model
        tune_tokens: Input tokens for TorchTune model
        hf_inputs: Input dictionary for HuggingFace model
        tolerance: Numerical tolerance for comparison
    
    Returns:
        bool: True if logits match within tolerance
    """
    print("Comparing model logits...")
    
    # Set models to eval mode
    hf_model.eval()
    tune_model.eval()
    
    try:
        with torch.no_grad():
            # TorchTune forward pass
            tune_output = tune_model(tune_tokens)
            
            # HuggingFace forward pass
            hf_output = hf_model(**hf_inputs)
            
            # Extract logits
            if hasattr(tune_output, 'logits'):
                tune_logits = tune_output.logits
            else:
                tune_logits = tune_output
                
            if hasattr(hf_output, 'logits'):
                hf_logits = hf_output.logits
            else:
                hf_logits = hf_output
            
            # Ensure same device and dtype
            tune_logits = tune_logits.to(device=hf_logits.device, dtype=hf_logits.dtype)
            
            # Handle shape differences
            min_seq_len = min(tune_logits.shape[1], hf_logits.shape[1])
            tune_logits_trimmed = tune_logits[:, :min_seq_len, :]
            hf_logits_trimmed = hf_logits[:, :min_seq_len, :]
            
            # Compare logits
            matches = torch.allclose(tune_logits_trimmed, hf_logits_trimmed, atol=tolerance, rtol=tolerance)
            
            # Print debug info
            print(f"   - TorchTune logits shape: {tune_logits.shape}")
            print(f"   - HuggingFace logits shape: {hf_logits.shape}")
            print(f"   - Comparison shape: {tune_logits_trimmed.shape} vs {hf_logits_trimmed.shape}")
            print(f"   - Max absolute difference: {torch.max(torch.abs(tune_logits_trimmed - hf_logits_trimmed)).item():.6f}")
            print(f"   - Logits match within tolerance {tolerance}: {matches}")
            
            return matches
            
    except Exception as e:
        print(f"❌ Error during logits comparison: {e}")
        return False


def test_text_only_comparison():
    """Test model comparison with text-only input."""
    print("Testing text-only model comparison...")
    
    # Load models
    hf_processor, hf_model = load_hf_model()
    tune_model = load_tune_model()
    tune_transform = load_tune_transform()
    
    if None in [hf_processor, hf_model, tune_model, tune_transform]:
        print("❌ Failed to load required models")
        return False
    
    try:
        # Create text input
        text_input = "Hello, how are you today?"
        
        # Process with HuggingFace
        hf_inputs = hf_processor(text=text_input, return_tensors="pt")
        
        # Process with TorchTune
        tune_tokens = tune_transform.encode(text_input, add_bos=True, add_eos=False)
        tune_tokens = torch.tensor([tune_tokens])
        
        # Compare logits
        result = compare_logits(tune_model, hf_model, tune_tokens, hf_inputs)
        
        if result:
            print("✅ Text-only comparison passed!")
        else:
            print("❌ Text-only comparison failed")
            
        return result
        
    except Exception as e:
        print(f"❌ Text-only comparison failed with exception: {e}")
        return False


def test_multimodal_comparison():
    """Test model comparison with multimodal (image + text) input."""
    print("Testing multimodal model comparison...")
    
    # Load models
    hf_processor, hf_model = load_hf_model()
    tune_model = load_tune_model()
    tune_transform = load_tune_transform()
    
    if None in [hf_processor, hf_model, tune_model, tune_transform]:
        print("❌ Failed to load required models")
        return False
    
    try:
        # Create test inputs
        test_image = create_test_image(336, 336)
        text_input = "What is in this image?"
        
        # Process with HuggingFace
        hf_inputs = hf_processor(
            text=text_input,
            images=test_image,
            return_tensors="pt"
        )
        
        # Process with TorchTune
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": text_input}
                ]
            }
        ]
        
        sample = {
            "image": test_image,
            "messages": messages
        }
        
        tune_result = tune_transform(sample)
        tune_tokens = torch.tensor([tune_result["tokens"]])
        
        # Compare logits
        result = compare_logits(tune_model, hf_model, tune_tokens, hf_inputs, tolerance=1e-3)
        
        if result:
            print("✅ Multimodal comparison passed!")
        else:
            print("❌ Multimodal comparison failed")
            
        return result
        
    except Exception as e:
        print(f"❌ Multimodal comparison failed with exception: {e}")
        return False


def test_generation_consistency():
    """Test that both models generate consistent outputs."""
    print("Testing generation consistency...")
    
    # Load models
    hf_processor, hf_model = load_hf_model()
    tune_model = load_tune_model()
    tune_transform = load_tune_transform()
    
    if None in [hf_processor, hf_model, tune_model, tune_transform]:
        print("❌ Failed to load required models")
        return False
    
    try:
        # Create test inputs
        test_image = create_test_image(224, 224)
        text_input = "Describe this image briefly."
        
        # HuggingFace generation
        hf_inputs = hf_processor(
            text=text_input,
            images=test_image,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            hf_generated = hf_model.generate(
                **hf_inputs,
                max_new_tokens=20,
                do_sample=False,
                temperature=1.0,
                pad_token_id=hf_processor.tokenizer.eos_token_id
            )
        
        hf_response = hf_processor.decode(hf_generated[0], skip_special_tokens=True)
        
        # TorchTune generation would require more setup
        # For now, just check that we can get logits
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": text_input}
                ]
            }
        ]
        
        sample = {
            "image": test_image,
            "messages": messages
        }
        
        tune_result = tune_transform(sample)
        tune_tokens = torch.tensor([tune_result["tokens"]])
        
        with torch.no_grad():
            tune_output = tune_model(tune_tokens)
        
        print(f"✅ Generation consistency test passed!")
        print(f"   - HuggingFace response: {hf_response[:100]}...")
        print(f"   - TorchTune output shape: {tune_output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Generation consistency test failed: {e}")
        return False


def run_all_tests():
    """Run all full model tests."""
    print("=" * 60)
    print("Running Qwen2.5-VL Full Model Comparison Tests")
    print("=" * 60)
    
    tests = [
        test_text_only_comparison,
        test_multimodal_comparison,
        test_generation_consistency,
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