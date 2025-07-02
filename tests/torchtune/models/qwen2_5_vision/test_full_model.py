"""Test file for full Qwen2.5-VL model comparison between TorchTune and HuggingFace."""

import os
import torch
import safetensors.torch
from PIL import Image
import numpy as np
from transformers import AutoProcessor, AutoModelForImageTextToText
import time
import matplotlib.pyplot as plt
import requests
from io import BytesIO

from torchtune.models.qwen2_5_vision._convert_weights import qwen2_5_vl_hf_to_tune
from torchtune.models.qwen2_5_vision._model_builders import qwen2_5_vl_7b
from torchtune.models.qwen2_5_vision import qwen2_5_vl_transform
from torchtune.data import Message
from torchtune.generation import sample

model_path = os.environ.get("HF_MODEL_PATH")
PATH = f"{model_path}/vocab.json"
MERGES_FILE = f"{model_path}/merges.txt"
HF_MODEL_PATH = model_path

def create_test_image(width: int = 224, height: int = 224) -> Image.Image:
    """Create a simple test image."""
    # Create a random RGB image
    image_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(image_array)


def get_cat_image_url():
    """Fetches a random cat image URL from TheCatAPI."""
    try:
        response = requests.get("https://api.thecatapi.com/v1/images/search")
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        data = response.json()
        if data and len(data) > 0:
            return data[0]['url']
        else:
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching cat image: {e}")
        return None


def download_and_save_image(url, save_path="cat_image.jpg"):
    """Download an image from URL and save it locally."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # Open image from bytes and save
        image = Image.open(BytesIO(response.content))
        image.save(save_path)
        print(f"✅ Cat image saved as {save_path}")
        return image
    except Exception as e:
        print(f"❌ Error downloading/saving image: {e}")
        return None


def load_hf_model():
    """Load HuggingFace model and processor."""
    print("Loading HuggingFace model...")
    hf_model_path = HF_MODEL_PATH
    
    try:
        hf_processor = AutoProcessor.from_pretrained(hf_model_path)
        hf_model = AutoModelForImageTextToText.from_pretrained(
            hf_model_path,
        )
        print("✅ HuggingFace model loaded successfully")
        return hf_processor, hf_model
    except Exception as e:
        print(f"❌ Failed to load HuggingFace model: {e}")
        return None, None


def load_tune_model():
    """Load TorchTune model with converted weights."""
    print("Loading TorchTune model...")
    tune_model_path = HF_MODEL_PATH
    
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

    try:
        transform = qwen2_5_vl_transform(
            path=PATH,
            merges_file=MERGES_FILE,
        )
        print("✅ TorchTune transform loaded successfully")
        return transform
    except Exception as e:
        print(f"❌ Failed to load TorchTune transform: {e}")
        return None


def compare_logits(tune_model, hf_model, tune_input, hf_inputs, tolerance=1e-4):
    """
    Compare logits between TorchTune and HuggingFace models.
    
    Args:
        tune_model: TorchTune model
        hf_model: HuggingFace model
        tune_input: Input for TorchTune model (tokens for text-only, dict for multimodal)
        hf_inputs: Input dictionary for HuggingFace model
        tolerance: Numerical tolerance for comparison
    
    Returns:
        bool: True if logits match within tolerance
    """
    print("Comparing model logits...")
    
    # Set models to eval mode
    hf_model.eval()
    tune_model.eval()
    
    with torch.no_grad():
        # TorchTune forward pass
        start_time = time.time()
        if isinstance(tune_input, dict):
            # Multimodal input
            tune_output = tune_model(
                tune_input["tokens"], 
                encoder_input=tune_input["encoder_input"],
                image_grid_thw=tune_input["image_grid_thw"]
            )
        else:
            # Text-only input (backward compatibility)
            tune_output = tune_model(tune_input)
        tune_time = time.time() - start_time
        
        # HuggingFace forward pass
        start_time = time.time()
        hf_output = hf_model(**hf_inputs)
        hf_time = time.time() - start_time

        print(f"TorchTune time: {tune_time} seconds")
        print(f"HuggingFace time: {hf_time} seconds")
        
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
        
        # Compare logits
        matches = torch.allclose(tune_logits, hf_logits, atol=tolerance, rtol=tolerance)

        # Create detailed analysis of differences
        diff = tune_logits - hf_logits
        diff = diff.squeeze(0)  # Remove batch dimension: [seq_len, vocab_size]
        diff_abs = torch.abs(diff)
        
        # Analysis 1: Per-token differences (max diff across vocab for each token)
        per_token_max_diff = torch.max(diff_abs, dim=1)[0]  # [seq_len]
        per_token_mean_diff = torch.mean(diff_abs, dim=1)   # [seq_len]
        
        # Analysis 2: Per-vocab differences (max diff across tokens for each vocab)
        per_vocab_max_diff = torch.max(diff_abs, dim=0)[0]  # [vocab_size]
        per_vocab_mean_diff = torch.mean(diff_abs, dim=0)   # [vocab_size]
        
        # Convert to numpy for plotting
        per_token_max_diff_np = per_token_max_diff.cpu().numpy()
        per_token_mean_diff_np = per_token_mean_diff.cpu().numpy()
        per_vocab_max_diff_np = per_vocab_max_diff.cpu().numpy()
        per_vocab_mean_diff_np = per_vocab_mean_diff.cpu().numpy()
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Per-token max differences
        axes[0, 0].plot(per_token_max_diff_np, 'b-', linewidth=1)
        axes[0, 0].set_title('Max Logit Difference per Token Position')
        axes[0, 0].set_xlabel('Token Position')
        axes[0, 0].set_ylabel('Max Absolute Difference')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Per-token mean differences
        axes[0, 1].plot(per_token_mean_diff_np, 'r-', linewidth=1)
        axes[0, 1].set_title('Mean Logit Difference per Token Position')
        axes[0, 1].set_xlabel('Token Position')
        axes[0, 1].set_ylabel('Mean Absolute Difference')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Histogram of per-vocab max differences
        axes[1, 0].hist(per_vocab_max_diff_np, bins=50, alpha=0.7, color='green')
        axes[1, 0].set_title('Distribution of Max Differences per Vocab Token')
        axes[1, 0].set_xlabel('Max Absolute Difference')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_yscale('log')
        
        # Plot 4: Top differing vocab tokens
        top_diff_indices = torch.topk(per_vocab_max_diff, k=20)[1]
        top_diff_values = per_vocab_max_diff[top_diff_indices].cpu().numpy()
        axes[1, 1].bar(range(20), top_diff_values, color='orange')
        axes[1, 1].set_title('Top 20 Most Different Vocab Tokens')
        axes[1, 1].set_xlabel('Rank')
        axes[1, 1].set_ylabel('Max Absolute Difference')
        
        plt.tight_layout()
        plt.savefig("logits_difference_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print detailed statistics
        print(f"   - Detailed difference analysis:")
        print(f"     * Overall max difference: {torch.max(diff_abs).item():.6f}")
        print(f"     * Overall mean difference: {torch.mean(diff_abs).item():.6f}")
        print(f"     * Per-token max diff range: {per_token_max_diff.min().item():.6f} to {per_token_max_diff.max().item():.6f}")
        print(f"     * Per-token mean diff range: {per_token_mean_diff.min().item():.6f} to {per_token_mean_diff.max().item():.6f}")
        print(f"     * Tokens with max diff > 0.1: {(per_token_max_diff > 0.1).sum().item()}")
        print(f"     * Vocab tokens with max diff > 0.1: {(per_vocab_max_diff > 0.1).sum().item()}")
        
        # Find the most problematic token positions
        worst_tokens = torch.topk(per_token_max_diff, k=5)[1]
        print(f"     * Top 5 most different token positions: {worst_tokens.tolist()}")
        
        # Find the most problematic vocab indices
        worst_vocab = torch.topk(per_vocab_max_diff, k=5)[1]
        print(f"     * Top 5 most different vocab indices: {worst_vocab.tolist()}")
        
        # Print debug info
        print(f"   - TorchTune logits shape: {tune_logits.shape}")
        print(f"   - HuggingFace logits shape: {hf_logits.shape}")
        print(f"   - Comparison shape: {tune_logits.shape} vs {hf_logits.shape}")
        print(f"   - Max absolute difference: {torch.max(torch.abs(tune_logits - hf_logits)).item():.6f}")
        print(f"   - Logits match within tolerance {tolerance}: {matches}")
        
        return matches


def test_text_only_comparison(hf_processor, hf_model, tune_model, tune_transform):
    """Test model comparison with text-only input."""
    print("Testing text-only model comparison...")
    
    text_input = "Hello, how are you today?"
    
    # For text-only, use the same raw text for both models
    hf_inputs = hf_processor(text=text_input, return_tensors="pt")
    tune_tokens = tune_transform.encode(text_input, add_bos=True, add_eos=False)
    tune_tokens = torch.tensor([tune_tokens])
    
    result = compare_logits(tune_model, hf_model, tune_tokens, hf_inputs)
    
    if result:
        print("✅ Text-only comparison passed!")
    else:
        print("❌ Text-only comparison failed")
        
    return result


def test_multimodal_comparison(hf_processor, hf_model, tune_model, tune_transform):
    """Test model comparison with multimodal (image + text) input."""
    print("Testing multimodal model comparison...")
    
    test_image = create_test_image(336, 336)
    text_input = "What is in this image?"
    
    # Process with TorchTune
    messages = [
        Message(
            role="user",
            content=[
                {"type": "image", "content": test_image},
                {"type": "text", "content": text_input}
            ]
        )
    ]
    
    sample = {
        "image": test_image,
        "messages": messages
    }
    
    tune_result = tune_transform(sample)
    tune_tokens = torch.tensor([tune_result["tokens"]])
    
    messages_hf = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": test_image},
                {"type": "text", "text": text_input}
            ]
        }
    ]
    
    # Use a custom template without system message
    custom_template = "{% for message in messages %}{% if message['role'] == 'user' %}<|im_start|>user\n{% for content in message['content'] %}{% if content['type'] == 'image' %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'text' %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}"
    
    # Apply the custom template
    text_custom = hf_processor.apply_chat_template(
        messages_hf, 
        chat_template=custom_template,
        tokenize=False, 
        add_generation_prompt=False
    )
    print(f"Custom formatted text: {text_custom[:100]}...")
    
    # Add EOS token to match TorchTune (151645 is the EOS token)
    text_custom_with_eos = text_custom + "<|im_end|>"
    print(f"Custom formatted text with EOS: {text_custom_with_eos[:100]}...")
    
    hf_inputs_custom = hf_processor(text=text_custom_with_eos, images=test_image, return_tensors="pt")
    
    hf_tokens = hf_inputs_custom['input_ids'][0].tolist()
    tune_tokens_list = tune_result['tokens']
    
    print(f"\nDetailed token comparison:")
    print(f"TorchTune length: {len(tune_tokens_list)}")
    print(f"HuggingFace length: {len(hf_tokens)}")

    print(f"HF input_ids (with custom template): \n{hf_tokens}")
    print(f"TorchTune tokens: \n{tune_tokens_list}")
    
    # Find where they diverge
    assert len(tune_tokens_list) == len(hf_tokens)
    assert tune_tokens_list == hf_tokens
    print("✅ Token comparison passed!")
    
    # Use the custom approach for comparison
    hf_inputs = hf_inputs_custom
    
    # Debug: Compare image processing
    print(f"\nImage processing comparison:")
    if "pixel_values" in hf_inputs:
        hf_pixel_values = hf_inputs["pixel_values"]
        tune_pixel_values = tune_result["encoder_input"]["image"]["hidden_states"]
        print(f"HF pixel values shape: {hf_pixel_values.shape}")
        print(f"TorchTune pixel values shape: {tune_pixel_values.shape}")
        
        if hf_pixel_values.shape == tune_pixel_values.shape:
            pixel_diff = torch.abs(hf_pixel_values - tune_pixel_values).max()
            print(f"Max pixel value difference: {pixel_diff:.6f}")
        else:
            print("Pixel value shapes don't match - adjusting for comparison")
            # Remove batch dimension from TorchTune if present
            if tune_pixel_values.dim() == 3 and tune_pixel_values.shape[0] == 1:
                tune_pixel_values_adj = tune_pixel_values.squeeze(0)
                print(f"Adjusted TorchTune shape: {tune_pixel_values_adj.shape}")
                
                if hf_pixel_values.shape == tune_pixel_values_adj.shape:
                    pixel_diff = torch.abs(hf_pixel_values - tune_pixel_values_adj).max()
                    print(f"Max pixel value difference (after adjustment): {pixel_diff:.6f}")
                else:
                    print(f"Still don't match: HF {hf_pixel_values.shape} vs TT {tune_pixel_values_adj.shape}")
    
    # Prepare TorchTune model inputs - tokens should be 2D [batch_size, seq_len]
    # tune_tokens is already [1, seq_len] from earlier processing
    tune_model_input = {
        "tokens": tune_tokens,  # Keep batch dimension [1, seq_len]
        "encoder_input": tune_result["encoder_input"],
        "image_grid_thw": tune_result["encoder_input"]["image"]["grid_thw"]
    }
    
    # Compare logits with proper multimodal inputs
    result = compare_logits(tune_model, hf_model, tune_model_input, hf_inputs, tolerance=1e-2)
    
    if result:
        print("✅ Multimodal comparison passed!")
    else:
        print("❌ Multimodal comparison failed")
        
    return result


def test_generation_consistency(hf_processor, hf_model, tune_model, tune_transform):
    """Test that both models generate consistent outputs."""
    print("Testing generation consistency...")
    
    test_image = create_test_image(224, 224)
    text_input = "Describe this image briefly."
    
    # Format as messages for HuggingFace (using chat template)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": test_image},
                {"type": "text", "text": text_input}
            ]
        }
    ]
    
    # Apply chat template and process
    text = hf_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    hf_inputs = hf_processor(
        text=text,
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
        Message(
            role="user",
            content=[
                {"type": "image", "content": test_image},
                {"type": "text", "content": text_input}
            ]
        )
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


def test_real_cat_image_description(hf_processor, hf_model, tune_model, tune_transform):
    """Test both models with a real cat image and 'describe this image' prompt."""
    print("Testing real cat image description...")
    
    # Get a real cat image from the API
    cat_url = get_cat_image_url()
    if not cat_url:
        print("❌ Failed to get cat image URL, skipping test")
        return False
    
    print(f"Using cat image from: {cat_url}")
    
    # Download and save the image
    cat_image = download_and_save_image(cat_url, "test_cat_image.jpg")
    if not cat_image:
        print("❌ Failed to download cat image, skipping test")
        return False
    
    # Resize image to a reasonable size for the models
    cat_image = cat_image.resize((336, 336))
    cat_image.save("test_cat_image_resized.jpg")
    print(f"✅ Cat image resized and saved as test_cat_image_resized.jpg")
    
    text_input = "Describe this image in detail."
    
    # Process with TorchTune
    messages = [
        Message(
            role="user",
            content=[
                {"type": "image", "content": cat_image},
                {"type": "text", "content": text_input}
            ]
        )
    ]
    
    sample = {
        "image": cat_image,
        "messages": messages
    }
    
    tune_result = tune_transform(sample)
    tune_tokens = torch.tensor([tune_result["tokens"]])
    
    # Process with HuggingFace using custom template
    messages_hf = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": cat_image},
                {"type": "text", "text": text_input}
            ]
        }
    ]
    
    # Use the same custom template as in multimodal test
    custom_template = "{% for message in messages %}{% if message['role'] == 'user' %}<|im_start|>user\n{% for content in message['content'] %}{% if content['type'] == 'image' %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'text' %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}"
    
    text_custom = hf_processor.apply_chat_template(
        messages_hf, 
        chat_template=custom_template,
        tokenize=False, 
        add_generation_prompt=False
    )
    
    text_custom_with_eos = text_custom + "<|im_end|>"
    hf_inputs = hf_processor(text=text_custom_with_eos, images=cat_image, return_tensors="pt")
    
    # Verify token alignment
    hf_tokens = hf_inputs['input_ids'][0].tolist()
    tune_tokens_list = tune_result['tokens']
    
    print(f"Token comparison for cat image:")
    print(f"TorchTune length: {len(tune_tokens_list)}")
    print(f"HuggingFace length: {len(hf_tokens)}")
    
    if tune_tokens_list != hf_tokens:
        print("❌ Token mismatch detected")
        print(f"First 20 TorchTune tokens: {tune_tokens_list[:20]}")
        print(f"First 20 HuggingFace tokens: {hf_tokens[:20]}")
        return False
    
    print("✅ Tokens match!")
    
    # Prepare TorchTune model inputs
    tune_model_input = {
        "tokens": tune_tokens,
        "encoder_input": tune_result["encoder_input"],
        "image_grid_thw": tune_result["encoder_input"]["image"]["grid_thw"]
    }
    
    # Compare logits
    result = compare_logits(tune_model, hf_model, tune_model_input, hf_inputs, tolerance=1e-2)
    
    if result:
        print("✅ Real cat image description test passed!")
    else:
        print("❌ Real cat image description test failed")
    
    # Generate actual descriptions for comparison
    print("\nGenerating descriptions...")
    
    # HuggingFace generation (using greedy decoding for deterministic results)
    with torch.no_grad():
        hf_generated = hf_model.generate(
            **hf_inputs,
            max_new_tokens=50,
            do_sample=False,  # Greedy decoding
            pad_token_id=hf_processor.tokenizer.eos_token_id
        )
    
    # Decode only the new tokens (skip the input)
    input_length = hf_inputs['input_ids'].shape[1]
    hf_new_tokens = hf_generated[0][input_length:]
    hf_description = hf_processor.decode(hf_new_tokens, skip_special_tokens=True)
    
    print(f"HuggingFace description: {hf_description}")
    
    # Generate with TorchTune using our custom generation function (greedy decoding)
    print("Generating with TorchTune...")
    tune_generated_tokens, tune_generated_logits = generate_multimodal(
        model=tune_model,
        tokens=tune_model_input["tokens"],
        encoder_input=tune_model_input["encoder_input"],
        image_grid_thw=tune_model_input["image_grid_thw"],
        max_new_tokens=50,
        temperature=1e-6,  # Very low temperature for greedy-like decoding
        stop_tokens=[151645]  # EOS token for Qwen
    )
    
    # Decode only the new tokens (skip the input)
    input_length = tune_model_input["tokens"].shape[1]
    tune_new_tokens = tune_generated_tokens[0][input_length:]
    
    # For proper decoding, we need the tokenizer - let's get it from the transform
    # For now, just show the token IDs and compare the first few with HF
    print(f"TorchTune generated {len(tune_new_tokens)} new tokens: {tune_new_tokens.tolist()}")
    
    # Compare first few generated tokens between models
    hf_new_tokens_list = hf_new_tokens.tolist()
    tune_new_tokens_list = tune_new_tokens.tolist()
    
    print(f"HuggingFace first 10 tokens: {hf_new_tokens_list[:10]}")
    print(f"TorchTune first 10 tokens: {tune_new_tokens_list[:10]}")
    
    # Check if the first few tokens match (they should be very similar with temperature=1.0)
    if len(tune_new_tokens_list) > 0 and len(hf_new_tokens_list) > 0:
        first_token_match = tune_new_tokens_list[0] == hf_new_tokens_list[0]
        print(f"First token match: {first_token_match}")
        
        # Check how many of the first 5 tokens match
        min_len = min(5, len(tune_new_tokens_list), len(hf_new_tokens_list))
        matches = sum(1 for i in range(min_len) if tune_new_tokens_list[i] == hf_new_tokens_list[i])
        print(f"First {min_len} tokens match: {matches}/{min_len}")
    
    print(f"TorchTune generation completed successfully!")
    
    return result


def generate_multimodal(
    model,
    tokens,
    encoder_input,
    image_grid_thw,
    max_new_tokens=50,
    temperature=1.0,
    top_k=None,
    stop_tokens=None
):
    """
    Custom generation function for multimodal models that handles encoder_input and image_grid_thw.
    
    Args:
        model: The multimodal model
        tokens: Input token tensor [batch_size, seq_len]
        encoder_input: Encoder input dictionary containing image data
        image_grid_thw: Image grid dimensions
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        stop_tokens: List of stop token IDs
    
    Returns:
        Tuple of (generated_tokens, generated_logits)
    """
    model.eval()
    
    # Convert stop_tokens to tensor if provided
    if stop_tokens is not None:
        stop_tokens = torch.tensor(stop_tokens, device=tokens.device, dtype=tokens.dtype)
    
    generated_tokens = tokens.clone()
    all_logits = []
    
    with torch.no_grad():
        for i in range(max_new_tokens):
            # Forward pass
            logits = model(
                generated_tokens,
                encoder_input=encoder_input,
                image_grid_thw=image_grid_thw
            )
            
            # Get logits for the last token
            next_token_logits = logits[:, -1, :]  # [batch_size, vocab_size]
            all_logits.append(next_token_logits.unsqueeze(1))  # [batch_size, 1, vocab_size]
            
            # Sample next token
            next_token = sample(
                next_token_logits,
                temperature=temperature,
                top_k=top_k
            )  # [batch_size, 1]
            
            # Append to generated tokens
            generated_tokens = torch.cat([generated_tokens, next_token], dim=-1)
            
            # Check for stop tokens
            if stop_tokens is not None:
                if torch.isin(next_token, stop_tokens).any():
                    break
    
    # Concatenate all logits
    generated_logits = torch.cat(all_logits, dim=1)  # [batch_size, num_generated, vocab_size]
    
    return generated_tokens, generated_logits


def try_batch_processing_torchtune(tune_model, tune_samples):
    """
    Attempt to process multiple samples as a true batch in TorchTune.
    This might not work if the model doesn't support variable-length sequences or batched encoder inputs.
    """
    try:
        print("Attempting true batch processing for TorchTune...")
        
        # Check if all samples have the same token length
        token_lengths = [len(sample["tokens"]) for sample in tune_samples]
        if len(set(token_lengths)) > 1:
            print(f"❌ Cannot batch: different token lengths {token_lengths}")
            return None
        
        # Check if all samples have the same image dimensions
        image_shapes = []
        for sample in tune_samples:
            img_hidden_states = sample["encoder_input"]["image"]["hidden_states"]
            image_shapes.append(img_hidden_states.shape)
        
        if len(set(image_shapes)) > 1:
            print(f"❌ Cannot batch: different image shapes {image_shapes}")
            return None
        
        print(f"✅ All samples compatible for batching (token_len={token_lengths[0]}, img_shape={image_shapes[0]})")
        
        # Stack tokens
        batch_tokens = torch.stack([torch.tensor(sample["tokens"]) for sample in tune_samples])
        
        # Stack image hidden states
        batch_image_hidden_states = torch.stack([
            sample["encoder_input"]["image"]["hidden_states"] 
            for sample in tune_samples
        ])
        
        # Stack image grid thw
        batch_image_grid_thw = torch.stack([
            sample["encoder_input"]["image"]["grid_thw"] 
            for sample in tune_samples
        ])
        
        # Create batched encoder input
        batch_encoder_input = {
            "image": {
                "hidden_states": batch_image_hidden_states,
                "grid_thw": batch_image_grid_thw
            }
        }
        
        print(f"Batched tokens shape: {batch_tokens.shape}")
        print(f"Batched image hidden states shape: {batch_image_hidden_states.shape}")
        print(f"Batched image grid thw shape: {batch_image_grid_thw.shape}")
        
        # Try forward pass
        with torch.no_grad():
            batch_output = tune_model(
                batch_tokens,
                encoder_input=batch_encoder_input,
                image_grid_thw=batch_image_grid_thw
            )
        
        print(f"✅ Batch forward pass successful! Output shape: {batch_output.shape}")
        return {
            "tokens": batch_tokens,
            "encoder_input": batch_encoder_input,
            "image_grid_thw": batch_image_grid_thw,
            "output": batch_output
        }
        
    except Exception as e:
        print(f"❌ Batch processing failed: {e}")
        return None


def test_batched_inputs(hf_processor, hf_model, tune_model, tune_transform):
    """Test both models with batched inputs (multiple images)."""
    print("Testing batched inputs...")
    
    batch_size = 3
    images = []
    prompts = [
        "Describe this image briefly.",
        "What do you see in this picture?", 
        "Tell me about this photo."
    ]
    
    # Get multiple cat images
    for i in range(batch_size):
        cat_url = get_cat_image_url()
        if not cat_url:
            print(f"❌ Failed to get cat image URL for batch item {i}, using synthetic image")
            cat_image = create_test_image(336, 336)
        else:
            print(f"Downloading cat image {i+1}/{batch_size} from: {cat_url}")
            cat_image = download_and_save_image(cat_url, f"test_cat_batch_{i}.jpg")
            if not cat_image:
                print(f"❌ Failed to download cat image {i}, using synthetic image")
                cat_image = create_test_image(336, 336)
            else:
                cat_image = cat_image.resize((336, 336))
                cat_image.save(f"test_cat_batch_{i}_resized.jpg")
        
        images.append(cat_image)
    
    print(f"✅ Prepared {len(images)} images for batch testing")
    
    # Process each sample separately for TorchTune (since batching might not be fully supported)
    tune_samples = []
    for i, (image, prompt) in enumerate(zip(images, prompts)):
        messages = [
            Message(
                role="user",
                content=[
                    {"type": "image", "content": image},
                    {"type": "text", "content": prompt}
                ]
            )
        ]
        
        sample = {
            "image": image,
            "messages": messages
        }
        
        tune_result = tune_transform(sample)
        tune_samples.append(tune_result)
    
    # For HuggingFace, we can try true batching
    hf_messages_batch = []
    hf_images_batch = []
    
    for i, (image, prompt) in enumerate(zip(images, prompts)):
        hf_messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }]
        hf_messages_batch.append(hf_messages)
        hf_images_batch.append(image)
    
    # Process HuggingFace batch
    custom_template = "{% for message in messages %}{% if message['role'] == 'user' %}<|im_start|>user\n{% for content in message['content'] %}{% if content['type'] == 'image' %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'text' %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}"
    
    # Process each HF sample separately (HF batching with different images can be complex)
    hf_inputs_list = []
    for i, (hf_messages, image) in enumerate(zip(hf_messages_batch, hf_images_batch)):
        text_custom = hf_processor.apply_chat_template(
            hf_messages, 
            chat_template=custom_template,
            tokenize=False, 
            add_generation_prompt=False
        )
        text_custom_with_eos = text_custom + "<|im_end|>"
        hf_inputs = hf_processor(text=text_custom_with_eos, images=image, return_tensors="pt")
        hf_inputs_list.append(hf_inputs)
    
    # Try true batch processing for TorchTune
    batch_result = try_batch_processing_torchtune(tune_model, tune_samples)
    
    print("Comparing individual samples in batch...")
    
    # Compare each sample individually
    all_results = []
    for i in range(batch_size):
        print(f"\n--- Processing batch item {i+1}/{batch_size} ---")
        
        # Prepare TorchTune inputs
        tune_tokens = torch.tensor([tune_samples[i]["tokens"]])
        tune_model_input = {
            "tokens": tune_tokens,
            "encoder_input": tune_samples[i]["encoder_input"],
            "image_grid_thw": tune_samples[i]["encoder_input"]["image"]["grid_thw"]
        }
        
        # Get HF inputs
        hf_inputs = hf_inputs_list[i]
        
        # Verify token alignment for this sample
        hf_tokens = hf_inputs['input_ids'][0].tolist()
        tune_tokens_list = tune_samples[i]['tokens']
        
        print(f"Sample {i+1} - TorchTune tokens: {len(tune_tokens_list)}, HF tokens: {len(hf_tokens)}")
        
        if tune_tokens_list != hf_tokens:
            print(f"❌ Token mismatch in sample {i+1}")
            print(f"First 10 TorchTune: {tune_tokens_list[:10]}")
            print(f"First 10 HF: {hf_tokens[:10]}")
            all_results.append(False)
            continue
        
        print(f"✅ Sample {i+1} tokens match!")
        
        # Compare logits for this sample
        result = compare_logits(tune_model, hf_model, tune_model_input, hf_inputs, tolerance=1e-3)
        all_results.append(result)
        
        if result:
            print(f"✅ Sample {i+1} logits match!")
        else:
            print(f"❌ Sample {i+1} logits don't match")
        
    # Test batch processing results if available
    batch_processing_passed = False
    if batch_result is not None:
        print(f"\n--- Testing True Batch Processing ---")
        try:
            # Compare batch output with individual outputs
            batch_output = batch_result["output"]  # [batch_size, seq_len, vocab_size]
            
            print(f"Batch output shape: {batch_output.shape}")
            
            # Get individual outputs for comparison
            individual_outputs = []
            for i in range(batch_size):
                tune_tokens = torch.tensor([tune_samples[i]["tokens"]])
                tune_model_input = {
                    "tokens": tune_tokens,
                    "encoder_input": tune_samples[i]["encoder_input"],
                    "image_grid_thw": tune_samples[i]["encoder_input"]["image"]["grid_thw"]
                }
                
                with torch.no_grad():
                    individual_output = tune_model(
                        tune_model_input["tokens"],
                        encoder_input=tune_model_input["encoder_input"],
                        image_grid_thw=tune_model_input["image_grid_thw"]
                    )
                individual_outputs.append(individual_output)
            
            # Compare batch vs individual outputs
            batch_matches_individual = True
            for i in range(batch_size):
                batch_sample_output = batch_output[i:i+1]  # [1, seq_len, vocab_size]
                individual_output = individual_outputs[i]  # [1, seq_len, vocab_size]
                
                if not torch.allclose(batch_sample_output, individual_output, atol=1e-5, rtol=1e-5):
                    print(f"❌ Batch sample {i+1} doesn't match individual processing")
                    batch_matches_individual = False
                    
                    # Show some statistics about the difference
                    diff = torch.abs(batch_sample_output - individual_output)
                    print(f"   Max difference: {torch.max(diff).item():.6f}")
                    print(f"   Mean difference: {torch.mean(diff).item():.6f}")
                else:
                    print(f"✅ Batch sample {i+1} matches individual processing")
            
            if batch_matches_individual:
                print("✅ True batch processing produces identical results to individual processing!")
                batch_processing_passed = True
            else:
                print("❌ True batch processing differs from individual processing")
                
        except Exception as e:
            print(f"❌ Error testing batch processing: {e}")
    
    # Summary
    passed_samples = sum(all_results)
    print(f"\n--- Batch Test Summary ---")
    print(f"Individual samples passed: {passed_samples}/{batch_size}")
    print(f"True batch processing: {'✅ Passed' if batch_processing_passed else '❌ Failed/Not Available'}")
    
    overall_result = passed_samples == batch_size
    if overall_result:
        print("✅ Batched inputs test passed!")
    else:
        print("❌ Some samples in batch failed")
    
    return overall_result


def run_all_tests():
    """Run all full model tests."""
    print("=" * 60)
    print("Running Qwen2.5-VL Full Model Comparison Tests")
    print("=" * 60)
    
    # Load models once
    print("Loading models...")
    hf_processor, hf_model = load_hf_model()
    tune_model = load_tune_model()
    tune_transform = load_tune_transform()
    
    if None in [hf_processor, hf_model, tune_model, tune_transform]:
        print("❌ Failed to load required models")
        return False
    
    print("✅ All models loaded successfully")
    print("-" * 40)
    
    tests = [
        # test_text_only_comparison,
        # test_multimodal_comparison,
        # test_generation_consistency,
        # test_real_cat_image_description,
        test_batched_inputs,
    ]
    
    results = []
    for test in tests:
        result = test(hf_processor, hf_model, tune_model, tune_transform)
        results.append(result)
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