"""Test file for Qwen2.5-VL Vision Encoder component."""

import os
import torch
from torch import nn
import numpy as np
from PIL import Image
from torchtune.models.qwen2_5_vision import qwen2_5_vision_encoder
from torchtune.models.qwen2_5_vision._transform import Qwen2_5_VLTransform
from transformers import AutoProcessor, AutoModelForImageTextToText
from torchtune.data import Message
import safetensors
from torchtune.models.qwen2_5_vision import qwen2_5_vl_hf_to_tune, qwen2_5_vl_7b
import matplotlib.pyplot as plt

# ADD HF_MODEL_PATH to env
model_path = os.environ.get("HF_MODEL_PATH")
PATH = f"{model_path}/vocab.json"
MERGES_FILE = f"{model_path}/merges.txt"
HF_MODEL_PATH = model_path

def create_test_image(width: int = 224, height: int = 224) -> Image.Image:
    """Create a simple test image."""
    # Create a random RGB image
    image_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(image_array)

def load_tune_model():
    """Load TorchTune model with converted weights."""
    print("Loading TorchTune model...")
    tune_model_path = model_path
    
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

def load_models():
    """Load both HuggingFace and custom vision models."""
    
    # Load HF model
    hf_processor = AutoProcessor.from_pretrained(HF_MODEL_PATH)
    hf_model = AutoModelForImageTextToText.from_pretrained(HF_MODEL_PATH)
    hf_vision_encoder = hf_model.visual
    
    # Load custom model
    tune_qwen = load_tune_model()
    tune_vision_encoder = tune_qwen.encoders["image"]
    
    # Set both to eval mode
    hf_vision_encoder.eval()
    tune_vision_encoder.eval()
    
    return hf_processor, hf_vision_encoder, tune_vision_encoder


def test_vision_encoder_comparison():
    """Compare hidden states between HF and custom vision encoders."""
    print("Comparing HF vs Custom Vision Encoder hidden states...")
    
    try:
        # Load models
        hf_processor, hf_vision_encoder, tune_vision_encoder = load_models()
        
        # Create test image
        test_image = create_test_image(448, 448)
        
        # Process with HF processor
        hf_inputs = hf_processor(images=test_image, text="", return_tensors="pt")
        pixel_values = hf_inputs["pixel_values"]
        image_grid_thw = hf_inputs.get("image_grid_thw", torch.tensor([[1, 32, 32]]))  # Default grid
        
        print(f"HUGGINGFACE: Input shapes - Pixel values: {pixel_values.shape}, Grid THW: {image_grid_thw.shape}")
        print(f"HUGGINGFACE: Pixel values dtype: {pixel_values.dtype}")

        message = Message(
            role="user",
            content=[
                {"type": "image", "content": test_image}
            ]
        )
        sample = {"messages": [message]}
        tune_inputs = Qwen2_5_VLTransform(path=PATH, merges_file=MERGES_FILE)(sample)
        # pixel_values_tune is about the same as pixel_values; same shape; float32 vs bfloat16
        pixel_values_tune = tune_inputs["encoder_input"]["image"]["hidden_states"][0]
        image_grid_thw_tune = tune_inputs["encoder_input"]["image"]["grid_thw"]

        print(f"TORCHTUNE: Input shapes - Pixel values: {pixel_values_tune.shape}, Grid THW: {image_grid_thw_tune.shape}")
        print(f"TORCHTUNE: Pixel values dtype: {pixel_values_tune.dtype}")  # Should be bfloat16

        print(f"PIXEL VALUE DIFF: {torch.abs(pixel_values - pixel_values_tune).max()}")
        
        # Forward pass through both encoders
        with torch.no_grad():
            # HF encoder
            hf_hidden_states = hf_vision_encoder(pixel_values, grid_thw=image_grid_thw)
            custom_output = tune_vision_encoder(pixel_values_tune, image_grid_thw_tune)
        
        # Compare outputs
        hf_hidden_states = hf_hidden_states.squeeze(0)  # Remove batch dim
        custom_output = custom_output.squeeze(0)  # Remove batch dim
        
        print(f"HF output shape: {hf_hidden_states.shape}")
        print(f"Custom output shape: {custom_output.shape}")
        
        # Ensure same sequence length for comparison
        min_seq_len = min(hf_hidden_states.shape[0], custom_output.shape[0])
        print(f"sequences length are {hf_hidden_states.shape[0] == custom_output.shape[0]} the same")
        hf_truncated = hf_hidden_states[:min_seq_len]
        custom_truncated = custom_output[:min_seq_len]
        
        # Compare hidden states
        diff = torch.abs(hf_truncated - custom_truncated)
        max_diff = torch.max(diff)
        mean_diff = torch.mean(diff)
        
        print(f"Max absolute difference: {max_diff:.6f}")
        print(f"Mean absolute difference: {mean_diff:.6f}")
        
        # Check if differences are within reasonable tolerance
        tolerance = 1e-3  # Adjust based on expected precision
        close_match = max_diff < tolerance
        
        if close_match:
            print("✅ Hidden states match within tolerance!")
        else:
            print(f"⚠️  Hidden states differ beyond tolerance ({tolerance})")
            
        return close_match
        
    except Exception as e:
        print(f"❌ Vision encoder comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vision_encoder_consistency():
    """Test that the custom encoder produces consistent outputs."""
    print("Testing custom vision encoder consistency...")
    
    tune_vision_encoder = qwen2_5_vision_encoder(
        embed_dim=1280,
        num_layers=32,
        activation=nn.SiLU(),
        intermediate_size=3420,
        num_heads=16,
        in_channels=3,
        out_hidden_size=3584,
        patch_size=14,
        spatial_merge_size=2,
        # spatial_patch_size=14,
        window_size=112,
        full_att_block_indexes=[7, 15, 23, 31],
        temporal_patch_size=2,
        # tokens_per_second=2 # NOTE: needed for get_rope_index
    )
    tune_vision_encoder.eval()
    
    # Create test input
    seq_len = 256
    hidden_states = torch.randn(seq_len, 1176)
    grid_thw = torch.tensor([[1, 16, 16]])
    
    # Run multiple times and check consistency
    outputs = []
    with torch.no_grad():
        for _ in range(3):
            output = tune_vision_encoder(hidden_states, grid_thw)
            outputs.append(output)
    
    # Check all outputs are identical (deterministic)
    for i in range(1, len(outputs)):
        diff = torch.abs(outputs[0] - outputs[i])
        max_diff = torch.max(diff)
        if max_diff > 1e-6:
            print(f"⚠️  Outputs not consistent across runs (max diff: {max_diff})")
            return False
    
    print("✅ Custom encoder produces consistent outputs!")
    return True
        


def run_all_tests():
    """Run all vision encoder tests."""
    print("=" * 60)
    print("Qwen2.5-VL Vision Encoder Implementation Comparison Tests")
    print("=" * 60)
    
    tests = [
        # test_vision_encoder_consistency,
        test_vision_encoder_comparison,
    ]
    
    results = []
    for test in tests:
        print(f"\n{test.__name__.replace('_', ' ').title()}:")
        print("-" * 40)
        result = test()
        results.append(result)
    
    # Summary
    passed = sum(results)
    total = len(results)
    print(f"\n{'='*60}")
    print(f"Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed - check implementation differences")
        
    return passed == total

def compared_saved_tensors():
    print(f"{"="*60}")
    print("COMPARING SAVED TENSORS")
    print(f"{"-"*40}")

    # compare forward input
    tune_hidden_states = torch.load(os.path.join(os.environ["ENCODER_TEST_PATH"], "tune_hidden_states.pt"))
    hf_hidden_states = torch.load(os.path.join(os.environ["ENCODER_TEST_PATH"], "hf_hidden_states.pt"))
    print(f"TUNE hidden states and HF hidden states diff: {torch.abs(hf_hidden_states - tune_hidden_states).max()}")

    # compare hidden states after reshape (2)
    tune_hidden_states_2 = torch.load(os.path.join(os.environ["ENCODER_TEST_PATH"], "tune_hidden_states_2.pt"))
    hf_hidden_states_2 = torch.load(os.path.join(os.environ["ENCODER_TEST_PATH"], "hf_hidden_states_2.pt"))
    print(f"TUNE hidden states 2 and HF hidden states 2 diff: {torch.abs(hf_hidden_states_2 - tune_hidden_states_2).max()}")

    # compare hidden states after attention
    tune_hidden_states_3 = torch.load(os.path.join(os.environ["ENCODER_TEST_PATH"], "tune_hidden_states_3.pt"))
    hf_hidden_states_3 = torch.load(os.path.join(os.environ["ENCODER_TEST_PATH"], "hf_hidden_states_3.pt"))
    print(f"TUNE hidden states 3 and HF hidden states 3 diff: {torch.abs(hf_hidden_states_3 - tune_hidden_states_3).max()}")

    print(f"{"-"*40}")
    print("DIVING INTO ATTENTION MODULE")

    # compare inputs to attention
    print("COMPARING INPUTS TO ATTENTION")
    tune_input_to_attn = torch.load(os.path.join(os.environ["ENCODER_TEST_PATH"], "tune_input_to_attn.pt"))
    hf_input_to_attn = torch.load(os.path.join(os.environ["ENCODER_TEST_PATH"], "hf_input_to_attn.pt"))
    print(f"TUNE input to attn and HF input to attn diff: {torch.abs(hf_input_to_attn - tune_input_to_attn).max()}")
    print(f"{"-"*40}")

    # compare query vectors (dim 80)
    tune_q = torch.load(os.path.join(os.environ["ENCODER_TEST_PATH"], "tune_q_before_pos_embed.pt"))
    tune_first_query_vector = tune_q[0, :, 0, :]

    hf_q = torch.load(os.path.join(os.environ["ENCODER_TEST_PATH"], "hf_q_before_pos_embed.pt"))
    hf_first_query_vector = hf_q[:, 0, :] # [1024, 16, 80]
    
    print(f"TUNE Q and HF Q diff BEFORE RoPE: {torch.abs(hf_first_query_vector - tune_first_query_vector).max()}")

    diff = torch.abs(hf_first_query_vector - tune_first_query_vector)
    diff = diff.cpu().numpy()
    plt.figure(figsize=(12, 6), dpi=300)
    plt.imshow(diff, cmap="viridis")
    plt.colorbar()
    plt.savefig(os.path.join(os.environ["ENCODER_TEST_PATH"], "q_diff_before_rope.png"))
    plt.close()
    print(f"{"-"*40}")

    # compare query vectors after RoPE
    tune_q_after_rope = torch.load(os.path.join(os.environ["ENCODER_TEST_PATH"], "tune_q_after_pos_embed.pt"))
    hf_q_after_rope = torch.load(os.path.join(os.environ["ENCODER_TEST_PATH"], "hf_q_after_pos_embed.pt"))
    tune_first_query_vector_after_rope = tune_q_after_rope[0, :, 0, :]
    hf_first_query_vector_after_rope = hf_q_after_rope[:, 0, :]
    print(f"TUNE Q after RoPE shape: {tune_q_after_rope.shape}")
    print(f"HF Q after RoPE shape: {hf_q_after_rope.shape}")
    print(f"TUNE Q after RoPE and HF Q AFTER RoPE diff: {torch.abs(hf_first_query_vector_after_rope - tune_first_query_vector_after_rope).max()}")

    diff = torch.abs(hf_first_query_vector_after_rope - tune_first_query_vector_after_rope)
    diff = diff.cpu().numpy()
    plt.figure(figsize=(12, 6), dpi=300)
    plt.imshow(diff, cmap="viridis")
    plt.colorbar()
    plt.savefig(os.path.join(os.environ["ENCODER_TEST_PATH"], "q_diff_after_rope.png"))
    plt.close()

    # compare query projection matrices
    tune_q_proj_weight = torch.load(os.path.join(os.environ["ENCODER_TEST_PATH"], "tune_q_proj_weight.pt"))
    hf_qkv_weight = torch.load(os.path.join(os.environ["ENCODER_TEST_PATH"], "hf_qkv_weight.pt"))
    hf_qkv_weight = hf_qkv_weight.reshape(-1, 1280, 1280)
    hf_q_proj_weight = hf_qkv_weight[0, :, :]
    print(f"TUNE Q proj weight and HF QKV weight diff: {torch.abs(tune_q_proj_weight - hf_q_proj_weight).max()}")

    # generate heatmap
    diff = torch.abs(hf_q_proj_weight - tune_q_proj_weight)
    diff = diff.detach().cpu().numpy()
    plt.imshow(diff, cmap="viridis")
    plt.colorbar()
    plt.savefig(os.path.join(os.environ["ENCODER_TEST_PATH"], "tune_q_proj_weight_hf_q_proj_weight_diff.png"))
    plt.close()
    print(f"{"-"*40}")

    # compare attention mask
    tune_attention_mask = torch.load(os.path.join(os.environ["ENCODER_TEST_PATH"], "tune_attention_mask.pt"))
    hf_attention_mask = torch.load(os.path.join(os.environ["ENCODER_TEST_PATH"], "hf_attention_mask.pt"))
    print(f"TUNE attention mask and HF attention mask num different: {torch.logical_xor(tune_attention_mask, hf_attention_mask).sum()}")

    print(f"{"-"*40}")
    print("COMPARING WINDOW INDEX")
    tune_window_index = torch.load(os.path.join(os.environ["ENCODER_TEST_PATH"], "tune_window_index.pt"))
    hf_window_index = torch.load(os.path.join(os.environ["ENCODER_TEST_PATH"], "hf_window_index.pt"))
    print(f"TUNE window index and HF window index diff: {torch.abs(hf_window_index - tune_window_index).max()}")
    print(f"{"-"*40}")

    print("ROPE CACHE vs HF ROPE")
    tune_rope_cache_cos = torch.load(os.path.join(os.environ["ENCODER_TEST_PATH"], "tune_rope_cache_cos.pt"))
    tune_rope_cache_sin = torch.load(os.path.join(os.environ["ENCODER_TEST_PATH"], "tune_rope_cache_sin.pt"))
    tune_rope_cache_cos = tune_rope_cache_cos.squeeze(0).squeeze(1)
    tune_rope_cache_sin = tune_rope_cache_sin.squeeze(0).squeeze(1)
    hf_cos = torch.load(os.path.join(os.environ["ENCODER_TEST_PATH"], "hf_position_embeddings_cos.pt"))
    hf_sin = torch.load(os.path.join(os.environ["ENCODER_TEST_PATH"], "hf_position_embeddings_sin.pt"))
    print(f"HF cos shape: {hf_cos.shape}")
    print(f"HF sin shape: {hf_sin.shape}")
    print(f"HF cos: {hf_cos}")
    print(f"HF sin: {hf_sin}")
    print(f"TUNE rope cache cos shape: {tune_rope_cache_cos.shape}")
    print(f"TUNE rope cache sin shape: {tune_rope_cache_sin.shape}")
    print(f"TUNE rope cache cos: {tune_rope_cache_cos}")
    print(f"TUNE rope cache sin: {tune_rope_cache_sin}")
    hf_cos_half = hf_cos[:, :40]
    print(f"cos equivalence {torch.allclose(hf_cos_half, tune_rope_cache_cos)}")

    print(f"{"-"*40}")
    print(f"COMPARE VECTORS IN ROTATION OPERATION")
    tune_xshaped_0 = torch.load(os.path.join(os.environ["ENCODER_TEST_PATH"], "tune_xshaped_0.pt")) # [1, 1024, 16, 40]
    tune_xshaped_1 = torch.load(os.path.join(os.environ["ENCODER_TEST_PATH"], "tune_xshaped_1.pt"))
    hf_xshaped_0 = hf_q[:, :, :40]
    hf_xshaped_1 = hf_q[:, :, 40:]
    print(f"TUNE xshaped 0 shape: {tune_xshaped_0.shape}")
    print(f"TUNE xshaped 1 shape: {tune_xshaped_1.shape}")
    print(f"HF xshaped 0 shape: {hf_xshaped_0.shape}")
    print(f"HF xshaped 1 shape: {hf_xshaped_1.shape}")
    print(f"TUNE xshaped 0 and HF xshaped 0 diff: {torch.abs(hf_xshaped_0 - tune_xshaped_0).max()}")
    print(f"TUNE xshaped 1 and HF xshaped 1 diff: {torch.abs(hf_xshaped_1 - tune_xshaped_1).max()}")
    print(f"{"-"*40}")

    breakpoint()

if __name__ == "__main__":
    run_all_tests() 
    compared_saved_tensors()
    # debug_dimensional_differences()