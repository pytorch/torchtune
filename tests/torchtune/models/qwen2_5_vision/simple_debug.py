import torch
import os
from pathlib import Path

from torchtune.models.qwen2_5_vision._convert_weights import qwen2_5_vl_hf_to_tune
from torchtune.models.qwen2_5_vision._model_builders import qwen2_5_vl_7b
import safetensors.torch
from transformers import AutoProcessor, AutoModelForImageTextToText


def save_tensor(tensor, name, debug_dir="/mnt/vast/home/lawrence/debug_tensors"):
    """Save a tensor with a descriptive name."""
    debug_dir = Path(debug_dir)
    debug_dir.mkdir(exist_ok=True)
    
    if tensor is None:
        return
    
    filepath = debug_dir / f"{name}.pt"
    torch.save(tensor.detach().cpu(), filepath)
    print(f"Saved {name}: {tensor.shape}")


def debug_hf_model():
    """Debug HuggingFace model step by step."""
    print("=== Debugging HuggingFace Model ===")
    
    # Load model
    hf_model_path = "/mnt/vast/share/inf2-training/models/open_source/Qwen2.5-VL-7B-Instruct"
    hf_model = AutoModelForImageTextToText.from_pretrained(hf_model_path)
    hf_model.eval().to("cuda")
    
    # Input
    input_ids = torch.tensor([[1, 2, 3, 4, 5]]).to("cuda")
    print(f"Input: {input_ids}")
    
    # Explore model structure more thoroughly
    print("\nHF Model structure:")
    print(f"Type: {type(hf_model)}")
    print(f"Has model attr: {hasattr(hf_model, 'model')}")
    
    # Look for embeddings in different places
    embedding_layer = None
    if hasattr(hf_model, 'model') and hasattr(hf_model.model, 'embed_tokens'):
        embedding_layer = hf_model.model.embed_tokens
        print("✓ Found embed_tokens in model.embed_tokens")
    elif hasattr(hf_model, 'model') and hasattr(hf_model.model, 'language_model') and hasattr(hf_model.model.language_model, 'embed_tokens'):
        embedding_layer = hf_model.model.language_model.embed_tokens
        print("✓ Found embed_tokens in model.language_model.embed_tokens")
    elif hasattr(hf_model, 'transformer') and hasattr(hf_model.transformer, 'wte'):
        embedding_layer = hf_model.transformer.wte
        print("✓ Found embeddings in transformer.wte")
    else:
        # Try to find token embedding layer specifically (not visual embeddings)
        for name, module in hf_model.named_modules():
            if ('embed_tokens' in name or 'token_embed' in name) and hasattr(module, 'weight'):
                print(f"Found token embedding: {name} -> {type(module)}")
                embedding_layer = module
                break
        
        if embedding_layer is None:
            # Last resort - find any embedding that's not a conv layer
            for name, module in hf_model.named_modules():
                if 'embed' in name.lower() and hasattr(module, 'weight') and not isinstance(module, torch.nn.Conv3d):
                    print(f"Found potential embedding: {name} -> {type(module)}")
                    embedding_layer = module
                    break
    
    with torch.no_grad():
        # Step 1: Token embeddings
        if embedding_layer is not None:
            embeddings = embedding_layer(input_ids)
            save_tensor(embeddings, "hf_embed_tokens")
            print(f"Embeddings shape: {embeddings.shape}")
        else:
            print("⚠ Could not find embedding layer")
        
        # Step 2: Run full model
        output = hf_model(input_ids)
        save_tensor(output.logits, "hf_final_logits")
        print(f"Final logits shape: {output.logits.shape}")
        
        return output.logits


def debug_torchtune_model():
    """Debug TorchTune model step by step."""
    print("\n=== Debugging TorchTune Model ===")
    
    # Load model
    hf_model_path = "/mnt/vast/share/inf2-training/models/open_source/Qwen2.5-VL-7B-Instruct"
    tune_qwen = qwen2_5_vl_7b()
    
    state_dict = {}
    files = [f"{hf_model_path}/model-0000{i}-of-00005.safetensors" for i in range(1, 6)]
    for file in files:
        load_files_dict = safetensors.torch.load_file(file)
        state_dict.update(load_files_dict)

    converted = qwen2_5_vl_hf_to_tune(state_dict)
    tune_qwen.load_state_dict(converted)
    tune_qwen.eval().to("cuda")
    
    # Input
    input_ids = torch.tensor([[1, 2, 3, 4, 5]]).to("cuda")
    
    with torch.no_grad():
        # Step 1: Token embeddings
        if hasattr(tune_qwen.decoder, 'tok_embeddings'):
            embeddings = tune_qwen.decoder.tok_embeddings(input_ids)
            save_tensor(embeddings, "tt_embed_tokens")
            print(f"Embeddings shape: {embeddings.shape}")
        
        # Step 2: Run full model
        output = tune_qwen(input_ids)
        save_tensor(output, "tt_final_logits")
        print(f"Final logits shape: {output.shape}")
        
        return output


def compare_embeddings():
    """Compare token embeddings between models."""
    print("\n=== Comparing Token Embeddings ===")
    
    debug_dir = Path("/mnt/vast/home/lawrence/debug_tensors")
    
    hf_embed_file = debug_dir / "hf_embed_tokens.pt"
    tt_embed_file = debug_dir / "tt_embed_tokens.pt"
    
    if hf_embed_file.exists() and tt_embed_file.exists():
        hf_embed = torch.load(hf_embed_file)
        tt_embed = torch.load(tt_embed_file)
        
        print(f"HF embeddings shape: {hf_embed.shape}")
        print(f"TT embeddings shape: {tt_embed.shape}")
        
        # Handle shape differences (HF might have batch dim)
        if hf_embed.dim() == 3 and tt_embed.dim() == 2:
            hf_embed = hf_embed.squeeze(0)  # Remove batch dim
        elif hf_embed.dim() == 2 and tt_embed.dim() == 3:
            tt_embed = tt_embed.squeeze(0)  # Remove batch dim
            
        if hf_embed.shape == tt_embed.shape:
            max_diff = torch.max(torch.abs(hf_embed - tt_embed)).item()
            mean_diff = torch.mean(torch.abs(hf_embed - tt_embed)).item()
            close = torch.allclose(hf_embed, tt_embed, atol=1e-5, rtol=1e-4)
            
            status = "✅" if close else "❌"
            print(f"{status} Token embeddings: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}, close={close}")
            
            if not close:
                print("❌ Token embeddings already differ! This suggests:")
                print("   1. Different tokenizer/vocabulary")
                print("   2. Different embedding weights")
                print("   3. Weight conversion issues")
                return False
            else:
                print("✅ Token embeddings match - differences must be in transformer layers")
                return True
        else:
            print(f"❌ Shape mismatch: HF{hf_embed.shape} vs TT{tt_embed.shape}")
            return False
    else:
        print("⚠ Missing embedding files")
        return False


def analyze_logit_differences():
    """Analyze where the logit differences occur."""
    print("\n=== Analyzing Logit Differences ===")
    
    debug_dir = Path("/mnt/vast/home/lawrence/debug_tensors")
    
    hf_logits_file = debug_dir / "hf_final_logits.pt"
    tt_logits_file = debug_dir / "tt_final_logits.pt"
    
    if hf_logits_file.exists() and tt_logits_file.exists():
        hf_logits = torch.load(hf_logits_file)
        tt_logits = torch.load(tt_logits_file)
        
        print(f"HF logits shape: {hf_logits.shape}")
        print(f"TT logits shape: {tt_logits.shape}")
        
        if hf_logits.shape == tt_logits.shape:
            diff = torch.abs(hf_logits - tt_logits)
            
            # Overall statistics
            max_diff = torch.max(diff).item()
            mean_diff = torch.mean(diff).item()
            std_diff = torch.std(diff).item()
            
            print(f"Difference statistics:")
            print(f"  Max: {max_diff:.2e}")
            print(f"  Mean: {mean_diff:.2e}")
            print(f"  Std: {std_diff:.2e}")
            
            # Find where max differences occur
            max_indices = torch.unravel_index(torch.argmax(diff), diff.shape)
            print(f"  Max diff location: {max_indices}")
            print(f"  HF value at max: {hf_logits[max_indices].item():.6f}")
            print(f"  TT value at max: {tt_logits[max_indices].item():.6f}")
            
            # Analyze by position and vocabulary
            batch_size, seq_len, vocab_size = hf_logits.shape
            
            print(f"\nDifferences by position:")
            for pos in range(seq_len):
                pos_diff = diff[0, pos, :]
                pos_max = torch.max(pos_diff).item()
                pos_mean = torch.mean(pos_diff).item()
                print(f"  Position {pos}: max={pos_max:.2e}, mean={pos_mean:.2e}")
            
            print(f"\nDifferences by vocabulary range:")
            vocab_ranges = [
                (0, 1000, "0-1K (common)"),
                (1000, 10000, "1K-10K (medium)"),
                (10000, 50000, "10K-50K (rare)"),
                (50000, vocab_size, "50K+ (very rare)")
            ]
            
            for start, end, desc in vocab_ranges:
                range_diff = diff[:, :, start:end]
                if range_diff.numel() > 0:
                    range_max = torch.max(range_diff).item()
                    range_mean = torch.mean(range_diff).item()
                    print(f"  {desc}: max={range_max:.2e}, mean={range_mean:.2e}")
            
            # Check if differences are consistent across positions
            print(f"\nConsistency check:")
            first_pos_logits_hf = hf_logits[0, 0, :]
            first_pos_logits_tt = tt_logits[0, 0, :]
            
            for pos in range(1, min(seq_len, 3)):
                pos_logits_hf = hf_logits[0, pos, :]
                pos_logits_tt = tt_logits[0, pos, :]
                
                # Check if the pattern of differences is similar
                diff_pattern_consistency = torch.corrcoef(torch.stack([
                    first_pos_logits_hf - first_pos_logits_tt,
                    pos_logits_hf - pos_logits_tt
                ]))[0, 1].item()
                
                print(f"  Diff pattern correlation pos0 vs pos{pos}: {diff_pattern_consistency:.4f}")
            
            return max_diff < 1e-4
        else:
            print(f"❌ Shape mismatch: HF{hf_logits.shape} vs TT{tt_logits.shape}")
            return False
    else:
        print("⚠ Missing logits files")
        return False


def compare_final_logits():
    """Compare final logits between models."""
    print("\n=== Comparing Final Logits ===")
    
    debug_dir = Path("/mnt/vast/home/lawrence/debug_tensors")
    
    hf_logits_file = debug_dir / "hf_final_logits.pt"
    tt_logits_file = debug_dir / "tt_final_logits.pt"
    
    if hf_logits_file.exists() and tt_logits_file.exists():
        hf_logits = torch.load(hf_logits_file)
        tt_logits = torch.load(tt_logits_file)
        
        if hf_logits.shape == tt_logits.shape:
            max_diff = torch.max(torch.abs(hf_logits - tt_logits)).item()
            mean_diff = torch.mean(torch.abs(hf_logits - tt_logits)).item()
            close = torch.allclose(hf_logits, tt_logits, atol=1e-4, rtol=1e-4)
            
            status = "✅" if close else "❌"
            print(f"{status} Final logits: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}, close={close}")
            
            # Show some sample values
            print(f"HF logits sample: {hf_logits[0, 0, :5]}")
            print(f"TT logits sample: {tt_logits[0, 0, :5]}")
            
            return close
        else:
            print(f"❌ Shape mismatch: HF{hf_logits.shape} vs TT{tt_logits.shape}")
            return False
    else:
        print("⚠ Missing logits files")
        return False


def main():
    """Main debugging function."""
    print("=== Simple Model Debugging ===")
    
    # Debug both models
    hf_logits = debug_hf_model()
    tt_logits = debug_torchtune_model()
    
    # Compare at different levels
    embeddings_match = compare_embeddings()
    logits_match = compare_final_logits()
    
    # Detailed logit analysis
    analyze_logit_differences()
    
    print("\n=== DEBUGGING SUMMARY ===")
    if embeddings_match:
        print("✅ Token embeddings match")
        print("❌ Differences introduced in transformer layers")
        print("🔍 Next steps: Debug attention/MLP layers")
    else:
        print("❌ Token embeddings already differ")
        print("🔍 Next steps: Check weight conversion or tokenization")
    
    if logits_match:
        print("✅ Final logits match - models are equivalent!")
    else:
        print("❌ Final logits differ")
        print("🔍 Check the detailed analysis above for patterns")
        
    print(f"\n📁 Debug tensors saved to: /mnt/vast/home/lawrence/debug_tensors")


if __name__ == "__main__":
    main() 