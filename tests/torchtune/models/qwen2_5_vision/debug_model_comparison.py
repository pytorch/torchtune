import torch
import os
from pathlib import Path

from torchtune.models.qwen2_5_vision._convert_weights import qwen2_5_vl_hf_to_tune
from torchtune.models.qwen2_5_vision._model_builders import qwen2_5_vl_7b
import safetensors.torch
from transformers import AutoProcessor, AutoModelForImageTextToText


class ModelDebugger:
    """Debug model differences by saving intermediate tensors at key points."""
    
    def __init__(self, debug_dir="/mnt/vast/home/lawrence/debug_tensors"):
        self.debug_dir = Path(debug_dir)
        self.debug_dir.mkdir(exist_ok=True)
        self.step_counter = 0
        
    def save_tensor(self, tensor, name, model_type="hf"):
        """Save a tensor with a descriptive name."""
        if tensor is None:
            return
        
        filename = f"step_{self.step_counter:03d}_{model_type}_{name}.pt"
        filepath = self.debug_dir / filename
        torch.save(tensor.detach().cpu(), filepath)
        print(f"Saved {name}: {tensor.shape} -> {filename}")
        
    def increment_step(self):
        """Move to next debugging step."""
        self.step_counter += 1
        
    def compare_tensors(self, step, name):
        """Compare HF and TorchTune tensors at a specific step."""
        hf_file = self.debug_dir / f"step_{step:03d}_hf_{name}.pt"
        tt_file = self.debug_dir / f"step_{step:03d}_torchtune_{name}.pt"
        
        if not (hf_file.exists() and tt_file.exists()):
            print(f"⚠ Missing files for step {step}, {name}")
            return False
            
        hf_tensor = torch.load(hf_file)
        tt_tensor = torch.load(tt_file)
        
        if hf_tensor.shape != tt_tensor.shape:
            print(f"❌ Shape mismatch at step {step}, {name}: HF{hf_tensor.shape} vs TT{tt_tensor.shape}")
            return False
            
        # Compare values
        max_diff = torch.max(torch.abs(hf_tensor - tt_tensor)).item()
        mean_diff = torch.mean(torch.abs(hf_tensor - tt_tensor)).item()
        close = torch.allclose(hf_tensor, tt_tensor, atol=1e-4, rtol=1e-4)
        
        status = "✅" if close else "❌"
        print(f"{status} Step {step}, {name}: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}, close={close}")
        
        return close


def add_debug_hooks(model, debugger, model_type="hf"):
    """Add forward hooks to save intermediate tensors."""
    
    def make_hook(layer_name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                # Handle multiple outputs (e.g., attention)
                for i, out in enumerate(output):
                    if isinstance(out, torch.Tensor):
                        debugger.save_tensor(out, f"{layer_name}_output_{i}", model_type)
            elif isinstance(output, torch.Tensor):
                debugger.save_tensor(output, f"{layer_name}_output", model_type)
        return hook
    
    # Add hooks to key layers
    hooks = []
    
    # For HuggingFace model
    if hasattr(model, 'model'):
        # Token embeddings
        if hasattr(model.model, 'embed_tokens'):
            hooks.append(model.model.embed_tokens.register_forward_hook(
                make_hook("embed_tokens")))
            
        # Transformer layers
        if hasattr(model.model, 'layers'):
            for i, layer in enumerate(model.model.layers[:3]):  # First 3 layers only
                # Self-attention
                if hasattr(layer, 'self_attn'):
                    hooks.append(layer.self_attn.register_forward_hook(
                        make_hook(f"layer_{i}_self_attn")))
                
                # MLP
                if hasattr(layer, 'mlp'):
                    hooks.append(layer.mlp.register_forward_hook(
                        make_hook(f"layer_{i}_mlp")))
                    
        # Final norm and output
        if hasattr(model.model, 'norm'):
            hooks.append(model.model.norm.register_forward_hook(
                make_hook("final_norm")))
                
    # For TorchTune model  
    elif hasattr(model, 'decoder'):
        # Token embeddings
        if hasattr(model.decoder, 'tok_embeddings'):
            hooks.append(model.decoder.tok_embeddings.register_forward_hook(
                make_hook("embed_tokens")))
            
        # Transformer layers
        if hasattr(model.decoder, 'layers'):
            for i, layer in enumerate(model.decoder.layers[:3]):  # First 3 layers only
                # Self-attention
                if hasattr(layer, 'attn'):
                    hooks.append(layer.attn.register_forward_hook(
                        make_hook(f"layer_{i}_self_attn")))
                
                # MLP
                if hasattr(layer, 'mlp'):
                    hooks.append(layer.mlp.register_forward_hook(
                        make_hook(f"layer_{i}_mlp")))
                        
        # Final norm and output
        if hasattr(model.decoder, 'norm'):
            hooks.append(model.decoder.norm.register_forward_hook(
                make_hook("final_norm")))
                
        if hasattr(model.decoder, 'output'):
            hooks.append(model.decoder.output.register_forward_hook(
                make_hook("final_output")))
    
    return hooks


def load_models():
    """Load both HuggingFace and TorchTune models."""
    print("Loading models...")
    
    # Load HF model
    hf_model_path = "/mnt/vast/share/inf2-training/models/open_source/Qwen2.5-VL-7B-Instruct"
    hf_processor = AutoProcessor.from_pretrained(hf_model_path)
    hf_model = AutoModelForImageTextToText.from_pretrained(hf_model_path)
    
    # Load TorchTune model
    tune_qwen = qwen2_5_vl_7b()
    
    state_dict = {}
    files = [f"{hf_model_path}/model-0000{i}-of-00005.safetensors" for i in range(1, 6)]
    for file in files:
        load_files_dict = safetensors.torch.load_file(file)
        state_dict.update(load_files_dict)

    converted = qwen2_5_vl_hf_to_tune(state_dict)
    tune_qwen.load_state_dict(converted)
    
    return hf_model, tune_qwen


def debug_model_comparison():
    """Main debugging function."""
    debugger = ModelDebugger()
    
    # Load models
    hf_model, tt_model = load_models()
    
    # Move to GPU and set eval mode
    device = "cuda"
    hf_model.eval().to(device)
    tt_model.eval().to(device)
    
    # Create test input
    input_ids = torch.tensor([[1, 2, 3, 4, 5]]).to(device)
    print(f"Input shape: {input_ids.shape}")
    
    # Add debug hooks
    print("Adding debug hooks...")
    hf_hooks = add_debug_hooks(hf_model, debugger, "hf")
    tt_hooks = add_debug_hooks(tt_model, debugger, "torchtune")
    
    print(f"Added {len(hf_hooks)} HF hooks, {len(tt_hooks)} TorchTune hooks")
    
    try:
        # Run HF model
        print("\n=== Running HuggingFace model ===")
        with torch.no_grad():
            hf_output = hf_model(input_ids)
            debugger.save_tensor(hf_output.logits, "final_logits", "hf")
        
        # Reset step counter for TorchTune
        debugger.step_counter = 0
        
        # Run TorchTune model
        print("\n=== Running TorchTune model ===")
        with torch.no_grad():
            tt_output = tt_model(input_ids)
            debugger.save_tensor(tt_output, "final_logits", "torchtune")
            
    except Exception as e:
        print(f"Error during model execution: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Remove hooks
        for hook in hf_hooks + tt_hooks:
            hook.remove()
    
    print(f"\n=== Debug tensors saved to {debugger.debug_dir} ===")
    print("Use compare_debug_tensors() to analyze differences")


def compare_debug_tensors(debug_dir="/mnt/vast/home/lawrence/debug_tensors"):
    """Compare all saved debug tensors."""
    debug_dir = Path(debug_dir)
    debugger = ModelDebugger(debug_dir)
    
    # Find all unique tensor names
    tensor_names = set()
    for file in debug_dir.glob("step_*_hf_*.pt"):
        parts = file.stem.split("_")
        name = "_".join(parts[3:])  # Everything after "step_XXX_hf_"
        tensor_names.add(name)
    
    print(f"Found {len(tensor_names)} tensor types to compare")
    
    # Compare each tensor type
    results = {}
    for name in sorted(tensor_names):
        print(f"\n--- Comparing {name} ---")
        
        # Find all steps for this tensor
        steps = []
        for file in debug_dir.glob(f"step_*_hf_{name}.pt"):
            step = int(file.stem.split("_")[1])
            steps.append(step)
        
        step_results = []
        for step in sorted(steps):
            result = debugger.compare_tensors(step, name)
            step_results.append(result)
            
        results[name] = step_results
        
        # Summary
        all_match = all(step_results)
        status = "✅ ALL MATCH" if all_match else "❌ DIFFERENCES FOUND"
        print(f"{status} for {name}")
    
    # Overall summary
    print(f"\n=== SUMMARY ===")
    for name, step_results in results.items():
        all_match = all(step_results)
        status = "✅" if all_match else "❌"
        print(f"{status} {name}: {sum(step_results)}/{len(step_results)} steps match")
    
    return results


if __name__ == "__main__":
    print("=== Model Debugging Tool ===")
    print("1. Running debug comparison...")
    debug_model_comparison()
    
    print("\n2. Comparing saved tensors...")
    results = compare_debug_tensors()
    
    print("\n✅ Debugging complete!")
    print("Check the debug_tensors directory for detailed comparisons") 