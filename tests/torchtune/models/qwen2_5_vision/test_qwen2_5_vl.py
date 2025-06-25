import torch

from torchtune.models.qwen2_5_vision._convert_weights import qwen2_5_vl_hf_to_tune
from torchtune.models.qwen2_5_vision._model_builders import qwen2_5_vl_7b

import safetensors.torch
from transformers import AutoProcessor, AutoModelForImageTextToText


#--------------------------------
# load HF model
def load_hf_model():
    hf_model_path = "/mnt/vast/share/inf2-training/models/open_source/Qwen2.5-VL-7B-Instruct"
    hf_processor = AutoProcessor.from_pretrained(hf_model_path)
    hf_model = AutoModelForImageTextToText.from_pretrained(hf_model_path)

    return hf_processor, hf_model

#--------------------------------
# load TorchTune model
def load_tune_model():
    tune_qwen = qwen2_5_vl_7b()
    tune_model_path = "/mnt/vast/share/inf2-training/models/open_source/Qwen2.5-VL-7B-Instruct"

    state_dict = {}
    files = [f"{tune_model_path}/model-0000{i}-of-00005.safetensors" for i in range(1, 6)]
    for file in files:
        load_files_dict = safetensors.torch.load_file(file)
        state_dict.update(load_files_dict)

    converted = qwen2_5_vl_hf_to_tune(state_dict)

    # load the vision encoder weights
    tune_qwen.load_state_dict(converted)

    return tune_qwen

# load transform
# tune_transform = qwen2_5_vl_transform(
#     path=tune_model_path,
#     special_tokens_path=hf_model_path,
# )

#--------------------------------
# compare logits

def compare_logits(tune_model, hf_model, input_ids, tolerance=1e-4):
    """
    Compare logits between two models on the same input.
    
    Args:
        modelA: First model (e.g., HF model)
        modelB: Second model (e.g., TorchTune model)  
        input_ids: Input token IDs
        tolerance: Numerical tolerance for comparison
    
    Returns:
        bool: True if logits match within tolerance
    """
    # Set models to eval mode
    hf_model.eval().to("cuda")
    tune_model.eval().to("cuda")

    
    with torch.no_grad():
        # Forward pass through both models
        outputA = tune_model(input_ids)
        outputB = hf_model(input_ids)
        
        # Extract logits (handle different output formats)
        if hasattr(outputA, 'logits'):
            logitsA = outputA.logits
        else:
            logitsA = outputA
            
        if hasattr(outputB, 'logits'):
            logitsB = outputB.logits
        else:
            logitsB = outputB
        
        # Compare logits
        matches = torch.allclose(logitsA, logitsB, atol=tolerance, rtol=tolerance)
        
        # Print some debug info
        print(f"Model A logits shape: {logitsA.shape}")
        print(f"Model B logits shape: {logitsB.shape}")
        print(f"Max absolute difference: {torch.max(torch.abs(logitsA - logitsB)).item():.6f}")
        print(f"Logits match within tolerance {tolerance}: {matches}")
        
        return matches


def test_basic_comparison():
    """
    Simple test to compare HF and TorchTune models on dummy input.
    """
    # Create simple input (just a few tokens)
    input_ids = torch.tensor([[1, 2, 3, 4, 5]]).to("cuda")  # dummy token IDs
    
    hf_processor, hf_model = load_hf_model()
    print("Loaded HF model")
    tune_qwen = load_tune_model()
    print("Loaded TorchTune model")

    print("Testing basic model comparison...")
    result = compare_logits(tune_qwen, hf_model, input_ids)
    
    if result:
        print("Models produce matching logits!")
    else:
        print("Models produce different logits")
    
    return result

def test_tune_model():
    tune_qwen = load_tune_model()
    tune_qwen.eval().to("cuda")
    print("Loaded TorchTune model")
    
    input_ids = torch.tensor([[1, 2, 3, 4, 5]]).to("cuda")  # dummy token IDs
    output = tune_qwen(input_ids)
    print(output)

if __name__ == "__main__":
    test_basic_comparison()
    # test_tune_model()



