#!/usr/bin/env python3
"""
Generate reference tensors for different input modalities to test MRoPE implementation.
"""

import os
import sys
import torch
from PIL import Image
import numpy as np

# Add transformers to path
transformers_path = "/mnt/vast/home/lawrence/inf2-training/3rdparty/torchtune/.venv/lib/python3.12/site-packages/transformers"
if transformers_path not in sys.path:
    sys.path.insert(0, transformers_path)

from transformers import AutoModel, AutoTokenizer, AutoProcessor
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

def create_dummy_image(width=224, height=224):
    """Create a dummy image for testing."""
    # Create a simple gradient image
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            image[i, j] = [i % 256, j % 256, (i + j) % 256]
    return Image.fromarray(image)

def create_dummy_video(frames=8, width=224, height=224):
    """Create a dummy video as a sequence of images."""
    video_frames = []
    for frame_idx in range(frames):
        # Create frames with different patterns
        image = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(height):
            for j in range(width):
                image[i, j] = [
                    (i + frame_idx * 10) % 256, 
                    (j + frame_idx * 20) % 256, 
                    (i + j + frame_idx * 30) % 256
                ]
        video_frames.append(Image.fromarray(image))
    return video_frames

def save_tensors_to_directory(tensor_dict, directory):
    """Save tensors to a specific directory."""
    os.makedirs(directory, exist_ok=True)
    for name, tensor in tensor_dict.items():
        torch.save(tensor, f"{directory}/{name}.pt")
    print(f"✓ Saved {len(tensor_dict)} tensors to {directory}")

def run_test_case(case_name, model, processor, inputs, base_path="/mnt/vast/home/lawrence/tensors"):
    """Run a test case and save the generated tensors."""
    print(f"\n=== Running {case_name} ===")
    
    # Create directory for this test case
    case_dir = f"{base_path}/{case_name}"
    
    try:
        # Run the model
        output = model(**inputs)
        print(f"✓ Model executed successfully")
        print(f"  Output keys: {list(output.keys()) if hasattr(output, 'keys') else 'No keys'}")
        
        # The tensors should be saved by the modified HuggingFace code
        # Let's check if they exist and move them to the case-specific directory
        
        # Expected tensor files from the HuggingFace modifications
        expected_tensors = [
            "position_ids", "rope_input_x", "rope_input_position_ids", 
            "rope_output_cos_sin", "position_embeddings", "mrope_input_q",
            "mrope_input_k", "mrope_input_cos", "mrope_input_sin", 
            "mrope_section", "q_embed", "k_embed"
        ]
        
        # Move tensors from base path to case-specific directory
        moved_tensors = {}
        for tensor_name in expected_tensors:
            src_path = f"{base_path}/{tensor_name}.pt"
            if os.path.exists(src_path):
                tensor = torch.load(src_path)
                moved_tensors[tensor_name] = tensor
                
        if moved_tensors:
            save_tensors_to_directory(moved_tensors, case_dir)
            
            # Clean up the base directory
            for tensor_name in expected_tensors:
                src_path = f"{base_path}/{tensor_name}.pt"
                if os.path.exists(src_path):
                    os.remove(src_path)
        else:
            print(f"⚠ No tensors found for {case_name}")
            
    except Exception as e:
        print(f"✗ Error running {case_name}: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function to run all test cases."""
    print("=== Qwen2.5-VL Multi-Modal MRoPE Reference Generator ===")
    
    # Load model and processor
    model_path = "/mnt/vast/share/inf2-training/models/open_source/Qwen2.5-VL-7B-Instruct"
    
    print("Loading model and processor...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(model_path)
    
    print("✓ Model and processor loaded")
    
    # Test Case 1: Text Only
    print("\n" + "="*50)
    text_only_messages = [
        {"role": "user", "content": [{"type": "text", "text": "Hello, how are you?"}]}
    ]
    text_only_inputs = processor.apply_chat_template(
        text_only_messages, tokenize=False, add_generation_prompt=True
    )
    text_only_processed = processor(text=[text_only_inputs], return_tensors="pt")
    
    run_test_case("text_only", model, processor, text_only_processed)
    
    # Test Case 2: Text + Image
    print("\n" + "="*50)
    image = create_dummy_image()
    text_image_messages = [
        {
            "role": "user", 
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What do you see in this image?"}
            ]
        }
    ]
    text_image_inputs = processor.apply_chat_template(
        text_image_messages, tokenize=False, add_generation_prompt=True
    )
    text_image_processed = processor(
        text=[text_image_inputs], 
        images=[image], 
        return_tensors="pt"
    )
    
    run_test_case("text_image", model, processor, text_image_processed)
    
    # Test Case 3: Text + Video
    print("\n" + "="*50)
    video_frames = create_dummy_video(frames=4)  # Short video for testing
    text_video_messages = [
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": "What happens in this video?"}
            ]
        }
    ]
    text_video_inputs = processor.apply_chat_template(
        text_video_messages, tokenize=False, add_generation_prompt=True
    )
    text_video_processed = processor(
        text=[text_video_inputs],
        videos=[video_frames],
        return_tensors="pt"
    )
    
    run_test_case("text_video", model, processor, text_video_processed)
    
    # Test Case 4: Text + Image + Video (if processor supports it)
    print("\n" + "="*50)
    try:
        mixed_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "video"}, 
                    {"type": "text", "text": "Compare this image and video."}
                ]
            }
        ]
        mixed_inputs = processor.apply_chat_template(
            mixed_messages, tokenize=False, add_generation_prompt=True
        )
        mixed_processed = processor(
            text=[mixed_inputs],
            images=[image],
            videos=[video_frames],
            return_tensors="pt"
        )
        
        run_test_case("text_image_video", model, processor, mixed_processed)
        
    except Exception as e:
        print(f"⚠ Mixed input test failed (may not be supported): {e}")
    
    print("\n" + "="*50)
    print("✓ Reference tensor generation complete!")
    print("Generated test cases:")
    print("  - text_only: Pure text input")
    print("  - text_image: Text + single image")
    print("  - text_video: Text + video sequence")
    print("  - text_image_video: Text + image + video (if supported)")
    
    print(f"\nTensors saved to: /mnt/vast/home/lawrence/tensors/{{case_name}}/")

if __name__ == "__main__":
    main()