# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""This file converts download checkpoints to a format compatible with Torchtune."""

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from tqdm import tqdm


def _layer_template(layer_name: str, idx: int) -> Tuple[str, int]:
    """Template for mapping layer names.
    Inspiration: https://github.com/Lightning-AI/lit-gpt/scripts/convert_hf_checkpoint.py

    Args:
        layer_name (str): Name of the layer.
        idx (int): Index of the layer.

    Returns:
        Tuple[str, int]: Tuple of the layer name and the layer number.
    """
    split = layer_name.split(".")
    number = int(split[idx])
    split[idx] = "{}"
    from_name = ".".join(split)
    return from_name, number


def _convert_llama_from_fair(checkpoint_path: Path) -> Dict[str, torch.Tensor]:
    """Convert a FAIR Llama2 model checkpoint to a PyTorch-native format.

    Args:
        checkpoint_path (Path): Path to the checkpoint.

    Returns:
        state_dict: PyTorch-native state dict.
    """
    # Load the original state dict
    original_state_dict = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    print("Loaded original state dict")

    # Construct mapping
    to_native_mapping = {
        # Map token embeddings
        "tok_embeddings.weight": "tok_embeddings.weight",
        # Map final layer norm
        "norm.weight": "norm.scale",
        # Map final output proj layer
        "output.weight": "output.weight",
        # Map attention weights
        "layers.{}.attention.wk.weight": "layers.{}.attn.k_proj.weight",
        "layers.{}.attention.wq.weight": "layers.{}.attn.q_proj.weight",
        "layers.{}.attention.wv.weight": "layers.{}.attn.v_proj.weight",
        "layers.{}.attention.wo.weight": "layers.{}.attn.output_proj.weight",
        # Map per-layer norms
        "layers.{}.attention_norm.weight": "layers.{}.sa_norm.scale",
        "layers.{}.ffn_norm.weight": "layers.{}.mlp_norm.scale",
        # Map feedforward weights
        "layers.{}.feed_forward.w1.weight": "layers.{}.mlp.w1.weight",
        "layers.{}.feed_forward.w2.weight": "layers.{}.mlp.w2.weight",
        "layers.{}.feed_forward.w3.weight": "layers.{}.mlp.w3.weight",
    }

    state_dict = {}
    # Map the weights
    for name, param in tqdm(original_state_dict.items(), desc="Mapping weights"):
        if name not in ["rope.freqs"]:  # Skip loading the position embeddings
            if "layers" in name:
                # Map the correct layer idx to the correct layer name
                from_name, number = _layer_template(name, 1)
                to_name = to_native_mapping[from_name].format(number)
            else:
                to_name = to_native_mapping[name]
            state_dict[to_name] = param

    return state_dict


def convert_checkpoint(checkpoint_path: Path, output_path: Optional[Path] = None):
    """Convert model checkpoint to a PyTorch-native format compatible with Torchtune.

    Args:
        checkpoint_path (Path): Path to the checkpoint path.
        output_path (Optional[Path]): Path to the output checkpoint.

    Raises:
        Exception: If KeyError arises in the conversion. Likely due to incorrect checkpoint file.
    """
    # Convert checkpoint
    try:
        state_dict = _convert_llama_from_fair(checkpoint_path)
    except KeyError as e:
        # Add helpful error message to cover common error
        # This will be changed to support other model checkpoints in the future
        raise Exception(
            "Error converting checkpoint. Are you sure this is a FAIR Llama2 checkpoint?"
        ) from e

    # Save the state dict
    if output_path is None:
        checkpoint_dir = checkpoint_path.parent
        output_path = checkpoint_dir / "native_pytorch_model.pt"
    torch.save(state_dict, output_path)
    print(f"Succesfully wrote PyTorch-native model checkpoint to {output_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-path", type=Path, help="Path to the checkpoint to convert."
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        help="Where to write the converted checkpoint."
        "Will default to the same directory as the original checkpoint if no arg is provided.",
        required=False,
        default=None,
    )
    args = parser.parse_args()
    convert_checkpoint(args.checkpoint_path, args.output_path)
