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

_PYTORCH_MODEL_FILENAME = "native_pytorch_model.pt"


def _complete_numerical_validation_on_fair_ckpt_conversion(
    original_ckpt: Path, converted_ckpt: Path
):
    """Complete numerical validation on FAIR ckpt conversion for Llama2 7B.

    Args:
        original_ckpt (Path): Path to the original FAIR checkpoint.
        converted_ckpt (Path): Path to the converted Torchtune checkpoint.

    Raises:
        AssertionError: If the outputs differ.
    """
    print("Completing numerical validation on FAIR ckpt conversion for Llama2 7B...")

    from tests.torchtune.models.llama2.scripts.compare_decoder import Transformer
    from torchtune.models.llama2 import llama2

    nums = [torch.randint(0, 32_000, (16, 128)) for _ in range(5)]

    # Load the original state dict
    original_state_dict = torch.load(
        original_ckpt, map_location="cpu", weights_only=True
    )
    fair_transfomer = Transformer(
        vocab_size=32_000,
        n_layers=32,
        n_heads=32,
        dim=4096,
        max_seq_len=2048,
        n_kv_heads=32,
    )
    fair_transfomer.load_state_dict(original_state_dict, strict=False)
    fair_transfomer.eval()

    fair_outputs = []
    for num in tqdm(
        nums, desc="[In validation] Running forward pass on original model."
    ):
        fair_outputs.append(fair_transfomer(num).detach().sum())

    # Clean up the original model
    del fair_transfomer

    # Load the converted state dict
    converted_state_dict = torch.load(
        converted_ckpt, map_location="cpu", weights_only=True
    )
    native_transformer = llama2(
        vocab_size=32_000,
        num_layers=32,
        num_heads=32,
        embed_dim=4096,
        max_seq_len=2048,
        num_kv_heads=32,
    )
    native_transformer.load_state_dict(converted_state_dict, strict=True)
    native_transformer.eval()

    native_outputs = []
    for num in tqdm(
        nums, desc="[In validation] Running forward pass on converted model."
    ):
        native_outputs.append(native_transformer(num).detach().sum())

    # Clean up the converted model
    del native_transformer

    # Compare the outputs
    for i, (fair_output, native_output) in enumerate(zip(fair_outputs, native_outputs)):
        if not torch.allclose(fair_output, native_output):
            raise AssertionError(
                f"[In validation] Outputs differ at index {i}. FAIR output: {fair_output}. Native output: {native_output}"
            )

    print(
        "Numerical validation on FAIR ckpt conversion for Llama2 7B complete. All outputs match!"
    )


def _layer_template(layer_name: str, idx: int) -> Tuple[str, int]:
    """Template for mapping layer names.

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


def convert_checkpoint(
    checkpoint_path: Path,
    output_path: Optional[Path] = None,
    output_numerical_validation: bool = False,
):
    """Convert model checkpoint to a PyTorch-native format compatible with Torchtune.

    Args:
        checkpoint_path (Path): Path to the checkpoint path.
        output_path (Optional[Path]): Path to the output checkpoint.
        output_numerical_validation (bool): Whether to run numerical validation on the converted checkpoint.

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
        output_path = checkpoint_dir / _PYTORCH_MODEL_FILENAME
    torch.save({"model": state_dict}, output_path)

    # Run numerical validation
    if output_numerical_validation:
        _complete_numerical_validation_on_fair_ckpt_conversion(
            checkpoint_path, output_path
        )

    print(f"Succesfully wrote PyTorch-native model checkpoint to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-path", type=Path, help="Path to the checkpoint to convert."
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        help="Where to write the converted checkpoint."
        "Will default to the same directory as the original checkpoint if no arg is provided"
        f"under the filename {_PYTORCH_MODEL_FILENAME}.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--output-numerical-validation",
        action="store_true",
        help="Whether to load the original checkpoint and the converted checkpoint and compare"
        "the numerical output of a forward pass to ensure that the conversion was successful."
        "Prints results to stdout. This additional check is only available for Llama2 7B."
        "This will take awhile and may consume lots of memory. If you see an OOM error,"
        "please disable this flag. Note: All our checkpoints conversions are already validated"
        "in unit tests for smaller checkpoints and integration tests for larger checkpoints."
        "This flag is primarily for debugging purposes.",
        required=False,
        default=False,
    )
    args = parser.parse_args()
    convert_checkpoint(
        args.checkpoint_path, args.output_path, args.output_numerical_validation
    )
