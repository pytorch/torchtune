# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Tuple

import torch

from torchtune.utils.logging import get_logger

from tqdm import tqdm

log = get_logger("DEBUG")


def _layer_template(layer_name: str, idx: int) -> Tuple[str, int]:
    """
    Template for mapping layer names.

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


def convert_llama2_fair_format(
    original_state_dict: Dict[str, torch.Tensor],
    output_numerical_validation: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Convert Llama2 state dict into TorchTune's native-PyTorch format. This function assumes
    the state dict has the same keys as the original checkpoint from FAIR. If the keys don't
    match the original checkpoint, conversion will fail.

    Args:
        original_state_dict (Dict[str, torch.Tensor]): State dict with keys corresponding to the original Llama2
            checkpoint.
        output_numerical_validation (bool): Whether to compare numerical parity between original and converted state dicts.
            Since this is a semi-expensive operation, it is disabled by default.

    Returns:
        state_dict: PyTorch-native state dict.

    Raises:
        Exception: If KeyError arises in the conversion. Likely due to incorrect checkpoint file.
    """
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
        try:
            if name not in ["rope.freqs"]:  # Skip loading the position embeddings
                if "layers" in name:
                    # Map the correct layer idx to the correct layer name
                    from_name, number = _layer_template(name, 1)
                    to_name = to_native_mapping[from_name].format(number)
                else:
                    to_name = to_native_mapping[name]
                state_dict[to_name] = param
        except KeyError as e:
            raise Exception(
                "Error converting the original Llama2 state dict into native PyTorch format. "
                f'The conversion found a key: "{name}", which is an unexpected key. '
                "Please make sure you're loading a Llama2 model and the keys match "
                "the original checkpoint from FAIR."
            ) from e

    if output_numerical_validation:
        log.info(
            "Running validation to ensure original checkpoint and converted checkpoint "
            "are numerically equivalent"
        )
        _run_numerical_validation(original_state_dict, state_dict)
        log.info("Numerical validation complete. All outputs match!")

    return state_dict


def _run_numerical_validation(
    original_state_dict: Dict[str, torch.Tensor],
    converted_state_dict: Dict[str, torch.Tensor],
):
    """Complete numerical validation on FAIR ckpt conversion for Llama2 7B.

    Args:
        original_state_dict (Dict[str, torch.Tensor]): Original state dict matching keys from a FAIR ckpt.
        converted_state_dict (Dict[str, torch.Tensor]): Converted Torchtune state dict.
    """

    from tests.torchtune.models.llama2.scripts.compare_decoder import Transformer
    from torchtune.models.llama2 import llama2

    # Generate random "toks" in the range [0, 32_000] following vocab size
    # bsz = 16
    # seq_len = 128
    nums = [torch.randint(0, 32_000, (16, 128)) for _ in range(1)]

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
    with torch.no_grad():
        for num in tqdm(
            nums, desc="[In validation] Running forward pass on original model."
        ):
            fair_outputs.append(fair_transfomer(num).sum())

    # Clean up the original model
    del fair_transfomer

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
    with torch.no_grad():
        for num in tqdm(
            nums, desc="[In validation] Running forward pass on converted model."
        ):
            native_outputs.append(native_transformer(num).sum())

    # Clean up the converted model
    del native_transformer

    # Compare the outputs
    for i, (fair_output, native_output) in enumerate(zip(fair_outputs, native_outputs)):
        torch.testing.assert_close(
            native_output,
            fair_output,
            rtol=1e-5,
            atol=1e-8,
            msg=f"[In validation] Outputs differ at index {i}. FAIR output: {fair_output}. Native output: {native_output}",
        )
