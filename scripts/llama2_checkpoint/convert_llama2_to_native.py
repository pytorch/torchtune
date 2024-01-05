# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import logging

import os

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from tests.torchtune.models.llama2.scripts.compare_decoder import Transformer

from torch import Tensor
from torchtune.models import llama2_7b

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LlamaArgs:
    vocab_size: int = 32_000
    embed_dim: int = 4096
    num_layers: int = 32
    num_heads: int = 32
    num_kv_heads: Optional[int] = None
    max_seq_len: int = 2048


def llama2_7b_args() -> LlamaArgs:
    return LlamaArgs(
        vocab_size=32_000,
        embed_dim=4096,
        num_layers=32,
        num_heads=32,
        num_kv_heads=32,
        max_seq_len=2048,
    )


def _is_qkv(s: str) -> bool:
    return any(["attention.wq" in s, "attention.wk" in s, "attention.wv" in s])


def load_orig_state_dict(path: str) -> Dict[str, Any]:
    orig_sd = torch.load(path, weights_only=True)
    return orig_sd


def build_orig_fqn_to_native_map(num_layers: int) -> Dict[str, Optional[str]]:
    # Map of original FQNs to native implementation FQNs.
    orig_fqn_to_native_fqn = {
        # Token embedding
        "tok_embeddings.weight": "tok_embeddings.weight",
        # Attention weights. They map to `None` since we separately handle
        # converting qkv tensors into the fused matrix used by the torchTBD
        # implementation, so they don't need to be mapped to a native FQN
        # here.
        "layers.{}.attention.wq.weight": None,
        "layers.{}.attention.wk.weight": None,
        "layers.{}.attention.wv.weight": None,
        # Output norm and output weight
        "norm.weight": "norm.scale",
        "output.weight": "output.weight",
    }

    # attention norms
    orig_attn_norm_format = "layers.{}.attention_norm.weight"
    new_attn_norm_format = "layers.{}.sa_norm.scale"
    # ffn norm
    orig_ffn_norm_format = "layers.{}.ffn_norm.weight"
    new_ffn_norm_format = "layers.{}.mlp_norm.scale"
    # ffn weights w1, w2, and w3
    orig_ffn_weight_format = "layers.{}.feed_forward.w{}.weight"
    new_ffn_weight_format = "layers.{}.mlp.w{}.weight"
    # attention output proj
    orig_attn_output_proj_format = "layers.{}.attention.wo.weight"
    new_attn_output_proj_format = "layers.{}.attn.output_proj.weight"

    for layer_idx in range(num_layers):
        # attn norm
        orig_fqn_to_native_fqn[
            orig_attn_norm_format.format(layer_idx)
        ] = new_attn_norm_format.format(layer_idx)
        # ffn norm
        orig_fqn_to_native_fqn[
            orig_ffn_norm_format.format(layer_idx)
        ] = new_ffn_norm_format.format(layer_idx)
        # ffn weight
        for i in range(1, 4):
            orig_fqn_to_native_fqn[
                orig_ffn_weight_format.format(layer_idx, i)
            ] = new_ffn_weight_format.format(layer_idx, i)
        # attn output proj
        orig_fqn_to_native_fqn[
            orig_attn_output_proj_format.format(layer_idx)
        ] = new_attn_output_proj_format.format(layer_idx)

    return orig_fqn_to_native_fqn


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Checkpoint converter to PyTorch native format.
            Please see the associated README in this directory for usage details."""
    )
    parser.add_argument(
        "--checkpoint_path", type=str, help="Path to original checkpoint file."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="""
        Device to initialize tensors on. This defaults to cuda:0 which is much faster
        than CPU for checkpoint conversion. If GPU is unavailable, pass in "cpu".
        """,
    )

    args = parser.parse_args()
    path = args.checkpoint_path
    torch.set_default_device(args.device)

    llama2_args = llama2_7b_args()

    # Initialize new decoder architecture
    decoder = llama2_7b()
    # Initialize original decoder architecture
    tformer = Transformer(
        vocab_size=llama2_args.vocab_size,
        dim=llama2_args.embed_dim,
        n_layers=llama2_args.num_layers,
        n_heads=llama2_args.num_heads,
        max_seq_len=llama2_args.max_seq_len,
        n_kv_heads=llama2_args.num_kv_heads,
    )

    # Original state_dict to convert.
    orig_sd = load_orig_state_dict(path)
    # Reference state_dict for checking expected keys and tensor shapes.
    ref_sd = decoder.state_dict()
    # new state_dict that represents that conversion.
    new_state_dict = {}
    # This set will contain the successfully processed keys from the original
    # checkpoint, used for validation by comparing to orig_sd.keys() to ensure
    # all expected keys have been processed.
    orig_sd_processed_keys = set()
    # Build a mapping of original FQN -> native FQN for key mapping.
    orig_fqn_to_native_fqn = build_orig_fqn_to_native_map(
        num_layers=llama2_args.num_layers
    )

    # qkv_dict will map a layer index to its qkv tensors.
    qkv_dict: Dict[int, Dict[str, Tensor]] = {}

    # Iterate through original state_dict, doing 1 of 2 things:
    # 1) if the tensor is a QKV tensor, save it in qkv_dict.
    # 2) Otherwise, copy over the tensor into the new state_dict, and validate
    # the key is expected and the shape is expected.
    for key, tensor in orig_sd.items():
        if _is_qkv(key):
            # Process QKV tensor.
            # Grab layer index from key. For example, key is
            # layers.0.attention.wk.weight.
            splits = key.split(".")
            layer_index = splits[1]
            layer_index = int(layer_index)
            if layer_index not in qkv_dict:
                qkv_dict[layer_index] = {}
            # Grab wq/wk/wv string from split string.
            weight_key = splits[-2]
            assert weight_key not in qkv_dict[layer_index]
            qkv_dict[layer_index][weight_key] = tensor
        else:
            # Process non QKV tensor. Here we can directly copy into the new state_dict.
            if key in orig_fqn_to_native_fqn:
                # Copy over
                mapped_key = orig_fqn_to_native_fqn[key]
                new_state_dict[mapped_key] = tensor.clone()
                # do some sanity checks around shape
                assert mapped_key in ref_sd, f"{mapped_key} not in reference state_dict"
                ref_sd_tensor = ref_sd[mapped_key]
                assert ref_sd_tensor.shape == new_state_dict[mapped_key].shape
                # Successfully processed key
                orig_sd_processed_keys.add(key)
            else:
                logger.warning(f"Warning: {key} in orig state_dict, but not mapped!")

    # sanity check qkv_dict to ensure each layer has qkv tensors.
    for i in range(llama2_args.num_layers):
        assert i in qkv_dict
        assert "wq" in qkv_dict[i]
        assert "wk" in qkv_dict[i]
        assert "wv" in qkv_dict[i]

    # Go through qkv_dict and batch qkv for torchTBD's batched implementation
    embed_dim = llama2_args.embed_dim
    num_heads = llama2_args.num_heads
    num_kv_heads = llama2_args.num_kv_heads
    for layer_idx in qkv_dict:
        # Map individual qkv matrices to the fused matrix. This approach is motived from Lightning AI:
        # https://github.com/Lightning-AI/lit-gpt/blob/main/scripts/convert_hf_checkpoint.py#L112
        head_dim = embed_dim // num_heads
        q = qkv_dict[layer_idx]["wq"]
        k = qkv_dict[layer_idx]["wk"]
        v = qkv_dict[layer_idx]["wv"]
        q_per_kv = num_heads // (num_kv_heads if num_kv_heads else num_heads)
        qs = torch.split(q, head_dim * q_per_kv)
        ks = torch.split(k, head_dim)
        vs = torch.split(v, head_dim)
        cycled = [t for group in zip(qs, ks, vs) for t in group]
        qkv = torch.cat(cycled)
        qkv_dict[layer_idx] = qkv

    # Map qkv tensors into the native state_dict. qkv_dict now contains
    # the fused QKV matrix for each layer, i.e. layer i's fused QKV matrix
    # is given by qkv_dict[i]. We map this into the state_dict to be loaded
    # into torchTBD's implementation by reconstructing the expected name based
    # on the layer index.
    for layer_idx in qkv_dict:
        sd_key = f"layers.{layer_idx}.attn.qkv_proj.weight"
        new_state_dict[sd_key] = qkv_dict[layer_idx].clone()
        # Validate name and shape
        assert sd_key in ref_sd, f"{sd_key} not in ref_sd!"
        assert ref_sd[sd_key].shape == new_state_dict[sd_key].shape
        # successfully processed the original qkv keys
        orig_sd_processed_keys.add(f"layers.{layer_idx}.attention.wq.weight")
        orig_sd_processed_keys.add(f"layers.{layer_idx}.attention.wv.weight")
        orig_sd_processed_keys.add(f"layers.{layer_idx}.attention.wk.weight")

    # Do some validation that 1) the only native keys we did not process are
    # RoPE related, as we aren't loading into RoPE, and 2) the only original
    # key we did not process is the saved rope.freqs buffer.
    unprocessed_native_keys = set(ref_sd.keys()) - set(new_state_dict.keys())
    # we aren't loading into RoPE
    assert all(
        [
            "pos_embeddings" in key or "kv_cache" in key
            for key in unprocessed_native_keys
        ]
    )

    unproc_orig_keys = set(orig_sd.keys()) - orig_sd_processed_keys
    assert (
        len(list(unproc_orig_keys)) == 1 and list(unproc_orig_keys)[0] == "rope.freqs"
    )

    # Load into decoder. We should not have any unexpected keys, and only be missing
    # rope-related params (which is expected)
    missing, unexpected = decoder.load_state_dict(new_state_dict, strict=False)
    assert not unexpected
    assert all(["pos_embeddings" in key or "kv_cache" in key for key in missing])

    # Load the original state_dict into the reference implementation
    missing_keys, unexpected_keys = tformer.load_state_dict(orig_sd, strict=False)
    # We don't expect any keys to be missing (not loaded into)
    assert not missing_keys

    # Validate equivalence.
    bsz, seqlen = 16, 128
    with torch.no_grad():
        for i in range(10):
            toks = torch.randint(
                low=0, high=llama2_args.vocab_size, size=(bsz, seqlen)
            )
            y = decoder(toks).sum()
            x = tformer(toks).sum()
            assert torch.allclose(x, y), f"{x} vs {y} @ {i}"

    native_state_dict = decoder.state_dict()

    # TODO: we'll make this configurable when we switch to torch.distributed.checkpoint
    # and enable scales other than 7b.
    native_dirpath = "/tmp/native_checkpoints"

    checkpoint_file = "llama2-7b"
    if not os.path.exists(native_dirpath):
        os.makedirs(native_dirpath)

    with open(os.path.join(native_dirpath, checkpoint_file), "w+") as f:
        torch.save(native_state_dict, f.name)

    logger.info(f"Wrote native checkpoint to {f.name}")
