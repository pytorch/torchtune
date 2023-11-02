# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from typing import Dict, Optional, Tuple

import torch

from llm.llama2.transformer import TransformerDecoder
from tests.llm.llama2.scripts.compare_decoder import Transformer

from torch import Tensor


def _is_qkv(s):
    return any(["attention.wq" in s, "attention.wk" in s, "attention.wv" in s])


def args_7b() -> Tuple[int, int, int, int, Optional[int], int]:
    return (32_000, 4096, 32, 32, None, 2048)


def load_orig_state_dict(path):
    orig_sd = torch.load(path)
    return orig_sd


def build_orig_fqn_to_native_map() -> Dict[str, Optional[str]]:
    # Map of original FQNs to native implementation FQNs.
    orig_fqn_to_native_fqn = {
        # Token embedding
        "tok_embeddings.weight": "tok_embeddings.weight",
        # Attention weights
        "layers.{}.attention.wq.weight": None,
        "layers.{}.attention.wk.weight": None,
        "layers.{}.attention.wv.weight": None,
        # Output norm and output weight
        "norm.weight": "norm.scale",
        "output.weight": "output.weight",
    }

    # attention norms
    orig_attn_norm_format = "layers.{}.attention_norm.weight"
    new_attn_norm_format = "layers.{}.attn_norm.scale"
    # ffn norm
    orig_ffn_norm_format = "layers.{}.ffn_norm.weight"
    new_ffn_norm_format = "layers.{}.ff_norm.scale"
    # ffn weights
    orig_ffn_weight_format = "layers.{}.feed_forward.w{}.weight"
    new_ffn_weight_format = "layers.{}.mlp.w{}.weight"
    # attention output proj
    orig_attn_output_proj_format = "layers.{}.attention.wo.weight"
    new_attn_output_proj_format = "layers.{}.attn.output_proj.weight"

    for layer_idx in range(n_layers):
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
    args = parser.parse_args()
    path = args.checkpoint_path
    print(f"RV: {path}", flush=True)
    vs, embed_dim, n_layers, n_heads, n_kv_heads, max_slen = args_7b()
    vocab_size = vs

    decoder = TransformerDecoder(
        vocab_size=vs,
        num_layers=n_layers,
        num_heads=n_heads,
        num_kv_heads=n_kv_heads,
        embed_dim=embed_dim,
        max_seq_len=max_slen,
    )

    tformer = Transformer(
        vocab_size=vs,
        dim=embed_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        max_seq_len=max_slen,
        n_kv_heads=n_kv_heads,
    )

    # Original state_dict to convert.
    orig_sd = load_orig_state_dict(path)
    # Reference state_dict for checking expected keys and tensor shapes.
    ref_sd = decoder.state_dict()
    # new state_dict that represents that conversion.
    new_state_dict = {}
    # Successfully processed keys from the original checkpoint, used for
    # validation.
    orig_sd_processed_keys = set()
    # Build a mapping of original FQN -> native FQN for key mapping.
    orig_fqn_to_native_fqn = build_orig_fqn_to_native_map()

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
                new_state_dict[mapped_key] = tensor.cpu().clone()
                # do some sanity checks around shape
                assert mapped_key in ref_sd, f"{mapped_key} not in reference state_dict"
                ref_sd_tensor = ref_sd[mapped_key]
                assert ref_sd_tensor.shape == new_state_dict[mapped_key].shape
                # Successfully processed key
                orig_sd_processed_keys.add(key)
            else:
                print(f"Warning: {key} in orig state_dict, but not mapped!")

    # sanity check qkv_dict to ensure each layer has qkv tensors.
    for i in range(n_layers):
        assert i in qkv_dict
        qkv_tensors = qkv_dict[i]
        assert "wq" in qkv_tensors
        assert "wk" in qkv_tensors
        assert "wv" in qkv_tensors

    qkv_tensors = qkv_dict
    # Go through qkv_tensors and batch qkv for torchTBD's batched implementation
    for key in qkv_tensors:
        head_dim = embed_dim // n_heads
        q = qkv_tensors[key]["wq"]
        k = qkv_tensors[key]["wk"]
        v = qkv_tensors[key]["wv"]
        q_per_kv = n_heads // (n_kv_heads if n_kv_heads else n_heads)
        qs = torch.split(q, head_dim * q_per_kv)
        ks = torch.split(k, head_dim)
        vs = torch.split(v, head_dim)
        cycled = [t for group in zip(qs, ks, vs) for t in group]
        qkv = torch.cat(cycled)
        qkv_tensors[key] = qkv.clone()

    # Map qkv tensors into the native state_dict
    for key in qkv_tensors:
        sd_key = f"layers.{key}.attn.qkv_proj.weight"
        new_state_dict[sd_key] = qkv_tensors[key].cpu().clone()
        # Validate name and shape
        assert sd_key in ref_sd, f"{sd_key} not in ref_sd!"
        assert ref_sd[sd_key].shape == new_state_dict[sd_key].shape
        # successfully processed the original qkv keys
        orig_sd_processed_keys.add(f"layers.{key}.attention.wq.weight")
        orig_sd_processed_keys.add(f"layers.{key}.attention.wv.weight")
        orig_sd_processed_keys.add(f"layers.{key}.attention.wk.weight")

    # Do some validation that 1) the only native keys we did not process are
    # RoPE related, as we aren't loading into RoPE, and 2) the only original
    # key we did not process is the saved rope.freqs buffer.
    unprocessed_native_keys = set(ref_sd.keys()) - set(new_state_dict.keys())
    # we aren't loading into RoPE
    assert all(["rope" in key for key in unprocessed_native_keys])

    unproc_orig_keys = set(orig_sd.keys()) - orig_sd_processed_keys
    assert (
        len(list(unproc_orig_keys)) == 1 and list(unproc_orig_keys)[0] == "rope.freqs"
    )

    # Load into decoder. We should not have any unexpected keys, and only be missing
    # rope-related params (which is expected)
    missing, unexpected = decoder.load_state_dict(new_state_dict, strict=False)
    assert not unexpected
    assert all(["rope" in key for key in missing])

    # Load the original state_dict into the reference implementation
    m, u = tformer.load_state_dict(orig_sd, strict=False)
    assert not m and u == ["rope.freqs"]

    # Validate equivalence.
    bsz, seqlen = 16, 128
    for _ in range(10):
        toks = torch.randint(low=0, high=vocab_size + 1, size=(bsz, seqlen))
        x = tformer(toks).sum()
        y = decoder(toks).sum()
        assert torch.allclose(x, y), f"{x} vs {y}"

    native_state_dict = decoder.state_dict()
    with open("/tmp/native_checkpoints/llama2-7b", "w+") as f:
        torch.save(native_state_dict, f)
