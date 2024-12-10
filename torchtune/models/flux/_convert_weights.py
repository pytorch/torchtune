# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import re
from typing import List

REGEX_CONVERSIONS = [
    (r"^(encoder|decoder)\.norm_out\.(weight|bias)$", r"\1.end.0.\2"),
    (r"^(encoder|decoder)\.conv_out\.(weight|bias)$", r"\1.end.2.\2"),
]

REGEX_UNCHANGED = [r"^(encoder|decoder)\.conv_in\.(weight|bias)$"]

RESNET_LAYER_CONVERSION = {
    "norm1": "main.0",
    "conv1": "main.2",
    "norm2": "main.3",
    "conv2": "main.5",
    "nin_shortcut": "shortcut",
}

ATTN_LAYER_CONVERSION = {
    "q": "attn.q_proj",
    "k": "attn.k_proj",
    "v": "attn.v_proj",
    "proj_out": "attn.output_proj",
    "norm": "norm",
}


def flux_ae_hf_to_tune(state_dict: dict) -> dict:
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = _convert_key(key)
        if "proj" in new_key:
            value = value.squeeze()
        new_state_dict[new_key] = value
    return new_state_dict


class ConversionError(Exception):
    pass


def _convert_key(key: str) -> str:
    # check if we should leave this key unchanged
    for pattern in REGEX_UNCHANGED:
        if re.match(pattern, key):
            return key

    # check if we can do a simple regex conversion
    for pattern, replacement in REGEX_CONVERSIONS:
        if re.match(pattern, key):
            return re.sub(pattern, replacement, key)

    # build the new key part-by-part
    parts = key.split(".")
    new_parts = []
    i = 0

    # add the first encoder/decoder model part unchanged
    model = parts[i]
    assert model in ["encoder", "decoder"]
    new_parts.append(model)
    i += 1

    # add the next mid/down/up section part unchanged
    section = parts[i]
    new_parts.append(section)
    i += 1

    # convert mid section keys
    if section == "mid":
        layer = parts[i]
        i += 1
        for layer_idx, layer_name in enumerate(["block_1", "attn_1", "block_2"]):
            if layer == layer_name:
                new_parts.append(str(layer_idx))
                if layer_name.startswith("attn"):
                    _convert_attn_layer(new_parts, parts, i)
                else:
                    _convert_resnet_layer(new_parts, parts, i)
                break
        else:
            raise ConversionError(key)

    # convert down section keys
    elif section == "down":
        new_parts.append(parts[i])  # add the down block idx
        i += 1
        if parts[i] == "block":
            new_parts.append("layers")
            i += 1
            new_parts.append(parts[i])  # add the resnet layer idx
            i += 1
            _convert_resnet_layer(new_parts, parts, i)
        elif parts[i] == "downsample":
            new_parts.extend(parts[i:])  # the downsampling layer is left unchanged
        else:
            raise ConversionError(key)

    # convert up section keys
    elif section == "up":
        # the first part in the "up" section is the block idx: one of [0, 1, 2, 3]
        # up blocks are in reverse order in the original state dict
        # so we need to convert [0, 1, 2, 3] -> [3, 2, 1, 0]
        new_parts.append(str(3 - int(parts[i])))
        i += 1
        if parts[i] == "block":
            new_parts.append("layers")
            i += 1
            new_parts.append(parts[i])  # add the resnet layer idx
            i += 1
            _convert_resnet_layer(new_parts, parts, i)
        elif parts[i] == "upsample":
            new_parts.extend(parts[i:])  # the upsampling layer is left unchanged
        else:
            raise ConversionError(key)

    else:
        raise ConversionError("unknown section:", key)

    return ".".join(new_parts)


def _convert_attn_layer(new_parts: List[str], parts: List[str], i: int):
    new_parts.append(ATTN_LAYER_CONVERSION[parts[i]])
    i += 1
    new_parts.append(parts[i])


def _convert_resnet_layer(new_parts: List[str], parts: List[str], i: int):
    new_parts.append(RESNET_LAYER_CONVERSION[parts[i]])
    i += 1
    new_parts.append(parts[i])
