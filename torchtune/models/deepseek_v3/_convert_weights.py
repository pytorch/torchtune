import torch
from torchtune.models.convert_weights import get_mapped_key
import regex as re
from typing import Dict

_FROM_HF = {
    "model.embed_tokens.weight": "tok_embeddings.weight",
    "model.layers.{}.input_layernorm.weight": "layers.{}.sa_norm.scale",
    "model.layers.{}.post_attention_layernorm.weight": "layers.{}.mlp_norm.scale",
    "model.norm.weight": "norm.scale",

    # attenion weights
    "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attn.q_proj.weight",
    "model.layers.{}.self_attn.q_a_proj.weight": "layers.{}.attn.q_proj.a.weight",
    "model.layers.{}.self_attn.q_a_layernorm.weight": "layers.{}.attn.q_proj.norm.scale",
    "model.layers.{}.self_attn.q_b_proj.weight": "layers.{}.attn.q_proj.b.weight",
    "model.layers.{}.self_attn.kv_a_proj_with_mqa.weight": "layers.{}.attn.kv_proj.a.weight",
    "model.layers.{}.self_attn.kv_a_layernorm.weight": "layers.{}.attn.kv_proj.norm.scale",
    "model.layers.{}.self_attn.kv_b_proj.weight": "layers.{}.attn.kv_proj.b.weight",
    "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attn.output_proj.weight",

    # mlp non-expert weights
    "model.layers.{}.mlp.gate_proj.weight": "layers.{}.mlp.w1.weight",
    "model.layers.{}.mlp.up_proj.weight": "layers.{}.mlp.w3.weight",
    "model.layers.{}.mlp.down_proj.weight": "layers.{}.mlp.w2.weight",

    # mlp MoE expert weights
    "model.layers.{}.mlp.experts.{}.gate_proj.weight": "layers.{}.mlp.experts.{}.w1.weight",
    "model.layers.{}.mlp.experts.{}.up_proj.weight": "layers.{}.mlp.experts.{}.w3.weight",
    "model.layers.{}.mlp.experts.{}.down_proj.weight": "layers.{}.mlp.experts.{}.w2.weight",

    # mlp MoE shared expert weights
    "model.layers.{}.mlp.shared_experts.gate_proj.weight": "layers.{}.mlp.shared_expert.w1.weight",
    "model.layers.{}.mlp.shared_experts.up_proj.weight": "layers.{}.mlp.shared_expert.w3.weight",
    "model.layers.{}.mlp.shared_experts.down_proj.weight": "layers.{}.mlp.shared_expert.w2.weight",

    # mlp MoE token router weights
    "model.layers.{}.mlp.gate.weight": "layers.{}.mlp.router.gate",
    "model.layers.{}.mlp.gate.e_score_correction_bias": "layers.{}.mlp.router.e_score_correction_bias",

    "lm_head.weight": "output.weight",
    "model.layers.{}.self_attn.rotary_emb.inv_freq": None,
}


def get_mapped_key(key: str, mapping_dict: Dict[str, str]) -> str:
    try:
        # Checks if there is a layer # in the key
        if any(k.isdigit() for k in key.split(".")):
            # Replace all numbers with "{}" to create key for lookup
            abstract_key = re.sub(r"(\.\d+)", ".{}", key)
            # Find all numbers in the key in order
            layer_nums = re.findall(r"\d+", key)
            new_key = mapping_dict[abstract_key]
            # Format with all numbers
            new_key = new_key.format(*layer_nums)
        else:
            new_key = mapping_dict[key]
    except KeyError as e:
        raise Exception(
            f'Error converting the state dict. Found unexpected key: "{key}". '
            "Please make sure you're loading a checkpoint with the right format. "
        ) from e

    return new_key


def deepseek_v3_hf_to_tune(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    converted_state_dict = {}
    for key, value in state_dict.items():
        # Skip keys that should be ignored (like rotary embeddings)
        if "rotary_emb.inv_freq" in key:
            continue

        new_key = get_mapped_key(key, _FROM_HF)
        converted_state_dict[new_key] = value
    return converted_state_dict
