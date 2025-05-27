from collections import defaultdict
import torch
from torchtune.models.convert_weights import get_mapped_key
import regex as re
from typing import Dict
# hf_model
# DeepseekV3ForCausalLM(
#   (model): DeepseekV3Model(
#     (embed_tokens): Identity()
#     (layers): ModuleList(
#       (0): DeepseekV3DecoderLayer(
#         (self_attn): DeepseekV3Attention(
#           (q_a_proj): Linear(in_features=16, out_features=16, bias=False)
#           (q_a_layernorm): DeepseekV3RMSNorm((16,), eps=1e-06)
#           (q_b_proj): Linear(in_features=16, out_features=64, bias=False)
#           (kv_a_proj_with_mqa): Linear(in_features=16, out_features=32, bias=False)
#           (kv_a_layernorm): DeepseekV3RMSNorm((16,), eps=1e-06)
#           (kv_b_proj): Linear(in_features=16, out_features=64, bias=False)
#           (o_proj): Linear(in_features=32, out_features=16, bias=False)
#         )
#         (mlp): DeepseekV3MLP(
#           (gate_proj): Linear(in_features=16, out_features=32, bias=False)
#           (up_proj): Linear(in_features=16, out_features=32, bias=False)
#           (down_proj): Linear(in_features=32, out_features=16, bias=False)
#           (act_fn): SiLU()
#         )
#         (input_layernorm): DeepseekV3RMSNorm((16,), eps=1e-06)
#         (post_attention_layernorm): DeepseekV3RMSNorm((16,), eps=1e-06)
#       )
#       (1): DeepseekV3DecoderLayer(
#         (self_attn): DeepseekV3Attention(
#           (q_a_proj): Linear(in_features=16, out_features=16, bias=False)
#           (q_a_layernorm): DeepseekV3RMSNorm((16,), eps=1e-06)
#           (q_b_proj): Linear(in_features=16, out_features=64, bias=False)
#           (kv_a_proj_with_mqa): Linear(in_features=16, out_features=32, bias=False)
#           (kv_a_layernorm): DeepseekV3RMSNorm((16,), eps=1e-06)
#           (kv_b_proj): Linear(in_features=16, out_features=64, bias=False)
#           (o_proj): Linear(in_features=32, out_features=16, bias=False)
#         )
#         (mlp): DeepseekV3MoE(
#           (experts): ModuleList(
#             (0-255): 256 x DeepseekV3MLP(
#               (gate_proj): Linear(in_features=16, out_features=16, bias=False)
#               (up_proj): Linear(in_features=16, out_features=16, bias=False)
#               (down_proj): Linear(in_features=16, out_features=16, bias=False)
#               (act_fn): SiLU()
#             )
#           )
#           (gate): DeepseekV3TopkRouter()
#           (shared_experts): DeepseekV3MLP(
#             (gate_proj): Linear(in_features=16, out_features=16, bias=False)
#             (up_proj): Linear(in_features=16, out_features=16, bias=False)
#             (down_proj): Linear(in_features=16, out_features=16, bias=False)
#             (act_fn): SiLU()
#           )
#         )
#         (input_layernorm): DeepseekV3RMSNorm((16,), eps=1e-06)
#         (post_attention_layernorm): DeepseekV3RMSNorm((16,), eps=1e-06)
#       )
#     )
#     (norm): DeepseekV3RMSNorm((16,), eps=1e-06)
#     (rotary_emb): DeepseekV3RotaryEmbedding()
#   )
#   (lm_head): Linear(in_features=16, out_features=129280, bias=False)
# )
# TransformerDecoder(
#   (tok_embeddings): Identity()
#   (layers): ModuleList(
#     (0): TransformerSelfAttentionLayer(
#       (attn): DeepSeekV3Attention(
#         (q_proj): DeepSeekV3LatentLinear(
#           (a): Linear(in_features=16, out_features=16, bias=False)
#           (b): Linear(in_features=16, out_features=64, bias=False)
#           (norm): RMSNorm()
#         )
#         (kv_proj): DeepSeekV3LatentLinear(
#           (a): Linear(in_features=16, out_features=32, bias=False)
#           (b): Linear(in_features=16, out_features=64, bias=False)
#           (norm): RMSNorm()
#         )
#         (output_proj): Linear(in_features=32, out_features=16, bias=False)
#         (pos_embeddings): Identity()
#       )
#       (mlp): FeedForward(
#         (w1): Linear(in_features=16, out_features=32, bias=False)
#         (w2): Linear(in_features=32, out_features=16, bias=False)
#         (w3): Linear(in_features=16, out_features=32, bias=False)
#         (activation): SiLU()
#       )
#       (sa_norm): RMSNorm()
#       (mlp_norm): RMSNorm()
#       (sa_scale): Identity()
#       (mlp_scale): Identity()
#     )
#     (1): TransformerSelfAttentionLayer(
#       (attn): DeepSeekV3Attention(
#         (q_proj): DeepSeekV3LatentLinear(
#           (a): Linear(in_features=16, out_features=16, bias=False)
#           (b): Linear(in_features=16, out_features=64, bias=False)
#           (norm): RMSNorm()
#         )
#         (kv_proj): DeepSeekV3LatentLinear(
#           (a): Linear(in_features=16, out_features=32, bias=False)
#           (b): Linear(in_features=16, out_features=64, bias=False)
#           (norm): RMSNorm()
#         )
#         (output_proj): Linear(in_features=32, out_features=16, bias=False)
#         (pos_embeddings): Identity()
#       )
#       (mlp): MoE(
#         (experts): GroupedExperts()
#         (router): DeepSeekV3TokenChoiceTopKRouter(
#           (gate): Linear(in_features=16, out_features=256, bias=False)
#         )
#         (shared_expert): FeedForward(
#           (w1): Linear(in_features=16, out_features=16, bias=False)
#           (w2): Linear(in_features=16, out_features=16, bias=False)
#           (w3): Linear(in_features=16, out_features=16, bias=False)
#           (activation): SiLU()
#         )
#       )
#       (sa_norm): RMSNorm()
#       (mlp_norm): RMSNorm()
#       (sa_scale): Identity()
#       (mlp_scale): Identity()
#     )
#   )
#   (norm): RMSNorm()
#   (output): Linear(in_features=16, out_features=129280, bias=False)
# )


# state dict key mappings from HF's format to torchtune's format for DeepSeek V3
# Note: Conversion might require custom logic beyond simple key mapping,
# especially for kv_proj and MoE expert weights.
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
