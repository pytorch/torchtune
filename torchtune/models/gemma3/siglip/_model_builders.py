from _component_builders import SiglipVisionModel


def siglip_vision_model() -> SiglipVisionModel:
    """
    Builds siglip vision encoder for the gemma3.
    """
    
    return SiglipVisionModel(
        num_hidden_layers=27,
        embed_dim=1152,
        embedding_use_bias=True,
        input_channels=3,
        head_dim=72,
        image_size=896,
        patch_size=14,
        num_heads=16,
        intermediate_dim=4304,
        layer_norm_eps=1e-6,
    )
