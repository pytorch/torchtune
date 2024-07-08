from torchtune.models.clip._component_builders import clip_vision_encoder
from torchtune.torchtune.models.flamingo._encoders import FlamingoVisionAdapter

def flamingo_vision_encoder(
    tile_size: int,
    patch_size: int,
    embed_dim: int,
    num_layers: int,
    num_heads: int,
    num_adapter_layers: int,
    proj_out: int,
    out_indices: Optional[List[int]] = None,
    max_num_tiles: int = 4,
    in_channels: int = 3,
    ) -> FlamingoVisionEncoder:

    clip = clip_vision_encoder(
        tile_size=tile_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        out_indices=out_indices,
        max_num_tiles=max_num_tiles,
        in_channels=in_channels,
        output_cls_projection=False,
    )

    # we concatenate the output with hidden layers
    # and project it to proj_out
    clip_emb_size = clip.get_image_tokens_per_tile()
    proj_in = clip_emb_size + (hidden_emb_size * len(num_adapter_layers))

    adapter = FlamingoVisionAdapter(
        embed_dim=embed_dim,
        num_layers=num_adapter_layers,
        num_heads=num_heads,
        proj_in=proj_in,
        proj_out=proj_out,
    )

    return FlamingoVisionEncoder(vision_encoder=clip, adapter=adapter)
