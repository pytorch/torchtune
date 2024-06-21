from torchtune.models.clip._component_builders import CLIPImageTransform

def clip_vit_336_transform():

    image_transform = CLIPImageTransform(
        image_mean=[0.48145466, 0.4578275, 0.40821073],
        image_std=[0.26862954, 0.26130258, 0.27577711],
        tile_size=336,
        possible_resolutions=None,
        max_num_tiles=4,
        resample="bilinear",
        limit_upscaling_to_tile_size=True,
    )

    return image_transform
