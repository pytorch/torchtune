from torchtune.models.clip._transforms import CLIPImageTransform

def clip_vit_224_transform():
    image_transform = CLIPImageTransform(
        image_mean=[0.48145466, 0.4578275, 0.40821073],
        image_std=[0.26862954, 0.26130258, 0.27577711],
        tile_size=224,
        possible_resolutions=None,
        max_num_tiles=1,
        resample="bilinear",
        resize_to_max_canvas=True,
    )

    return image_transform
