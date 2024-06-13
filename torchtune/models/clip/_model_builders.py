from torchtune.modules.transforms.pipelines import VariableImageSizeTransforms

def build_image_processor():

    # Default for the pretrained model
    IMAGE_RES = 224
    IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
    IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)

    image_processor = VariableImageSizeTransforms(
        image_mean=IMAGE_MEAN,
        image_std=IMAGE_STD,
        patch_size=IMAGE_RES,
        possible_resolutions=None,
        max_num_chunks=4,
        resample="bilinear",
        do_rescale=True,
        rescale_factor=1 / 255,
        do_normalize=True,
        limit_upscaling_to_patch_size=True,
    )

    return image_processor
