import torch
from torchtune import utils
from torchtune.models.llama2._lora_llama2_builders import lora_llama2_7b
from torchtune import modules


def main():

    m = lora_llama2_7b(lora_attn_modules=["q_proj", "k_proj", "v_proj", "output_proj"])
    device = utils.get_device("cuda")

    m = utils.wrap_fsdp(m, device=device, dtype=torch.bfloat16, strategy="FULL_SHARD", auto_wrap_policy=utils.lora_fsdp_wrap_policy(modules_to_wrap={modules.TransformerDecoderLayer}),
            use_orig_params=True)

    loras = {k for k in m.modules() if hasattr(k, "_merge_lora_weights")}
    for l in loras:
        # Find the containing TransformerDecoderLayer for l
        containing_transformer = None
        all_transformers = {j for j in m.modules() if isinstance(j, modules.TransformerDecoderLayer)}
        for t in all_transformers:
            for submod in t.modules():
                if l is submod:
                    containing_transformer = t

        with torch.distributed.fsdp.FullyShardedDataParallel.summon_full_params(containing_transformer, recurse=False):
            l._merge_lora_weights()

    for l in loras:
        l._unmerge_lora_weights()

    print(f"done wrapping")




if __name__ == '__main__':
    torch.distributed.init_process_group(backend="gloo")
    main()
