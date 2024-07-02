from typing import List, Optional

from torchtune.models.phi3._component_builders import phi3, lora_phi3
from torchtune.models.phi3._tokenizer import Phi3MiniTokenizer

from torchtune.modules import TransformerDecoder
from torchtune.modules.peft import LORA_ATTN_MODULES
from functools import partial


"""
Model builders build specific instantiations using component builders. For example
the ``phi3_mini`` model builder uses the ``phi3`` component builder.
"""


def phi3_mini() -> TransformerDecoder:
    """
    Builder for creating the Phi3 Mini 4K Instruct Model.
    Ref: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct

    Note:
        This model does not currently support 128K context length nor optimizations
        such as sliding window attention.

    Returns:
        TransformerDecoder: Instantiation of Phi3 Mini 4K Instruct Model
    """
    return phi3(
        vocab_size=32_064,
        num_layers=32,
        num_heads=32,
        num_kv_heads=32,
        embed_dim=3072,
        intermediate_dim=8192,
        max_seq_len=4096,
        attn_dropout=0.0,
        norm_eps=1e-5,
    )

def phi3_mini_tokenizer(path: str, special_tokens_path: Optional[str] = None) -> Phi3MiniTokenizer:
    """Phi-3 Mini tokenizer.
    Ref: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/blob/main/tokenizer_config.json

    Args:
        path (str): Path to the SPM tokenizer model.
        special_tokens_path (Optional[str]): Path to ``tokenizer.json`` from Hugging Face
            model files that contains all registered special tokens, or a local json file 
            structured similarly. Default is None to use the canonical Phi3 special tokens.

    Note:
        This tokenizer includes typical LM EOS and BOS tokens like
        <s>, </s>, and <unk>. However, to support chat completion,
        it is also augmented with special tokens like <endoftext>
        and <assistant>.

    Warning:
        Microsoft currently opts to ignore system messages citing better performance.
        See https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/discussions/51 for more details.

    Returns:
        Phi3MiniSentencePieceBaseTokenizer: Instantiation of the SPM tokenizer.
    """
    special_tokens = parse_hf_tokenizer_json(special_tokens_path) if special_tokens_path is not None else None
    return Phi3MiniTokenizer(path=path, special_tokens=special_tokens)


def lora_phi3_mini(
    lora_attn_modules: List[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool = False,
    apply_lora_to_output: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 16,
    quantize_base: bool = False,
) -> TransformerDecoder:
    """
    Builder for creating a Phi3 Mini (3.8b) model with LoRA enabled.

    The Phi3 defaults are the same as in :func:`~torchtune.models.phi3.phi3_mini`,
    while LoRA default params are based on
    https://github.com/tloen/alpaca-lora/blob/8bb8579e403dc78e37fe81ffbb253c413007323f/finetune.py#L41-L43.

    Args:
        lora_attn_modules (List[LORA_ATTN_MODULES]): list of which linear layers
            LoRA should be applied to in each self-attention block. Options are
            ``{"q_proj", "k_proj", "v_proj", "output_proj"}``.
        apply_lora_to_mlp (bool): whether to apply LoRA to the MLP in each transformer layer.
            Default: False
        apply_lora_to_output (bool): whether to apply LoRA to the model's final output projection.
            Default: False
        lora_rank (int): rank of each low-rank approximation
        lora_alpha (float): scaling factor for the low-rank approximation
        quantize_base (bool): Whether to quantize base model weights

    Returns:
        TransformerDecoder: Instantiation of Phi3 Mini model with LoRA applied
    """
    return lora_phi3(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        vocab_size=32_064,
        num_layers=32,
        num_heads=32,
        num_kv_heads=32,
        embed_dim=3072,
        intermediate_dim=8192,
        max_seq_len=4096,
        attn_dropout=0.0,
        norm_eps=1e-5,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        quantize_base=quantize_base,
    )


qlora_phi3_mini = partial(lora_phi3_mini, quantize_base=True)
qlora_phi3_mini.__doc__ = """
Builder for creating a Phi3 mini model with QLoRA enabled. Base model weights in linear layers
that LoRA is applied to are quantized per the QLoRA paper: https://arxiv.org/abs/2305.14314.
Please see `lora_phi3_mini` for full API arguments.
"""
