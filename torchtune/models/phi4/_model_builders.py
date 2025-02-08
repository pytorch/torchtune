from typing import List, Optional

from torchtune.models.phi3._component_builders import phi3, lora_phi3
from torchtune.models.phi4._tokenizer import Phi4MiniTokenizer

from torchtune.modules import TransformerDecoder
from torchtune.modules.peft import LORA_ATTN_MODULES
from functools import partial
from torchtune.modules.tokenizers import parse_hf_tokenizer_json
from torchtune.data._prompt_templates import _TemplateType
from torchtune.data._prompt_templates import _get_prompt_template


"""
Model builders build specific instantiations using component builders. For example
the ``phi4`` model builder uses the ``phi4`` component builder.
"""


def phi4() -> TransformerDecoder:
    """
    Builder for creating the Phi4 (14B) Instruct Model.

    Note:
        This model does not currently support 128K context length nor optimizations
        such as sliding window attention.

    Returns:
        TransformerDecoder: Instantiation of Phi4 (14B) Instruct Model
    """
    return phi3(
        vocab_size=100_352,
        num_layers=40,
        num_heads=40,
        num_kv_heads=10,
        embed_dim=5120,
        intermediate_dim=17920,
        max_seq_len=16384,
        attn_dropout=0.0,
        norm_eps=1e-5,
    )

def phi4_tokenizer(vocab_path: str = None, merges_path: str = None, path: str = None, special_tokens_path: Optional[str] = None, max_seq_len: Optional[int] = None, prompt_template: Optional[_TemplateType] = None) -> Phi4MiniTokenizer:
    """Phi4 (14B) tokenizer.
    Args:
        path (str): Path to the tiktoken tokenizer model.
        special_tokens_path (Optional[str]): Path to ``tokenizer.json`` from Hugging Face
            model files that contains all registered special tokens, or a local json file 
            structured similarly. Default is None to use the canonical Phi4 special tokens.
        max_seq_len (Optional[int]): maximum sequence length for tokenizing a single list of messages,
            after which the input will be truncated. Default is None.
        prompt_template (Optional[_TemplateType]): optional specified prompt template.
            If a string, it is assumed to be the dotpath of a :class:`~torchtune.data.PromptTemplateInterface`
            class. If a dictionary, it is assumed to be a custom prompt template mapping role to the
            prepend/append tags.

    Returns:
        Phi4MiniTikTokenTokenizer: Instantiation of the TikToken tokenizer.
    """
    special_tokens = parse_hf_tokenizer_json(special_tokens_path) if special_tokens_path is not None else None
    template = _get_prompt_template(prompt_template) if prompt_template is not None else None
    return Phi4MiniTokenizer(vocab_path=vocab_path, merges_path=merges_path, path=path, special_tokens=special_tokens, max_seq_len=max_seq_len, prompt_template=template)


def lora_phi4(
    lora_attn_modules: List[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool = False,
    apply_lora_to_output: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 16,
    lora_dropout: float = 0.0,
    use_dora: bool = False,
    quantize_base: bool = False,
) -> TransformerDecoder:
    """
    Builder for creating a Phi4 (14b) model with LoRA enabled.

    The Phi4 defaults are the same as in :func:`~torchtune.models.phi4.phi4`.

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
        lora_dropout (float): dropout probability for the low-rank approximation. Default: 0.0
        use_dora (bool): Decompose the LoRA weight into magnitude and direction, as
            introduced in "DoRA: Weight-Decomposed Low-Rank Adaptation" (https://arxiv.org/abs/2402.09353).
        quantize_base (bool): Whether to quantize base model weights

    Returns:
        TransformerDecoder: Instantiation of Phi4 (14B) model with LoRA applied
    """
    return lora_phi3(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        vocab_size=100_352,
        num_layers=40,
        num_heads=40,
        num_kv_heads=10,
        embed_dim=5120,
        intermediate_dim=17920,
        max_seq_len=16384,
        attn_dropout=0.0,
        norm_eps=1e-5,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        quantize_base=quantize_base,
    )


qlora_phi4 = partial(lora_phi4, quantize_base=True)
qlora_phi4.__doc__ = """
Builder for creating a Phi4 (14B) model with QLoRA enabled. Base model weights in linear layers
that LoRA is applied to are quantized per the QLoRA paper: https://arxiv.org/abs/2305.14314.
Please see `lora_phi4` for full API arguments.
"""
