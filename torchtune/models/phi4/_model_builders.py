from functools import partial
from typing import List, Optional

from torchtune.data._prompt_templates import _get_prompt_template, _TemplateType

from torchtune.models.phi3._component_builders import lora_phi3, phi3
from torchtune.models.phi4._tokenizer import Phi4Tokenizer

from torchtune.modules import TransformerDecoder
from torchtune.modules.peft import LORA_ATTN_MODULES
from torchtune.modules.tokenizers import parse_hf_tokenizer_json


"""
Model builders build specific instantiations using component builders. For example
the ``phi4_14b`` model builder uses the ``phi3`` component builder.
"""


def phi4_14b() -> TransformerDecoder:
    """
    Builder for creating the Phi4 (14B) Instruct Model.

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


def phi4_tokenizer(
    vocab_path: str = None,
    merges_path: str = None,
    special_tokens_path: Optional[str] = None,
    max_seq_len: Optional[int] = None,
    prompt_template: Optional[_TemplateType] = None,
    truncation_type: str = "right",
) -> Phi4Tokenizer:
    """
    Phi4 tokenizer.

    Args:
        vocab_path (str): Path to vocab.json.
        merges_path (str): Path to merges.txt.
        special_tokens_path (Optional[str]): Path to ``tokenizer.json`` from Hugging Face
            model files that contains all registered special tokens, or a local json file
            structured similarly. Default is None to use the canonical Phi4 special tokens.
        max_seq_len (Optional[int]): maximum sequence length for tokenizing a single list of messages,
            after which the input will be truncated. Default is None.
        prompt_template (Optional[_TemplateType]): optional specified prompt template.
            If a string, it is assumed to be the dotpath of a :class:`~torchtune.data.PromptTemplateInterface`
            class. If a dictionary, it is assumed to be a custom prompt template mapping role to the
            prepend/append tags.
        truncation_type (str): type of truncation to apply, either "left" or "right".
            Default is "right".

    Returns:
        Phi4Tokenizer: Instantiation of the Phi-4 (14B) tokenizer.
    """
    special_tokens = (
        parse_hf_tokenizer_json(special_tokens_path)
        if special_tokens_path is not None
        else None
    )
    template = (
        _get_prompt_template(prompt_template) if prompt_template is not None else None
    )
    return Phi4Tokenizer(
        vocab_path=vocab_path,
        merges_path=merges_path,
        special_tokens=special_tokens,
        max_seq_len=max_seq_len,
        prompt_template=template,
        truncation_type=truncation_type,
    )


def lora_phi4_14b(
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


qlora_phi4_14b = partial(lora_phi4_14b, quantize_base=True)
qlora_phi4_14b.__doc__ = """
Builder for creating a Phi4 (14B) model with QLoRA enabled. Base model weights in linear layers
that LoRA is applied to are quantized per the QLoRA paper: https://arxiv.org/abs/2305.14314.
Please see `lora_phi4_14b` for full API arguments.
"""
