=================
torchtune.modules
=================

.. currentmodule:: torchtune.modules

Modeling Components and Building Blocks
---------------------------------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    MultiHeadAttention
    FeedForward
    KVCache
    RotaryPositionalEmbeddings
    RMSNorm
    Fp32LayerNorm
    TanhGate
    TiedLinear
    TransformerSelfAttentionLayer
    TransformerCrossAttentionLayer
    TransformerDecoder
    VisionTransformer

Losses
------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    loss.CEWithChunkedOutputLoss
    loss.ForwardKLLoss
    loss.ForwardKLWithChunkedOutputLoss

Base Tokenizers
---------------
Base tokenizers are tokenizer models that perform the direct encoding of text
into token IDs and decoding of token IDs into text. These are typically `byte pair
encodings <https://en.wikipedia.org/wiki/Byte_pair_encoding>`_ that underlie the
model specific tokenizers.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    tokenizers.SentencePieceBaseTokenizer
    tokenizers.TikTokenBaseTokenizer
    tokenizers.ModelTokenizer
    tokenizers.BaseTokenizer

Tokenizer Utilities
-------------------
These are helper methods that can be used by any tokenizer.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    tokenizers.tokenize_messages_no_special_tokens
    tokenizers.parse_hf_tokenizer_json


PEFT Components
---------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    peft.LoRALinear
    peft.AdapterModule
    peft.get_adapter_params
    peft.set_trainable_params
    peft.validate_missing_and_unexpected_for_lora
    peft.validate_state_dict_for_lora
    peft.disable_adapter


Fusion Components
-----------------
Components for building models that are a fusion of two+ pre-trained models.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    model_fusion.DeepFusionModel
    model_fusion.FusionLayer
    model_fusion.FusionEmbedding
    model_fusion.register_fusion_module
    model_fusion.get_fusion_params


Module Utilities
------------------
These are utilities that are common to and can be used by all modules.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   common_utils.reparametrize_as_dtype_state_dict_post_hook
   common_utils.local_kv_cache
   common_utils.disable_kv_cache
   common_utils.delete_kv_caches


Vision Transforms
------------------
Functions used for preprocessing images.

.. autosummary::
   :toctree: generated/
   :nosignatures:

    transforms.Transform
    transforms.VisionCrossAttentionMask
