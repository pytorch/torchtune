=================
torchtune.modules
=================

.. currentmodule:: torchtune.modules

Modeling Components and Building Blocks
---------------------------------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    CausalSelfAttention
    FeedForward
    KVCache
    get_cosine_schedule_with_warmup
    RotaryPositionalEmbeddings
    RMSNorm
    Fp32LayerNorm
    TransformerDecoderLayer
    TransformerDecoder
    VisionTransformer

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

Module Utilities
------------------
These are utilities that are common to and can be used by all modules.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   common_utils.reparametrize_as_dtype_state_dict_post_hook

Loss
------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   loss.DPOLoss
   loss.RSOLoss
   loss.IPOLoss


Vision Transforms
------------------
Functions used for preprocessing images.

.. autosummary::
   :toctree: generated/
   :nosignatures:

    transforms.get_canvas_best_fit
    transforms.resize_with_pad
    transforms.tile_crop
    transforms.find_supported_resolutions
    transforms.VisionCrossAttentionMask
