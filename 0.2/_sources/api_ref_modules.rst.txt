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
    TransformerDecoderLayer
    TransformerDecoder

Tokenizers
------------------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    tokenizers.SentencePieceTokenizer
    tokenizers.TikTokenTokenizer
    tokenizers.Tokenizer

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
