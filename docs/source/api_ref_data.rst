.. _data:

==============
torchtune.data
==============

.. currentmodule:: torchtune.data

.. _chat_formats:

Text templates
--------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    InstructTemplate
    AlpacaInstructTemplate
    GrammarErrorCorrectionTemplate
    SummarizeTemplate
    StackExchangedPairedTemplate

    ChatFormat
    ChatMLFormat
    Llama2ChatFormat
    MistralChatFormat

Types
-----

.. autosummary::
    :toctree: generated/
    :nosignatures:

    Message

Converters
----------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    get_sharegpt_messages
    get_openai_messages

Helper funcs
------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    validate_messages
    truncate
