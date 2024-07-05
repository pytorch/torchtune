.. _data:

==============
torchtune.data
==============

.. currentmodule:: torchtune.data

.. _chat_formats:

Text templates
--------------

Templates for instruct prompts and chat prompts. Includes some specific formatting for difference datasets
and models.

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
    Role

Converters
----------

Converts data from common JSON formats into a torchtune :class:`Message`.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    get_sharegpt_messages
    get_openai_messages

Helper funcs
------------

Miscellaneous helper functions used in modifying data.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    validate_messages
    truncate
