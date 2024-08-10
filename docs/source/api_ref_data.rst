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
    PromptTemplate
    PromptTemplateInterface
    ChatMLTemplate

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

Message transforms
------------------

Converts data from common schema and conversation JSON formats into a list of torchtune :class:`Message`.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    InputOutputToMessages
    ShareGPTToMessages
    JSONToMessages

Helper functions
----------------

Miscellaneous helper functions used in modifying data.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    validate_messages
    truncate
