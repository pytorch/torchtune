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

    AlpacaInstructTemplate
    GrammarErrorCorrectionTemplate
    SummarizeTemplate
    QuestionAnswerTemplate
    ChatMLTemplate
    Llama2ChatTemplate
    MistralChatTemplate

Types
-----

.. autosummary::
    :toctree: generated/
    :nosignatures:

    Message

Converters
----------

Converts data from common JSON formats into a torchtune :class:`Message`.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    ShareGptToMessages
    JsonToMessages

Helper funcs
------------

Miscellaneous helper functions used in modifying data.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    validate_messages
    truncate
