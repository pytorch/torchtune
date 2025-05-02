.. _data:

==============
torchtune.data
==============

.. currentmodule:: torchtune.data

Text templates
--------------

Templates for instruct prompts and chat prompts. Includes some specific formatting for difference datasets
and models.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    GrammarErrorCorrectionTemplate
    SummarizeTemplate
    QuestionAnswerTemplate
    PromptTemplate
    PromptTemplateInterface
    ChatMLTemplate

Types
-----

.. autosummary::
    :toctree: generated/
    :nosignatures:

    Message
    Role

.. _message_transforms_ref:

Message transforms
------------------

Converts data from common schema and conversation JSON formats into a list of torchtune :class:`Message`.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    InputOutputToMessages
    ShareGPTToMessages
    OpenAIToMessages
    ChosenRejectedToMessages
    AlpacaToMessages

Collaters
---------

Collaters used to collect samples into batches and handle any padding.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    padded_collate
    padded_collate_dpo
    padded_collate_packed
    padded_collate_sft
    padded_collate_tiled_images_and_mask
    left_pad_sequence

Helper functions
----------------

Miscellaneous helper functions used in modifying data.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    validate_messages
    truncate
    load_image
    format_content_with_images
    mask_messages
