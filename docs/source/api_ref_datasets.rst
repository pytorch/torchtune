.. _datasets:

==================
torchtune.datasets
==================

.. currentmodule:: torchtune.datasets

For a detailed general usage guide, please see our :ref:`datasets tutorial <dataset_tutorial_label>`.


Text datasets
------------------

torchtune supports several widely used text-only datasets to help quickly bootstrap your fine-tuning.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    alpaca_dataset
    alpaca_cleaned_dataset
    grammar_dataset
    hh_rlhf_helpful_dataset
    samsum_dataset
    slimorca_dataset
    stack_exchange_paired_dataset
    cnn_dailymail_articles_dataset
    wikitext_dataset

Image + Text datasets
---------------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    multimodal.llava_instruct_dataset
    multimodal.the_cauldron_dataset

.. _dataset_builders:

Generic dataset builders
------------------------

torchtune also supports generic dataset builders for common formats like chat models and instruct models.
These are especially useful for specifying from a YAML config.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    instruct_dataset
    chat_dataset
    preference_dataset
    text_completion_dataset

Generic dataset classes
-----------------------

Class representations for the above dataset builders.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    TextCompletionDataset
    ConcatDataset
    PackedDataset
    PreferenceDataset
    SFTDataset
