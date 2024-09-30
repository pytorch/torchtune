.. _datasets_overview:

=================
Datasets Overview
=================
torchtune lets you fine-tune LLMs and VLMs using any dataset found on Hugging Face Hub, downloaded locally,
or on a remote url. We provide built-in dataset builders to help you quickly bootstrap your fine-tuning project
for workflows including instruct tuning, preference alignment, continued pretraining, and more. Beyond those, torchtune
enables full customizability on your dataset pipeline, letting you train on any data format or schema.

The following tasks are supported:

- Text supervised fine-tuning
    - :ref:`instruct_dataset_usage_label`
    - :ref:`chat_dataset_usage_label`
- Multimodal supervised fine-tuning
    - :ref:`multimodal_dataset_usage_label`
- RLHF
    - :ref:`preference_dataset_usage_label`
- Continued pre-training
    - :ref:`text_completion_dataset_usage_label`

Data pipeline
-------------
.. image:: /_static/img/torchtune_datasets.svg

From raw data samples to the model inputs in the training recipe, all torchtune datasets follow
the same pipeline:

1. Raw data is queried one sample at a time from a Hugging Face dataset, local file, or remote file
2. :ref:`message_transform_usage_label` convert the raw sample which can take any format into a list of torchtune
   :ref:`messages_usage_label`. Images are contained in the message object they are associated with.
3. :ref:`model_transform_usage_label` applies model-specific transforms to the messages, including tokenization (see :ref:`tokenizers_usage_label`),
   prompt templating (see :ref:`prompt_templates_usage_label`), image transforms, and anything else required for that particular model.
4. The collater packages the processed samples together in a batch and the batch is passed into the model during training.
