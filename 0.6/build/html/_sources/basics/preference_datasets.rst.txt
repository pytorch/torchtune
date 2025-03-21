.. _preference_dataset_usage_label:

===================
Preference Datasets
===================


Preference datasets are used for reward modelling, where the downstream task is to fine-tune a base model
to capture some underlying human preferences. Currently, these datasets are used in torchtune with the
Direct Preference Optimization (DPO) `recipe <https://github.com/pytorch/torchtune/blob/main/recipes/lora_dpo_single_device.py>`_.

The ground-truth in preference datasets is usually the outcome of a binary comparison between two completions for the same prompt,
and where a human annotator has indicated that one completion is more preferable than the other, according to some pre-set criterion.
These prompt-completion pairs could be instruct style (single-turn, optionally with a single prompt), chat style (multi-turn), or
some other set of interactions between a user and model (e.g. free-form text completion).

The primary entry point for fine-tuning with preference datasets in torchtune with the DPO recipe is :func:`~torchtune.datasets.preference_dataset`.


Example local preference dataset
--------------------------------

.. code-block:: bash

   # my_preference_dataset.json
   [
       {
           "chosen_conversations": [
               {
                   "content": "What do I do when I have a hole in my trousers?",
                   "role": "user"
               },
               { "content": "Fix the hole.", "role": "assistant" }
           ],
           "rejected_conversations": [
               {
                   "content": "What do I do when I have a hole in my trousers?",
                   "role": "user"
               },
               { "content": "Take them off.", "role": "assistant" }
           ]
       }
   ]


.. code-block:: python

   from torchtune.models.mistral import mistral_tokenizer
   from torchtune.datasets import preference_dataset

    m_tokenizer = mistral_tokenizer(
        path="/tmp/Mistral-7B-v0.1/tokenizer.model",
        prompt_template="torchtune.models.mistral.MistralChatTemplate",
        max_seq_len=8192,
    )
   column_map = {
       "chosen": "chosen_conversations",
       "rejected": "rejected_conversations"
   }
   ds = preference_dataset(
       tokenizer=tokenizer,
       source="json",
       column_map=column_map,
       data_files="my_preference_dataset.json",
       train_on_input=False,
       split="train",
   )
   tokenized_dict = ds[0]
   print(m_tokenizer.decode(tokenized_dict["rejected_input_ids"]))
   # user\n\nWhat do I do when I have a hole in my trousers?assistant\n\nTake them off.
   print(tokenized_dict["rejected_labels"])
   # [-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100, -100,-100,\
   # -100,-100,-100,-100,-100,128006,78191,128007,271,18293,1124,1022,13,128009,-100]


This can also be accomplished via the yaml config:

.. code-block:: yaml

   # In config
   tokenizer:
     _component_: torchtune.models.mistral.mistral_tokenizer
     path: /tmp/Mistral-7B-v0.1/tokenizer.model
     prompt_template: torchtune.models.mistral.MistralChatTemplate
     max_seq_len: 8192

   dataset:
     _component_: torchtune.datasets.preference_dataset
     source: json
     data_files: my_preference_dataset.json
     column_map:
       chosen: chosen_conversations
       rejected: rejected_conversations
     train_on_input: False
     split: train

In this example, we've also shown how `column_map` can be used when the "chosen" and/or "rejected" column names differ from the corresponding columns in your dataset.

Preference dataset format
-------------------------

Preference datasets are expected to have two columns: *"chosen"*, which indicates the human annotator's preferred response, and *"rejected"*, indicating
the human annotator's dis-preferred response. Each of these columns should contain a list of messages with an identical prompt.
The list of messages could include a system prompt, an instruction, multiple turns between user and assistant, or tool calls/returns. Let's take a look at
Anthropic's helpfulness/harmlessness dataset `on Hugging Face <https://huggingface.co/datasets/RLHFlow/HH-RLHF-Helpful-standard>`_ as an example of a multi-turn
chat-style format:

.. code-block:: text

    | chosen                                | rejected                              |
    |---------------------------------------|---------------------------------------|
    |[{                                     |[{                                     |
    | "role": "user",                       | "role": "user",                       |
    | "content": "helping my granny with her| "content": "helping my granny with her|
    | mobile phone issue"                   | mobile phone issue"                   |
    | },                                    | },                                    |
    | {                                     | {                                     |
    | "role": "assistant",                  | "role": "assistant",                  |
    | "content": "I see you are chatting    | "content": "Well, the best choice here|
    | with your grandmother about an issue  | could be helping with so-called 'self-|
    | with her mobile phone. How can I      | management behaviors'. These are      |
    | help?"                                | things your grandma can do on her own |
    | },                                    | to help her feel more in control."    |
    | {                                     | }]                                    |
    | "role": "user",                       |                                       |
    | "content": "her phone is not turning  |                                       |
    | on"                                   |                                       |
    | },                                    |                                       |
    | {...},                                |                                       |
    |]                                      |                                       |

Currently, only JSON-format conversations are supported, as shown in the example above.
You can use this dataset out-of-the-box in torchtune through :func:`~torchtune.datasets.hh_rlhf_helpful_dataset`.

Loading preference datasets from Hugging Face
---------------------------------------------

To load in preference datasets from Hugging Face you'll need to pass in the dataset repo name to ``source``. For most HF datasets, you will also need to specify the ``split``.

.. code-block:: python

    from torchtune.models.gemma import gemma_tokenizer
    from torchtune.datasets import preference_dataset

    g_tokenizer = gemma_tokenizer("/tmp/gemma-7b/tokenizer.model")
    ds = chat_dataset(
        tokenizer=g_tokenizer,
        source="hendrydong/preference_700K",
        split="train",
    )

.. code-block:: yaml

    # Tokenizer is passed into the dataset in the recipe so we don't need it here
    dataset:
      _component_: torchtune.datasets.preference_dataset
      source: hendrydong/preference_700K
      split: train


Built-in preference datasets
----------------------------
- :func:`~torchtune.datasets.hh_rlhf_helpful_dataset`
- :func:`~torchtune.datasets.stack_exchange_paired_dataset`
