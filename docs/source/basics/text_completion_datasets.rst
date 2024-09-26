.. _text_completion_dataset_usage_label:

========================
Text-completion Datasets
========================


Text-completion datasets are typically used for continued pre-training paradigms which involve
fine-tuning a base model on an unstructured, unlabelled dataset in a self-supervised manner.

The primary entry point for fine-tuning with text completion datasets in torchtune :func:`~torchtune.datasets.text_completion`.
Text completion datasets are simply expected to contain a column, "text", which contains the text for each sample.


Example local text completion datasets
--------------------------------------

``.json`` format
^^^^^^^^^^^^^^^^

.. code-block:: bash

   # odyssey.json
   [
       {
           "input": "After we were clear of the river Oceanus, and had got out into the open sea, we went on till we reached the Aeaean island where there is dawn and sunrise as in other places. We then drew our ship on to the sands and got out of her on to the shore, where we went to sleep and waited till day should break."
       },
       {
           "input": "Then, when the child of morning, rosy-fingered Dawn, appeared, I sent some men to Circe's house to fetch the body of Elpenor. We cut firewood from a wood where the headland jutted out into the sea, and after we had wept over him and lamented him we performed his funeral rites. When his body and armour had been burned to ashes, we raised a cairn, set a stone over it, and at the top of the cairn we fixed the oar that he had been used to row with."
       }
   ]

.. code-block:: python

   from torchtune.models.llama3 import llama3_tokenizer
   from torchtune.datasets import text_completion_dataset

   m_tokenizer = llama3_tokenizer(
       path="/tmp/Meta-Llama-3.1-8B/original/tokenizer.model",
       max_seq_len=8192
   )

   ds = text_completion_dataset(
       tokenizer=m_tokenizer,
       source="json",
       column="input",
       data_files="odyssey.json",
       split="train",
   )
   tokenized_dict = ds[0]
   print(m_tokenizer.decode(tokenized_dict["tokens"]))
   # After we were clear of the river Oceanus, and had got out into the open sea,\
   # we went on till we reached the Aeaean island where there is dawn and sunrise \
   # as in other places. We then drew our ship on to the sands and got out of her on \
   # to the shore, where we went to sleep and waited till day should break.
   print(tokenized_dict["labels"])
   # [128000, 6153, 584, 1051, 2867, 315, 279, 15140, 22302, 355, 11, 323, 1047, \
   # 2751, 704, 1139, 279, 1825, 9581, 11, 584, 4024, 389, 12222, 584, 8813, 279, \
   # 362, 12791, 5420, 13218, 1405, 1070, 374, 39493, 323, 64919, 439, 304, 1023, \
   # 7634, 13, 1226, 1243, 24465, 1057, 8448, 389, 311, 279, 70163, 323, 2751, 704, \
   # 315, 1077, 389, 311, 279, 31284, 11, 1405, 584, 4024, 311, 6212, 323, 30315, \
   # 12222, 1938, 1288, 1464, 13, 128001]


This can also be accomplished via the yaml config:

.. code-block:: yaml

   # In config
   tokenizer:
     _component_: torchtune.models.llama3.llama3_tokenizer
     path: /tmp/Meta-Llama-3.1-8B/original/tokenizer.model
     max_seq_len: 8192

   dataset:
     _component_: torchtune.datasets.text_completion_dataset
     source: json
     data_files: odyssey.json
     column: input
     split: train

``.txt`` format
^^^^^^^^^^^^^^^

.. code-block:: text

   # odyssey.txt

   After we were clear of the river Oceanus, and had got out into the open sea, we went on till we reached the Aeaean island where there is dawn and sunrise as in other places. We then drew our ship on to the sands and got out of her on to the shore, where we went to sleep and waited till day should break.
   Then, when the child of morning, rosy-fingered Dawn, appeared, I sent some men to Circe's house to fetch the body of Elpenor. We cut firewood from a wood where the headland jutted out into the sea, and after we had wept over him and lamented him we performed his funeral rites. When his body and armour had been burned to ashes, we raised a cairn, set a stone over it, and at the top of the cairn we fixed the oar that he had been used to row with.


.. code-block:: python

   from torchtune.models.llama3 import llama3_tokenizer
   from torchtune.datasets import text_completion_dataset

   m_tokenizer = llama3_tokenizer(
       path="/tmp/Meta-Llama-3.1-8B/original/tokenizer.model",
       max_seq_len=8192
   )

   ds = text_completion_dataset(
       tokenizer=m_tokenizer,
       source="text",
       data_files="odyssey.txt",
       split="train",
   )
   # the outputs here are identical to above

Similarly, this can also be accomplished via the yaml config:

.. code-block:: yaml

   # In config
   tokenizer:
     _component_: torchtune.models.llama3.llama3_tokenizer
     path: /tmp/Meta-Llama-3.1-8B/original/tokenizer.model
     max_seq_len: 8192

   dataset:
     _component_: torchtune.datasets.text_completion_dataset
     source: text
     data_files: odyssey.txt
     split: train

Loading text completion datasets from Hugging Face
--------------------------------------------------

To load in a text completion dataset from Hugging Face you'll need to pass in the dataset repo name to ``source``. For most HF datasets, you will also need to specify the ``split``.

.. code-block:: python

    from torchtune.models.gemma import gemma_tokenizer
    from torchtune.datasets import text_completion_dataset

    g_tokenizer = gemma_tokenizer("/tmp/gemma-7b/tokenizer.model")
    ds = text_completion_dataset(
        tokenizer=g_tokenizer,
        source="wikimedia/wikipedia",
        split="train",
    )

.. code-block:: yaml

    # Tokenizer is passed into the dataset in the recipe so we don't need it here
    dataset:
      _component_: torchtune.datasets.text_completion_dataset
      source: wikimedia/wikipedia
      split: train


Built-in text completion datasets
---------------------------------
- :func:`~torchtune.datasets.cnn_dailymail_articles_dataset`
