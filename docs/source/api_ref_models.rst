================
torchtune.models
================


This page contains information on all model architectures available in the core torchtune library.

Llama3
------

.. currentmodule:: torchtune.models.llama3

Model architectures from the `Llama3 family <https://llama.meta.com/llama3/>`_.

Download the model
^^^^^^^^^^^^^^^^^^

Pre-trained models can be downloaded from the Hugging Face Hub with the following command:

.. code-block:: bash

    tune download meta-llama/Meta-Llama-3-8B-Instruct --output-dir /tmp/Meta-Llama-3-8B-Instruct --hf-token <ACCESS_TOKEN>

Model architectures
^^^^^^^^^^^^^^^^^^^

Default sizes
~~~~~~~~~~~~~

These are models utilizing the default sizes released by the Meta team.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    llama3_8b
    lora_llama3_8b
    qlora_llama3_8b
    llama3_70b
    lora_llama3_70b

Generic builders
~~~~~~~~~~~~~~~~

If you want to experiment with different sizes of models or train a model from scratch, we also provide generic builder
classes for our models. These with NOT work out-of-the-box with pretrained checkpoints.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    llama3
    lora_llama3

Tokenizer
^^^^^^^^^

.. autosummary::
    :toctree: generated/
    :nosignatures:

    llama3_tokenizer

Usage
^^^^^

.. code-block:: python

    from torchtune.models.llama3 import llama3_8b, llama3_tokenizer
    from torchtune.utils import FullModelMetaCheckpointer
    checkpointer = FullModelMetaCheckpointer(
        checkpoint_dir="/tmp/Meta-Llama-3-8B-Instruct/original",
        checkpoint_files=["consolidated.00.pth"],
        model_type="llama3",
        output_dir="/tmp/finetuned-model",
    )
    state_dict = checkpointer.load_checkpoint()
    model = llama3_8b()
    model.load_state_dict(state_dict)
    tokenizer = llama3_tokenizer(path="/tmp/Meta-Llama-3-8B-Instruct/original/tokenizer.model")


Llama2
------

.. currentmodule:: torchtune.models.llama2

Model architectures from the `Llama2 family <https://llama.meta.com/llama2/>`_.

Download the model
^^^^^^^^^^^^^^^^^^

Pre-trained models can be downloaded from the Hugging Face Hub with the following command:

.. code-block:: bash

    tune download meta-llama/Llama-2-7b-hf --output-dir /tmp/Llama-2-7b-hf --hf-token <ACCESS_TOKEN>

Model architectures
^^^^^^^^^^^^^^^^^^^

Default sizes
~~~~~~~~~~~~~

These are models utilizing the default sizes released by the Meta team.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    llama2_7b
    lora_llama2_7b
    qlora_llama2_7b
    llama2_13b
    lora_llama2_13b
    qlora_llama2_13b
    llama2_70b
    lora_llama2_70b

Generic builders
~~~~~~~~~~~~~~~~

If you want to experiment with different sizes of models or train a model from scratch, we also provide generic builder
classes for our models. These with NOT work out-of-the-box with pretrained checkpoints.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    llama2
    lora_llama2

Tokenizer
^^^^^^^^^

.. autosummary::
    :toctree: generated/
    :nosignatures:

    llama2_tokenizer

Usage
^^^^^

.. code-block:: python

    from torchtune.models.llama2 import llama2_7b, llama2_tokenizer
    from torchtune.utils import FullModelHFCheckpointer
    checkpointer = FullModelHFCheckpointer(
        checkpoint_dir="/tmp/Llama2-7b-hf",
        checkpoint_files=[
            "pytorch_model-00001-of-00002.bin",
            "pytorch_model-00002-of-00002.bin",
        ],
        model_type="llama2",
        output_dir="/tmp/finetuned-model",
    )
    state_dict = checkpointer.load_checkpoint()
    model = llama2_7b()
    model.load_state_dict(state_dict)
    tokenizer = llama2_tokenizer(path="/tmp/Llama2-7b-hf/tokenizer.model")


Phi-3
-----

.. currentmodule:: torchtune.models.phi3

Model architectures from the `Phi-3 mini family <https://news.microsoft.com/source/features/ai/the-phi-3-small-language-models-with-big-potential/>`_.

Download the model
^^^^^^^^^^^^^^^^^^

Pre-trained models can be download from the Hugging Face Hub with the following command:

.. code-block:: bash

    tune download microsoft/Phi-3-mini-4k-instruct --output-dir /tmp/Phi-3-mini-4k-instruct  --ignore-patterns "" --hf-token <HF_TOKEN>

Model architectures
^^^^^^^^^^^^^^^^^^^

Default sizes
~~~~~~~~~~~~~

These are models utilizing the default sizes released by the Microsoft team.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    phi3_mini
    lora_phi3_mini
    qlora_phi3_mini

Generic builders
~~~~~~~~~~~~~~~~

If you want to experiment with different sizes of models or train a model from scratch, we also provide generic builder
classes for our models. These with NOT work out-of-the-box with pretrained checkpoints.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    phi3
    lora_phi3

Tokenizer
^^^^^^^^^

.. autosummary::
    :toctree: generated/
    :nosignatures:

    phi3_mini_tokenizer

Usage
^^^^^

.. code-block:: python

    from torchtune.models.phi3 import phi3_mini, phi3_mini_tokenizer
    from torchtune.utils import FullModelHFCheckpointer
    checkpointer = FullModelHFCheckpointer(
        checkpoint_dir="/tmp/Phi-3-mini-4k-instruct",
        checkpoint_files=[
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ],
        model_type="phi3_mini",
        output_dir="/tmp/finetuned-model",
    )
    state_dict = checkpointer.load_checkpoint()
    model = phi3_mini()
    model.load_state_dict(state_dict)
    tokenizer = phi3_mini_tokenizer(path="/tmp/Phi-3-mini-4k-instruct/tokenizer.model")


Mistral
-------

.. currentmodule:: torchtune.models.mistral

Model architectures from `Mistral AI family <https://mistral.ai/technology/#models>`_.

Download the model
^^^^^^^^^^^^^^^^^^

Pre-trained models can be downloaded from the Hugging Face Hub with the following command:

.. code-block:: bash

    tune download mistralai/Mistral-7B-v0.1 --output-dir /tmp/Mistral-7B-v0.1 --hf-token <HF_TOKEN>

Model architectures
^^^^^^^^^^^^^^^^^^^

Default sizes
~~~~~~~~~~~~~

These are models utilizing the default sizes released by the MistralAI team.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    mistral_7b
    lora_mistral_7b
    qlora_mistral_7b

Generic builders
~~~~~~~~~~~~~~~~

If you want to experiment with different sizes of models or train a model from scratch, we also provide generic builder
classes for our models. These with NOT work out-of-the-box with pretrained checkpoints.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    mistral
    lora_mistral

Tokenizer
^^^^^^^^^

.. autosummary::
    :toctree: generated/
    :nosignatures:

    mistral_tokenizer

Usage
^^^^^

.. code-block:: python

    from torchtune.models.mistral import mistral_7b, mistral_tokenizer
    from torchtune.utils import FullModelHFCheckpointer
    checkpointer = FullModelHFCheckpointer(
        checkpoint_dir="/tmp/Mistral-7B-v0.1",
        checkpoint_files=[
            "pytorch_model-00001-of-00002.bin",
            "pytorch_model-00002-of-00002.bin",
        ],
        model_type="mistral",
        output_dir="/tmp/finetuned-model",
    )
    state_dict = checkpointer.load_checkpoint()
    model = mistral_7b()
    model.load_state_dict(state_dict)
    tokenizer = mistral_tokenizer(path="/tmp/Mistral-7B-v0.1/tokenizer.model")


Gemma
-----

.. currentmodule:: torchtune.models.gemma

Model architectures from the `Gemma family <https://blog.google/technology/developers/gemma-open-models/>`_.

Download the model
^^^^^^^^^^^^^^^^^^

Pre-trained models can be downloaded from the Hugging Face Hub with the following command:

.. code-block:: bash

    tune download google/gemma-2b --output-dir /tmp/gemma  --ignore-patterns "" --hf-token <ACCESS_TOKEN>

Model architectures
^^^^^^^^^^^^^^^^^^^

Default sizes
~~~~~~~~~~~~~

These are models utilizing the default sizes released by the Google team.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    gemma_2b

Generic builders
~~~~~~~~~~~~~~~~

If you want to experiment with different sizes of models or train a model from scratch, we also provide generic builder
classes for our models. These with NOT work out-of-the-box with pretrained checkpoints.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    gemma

Tokenizer
^^^^^^^^^

.. autosummary::
    :toctree: generated/
    :nosignatures:

    gemma_tokenizer

Usage
^^^^^

.. code-block:: python

    from torchtune.models.gemma import gemma_2b, gemma_tokenizer
    from torchtune.utils import FullModelHFCheckpointer
    checkpointer = FullModelHFCheckpointer(
        checkpoint_dir="/tmp/gemma",
        checkpoint_files=[
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",

        ],
        model_type="gemma",
        output_dir="/tmp/finetuned-model",
    )
    state_dict = checkpointer.load_checkpoint()
    model = gemma_2b()
    model.load_state_dict(state_dict)
    tokenizer = gemma_tokenizer(path="/tmp/gemma/tokenizer.model")
