.. _models:

================
torchtune.models
================

.. currentmodule:: torchtune.models

llama3.2
--------

Text-only models from the 3.2 version of `Llama3 family <https://llama.meta.com/llama3/>`_.

Important: You need to request access on `Hugging Face <https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct>`__ before downloading it.

To download the Llama-3.2-1B-Instruct model:

.. code-block:: bash

    tune download meta-llama/Llama-3.2-1B-Instruct --output-dir /tmp/Llama-3.2-1B-Instruct --ignore-patterns "original/consolidated.00.pth" --hf-token <HF_TOKEN>

To download the Llama-3.2-3B-Instruct model:

.. code-block:: bash

    tune download meta-llama/Llama-3.2-3B-Instruct --output-dir /tmp/Llama-3.2-3B-Instruct --ignore-patterns "original/consolidated*" --hf-token <HF_TOKEN>

.. autosummary::
    :toctree: generated/
    :nosignatures:

    llama3_2.llama3_2_1b
    llama3_2.llama3_2_3b
    llama3_2.lora_llama3_2_1b
    llama3_2.lora_llama3_2_3b
    llama3_2.qlora_llama3_2_1b
    llama3_2.qlora_llama3_2_3b

.. note::

    The Llama3.2 tokenizer reuses the :class:`~torchtune.models.llama3.llama3_tokenizer` class.

llama3.2 Vision
---------------

Vision-Language Models from the 3.2 version of `Llama3 family <https://llama.meta.com/llama3/>`_.

Important: You need to request access on `Hugging Face <https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct>`__ before downloading it.

To download the Llama-3.2-11B-Instruct model:

.. code-block:: bash

    tune download meta-llama/Llama-3.2-11B-Vision-Instruct --output-dir /tmp/Llama-3.2-11B-Vision-Instruct --hf-token <HF_TOKEN>

.. autosummary::
    :toctree: generated/
    :nosignatures:

    llama3_2_vision.llama3_2_vision_11b
    llama3_2_vision.llama3_2_vision_transform
    llama3_2_vision.lora_llama3_2_vision_11b
    llama3_2_vision.qlora_llama3_2_vision_11b
    llama3_2_vision.llama3_2_vision_decoder
    llama3_2_vision.llama3_2_vision_encoder
    llama3_2_vision.lora_llama3_2_vision_decoder
    llama3_2_vision.lora_llama3_2_vision_encoder
    llama3_2_vision.Llama3VisionEncoder
    llama3_2_vision.Llama3VisionProjectionHead
    llama3_2_vision.Llama3VisionTransform

.. note::

    The Llama3.2 tokenizer reuses the :class:`~torchtune.models.llama3.llama3_tokenizer` class.

llama3 & llama3.1
-----------------

Models 3 and 3.1 from the `Llama3 family <https://llama.meta.com/llama3/>`_.

Important: You need to request access on `Hugging Face <https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct>`__ before downloading it.

To download the Llama3.1-8B-Instruct model:

.. code-block:: bash

    tune download meta-llama/Meta-Llama-3.1-8B-Instruct --output-dir /tmp/Meta-Llama-3.1-8B-Instruct --ignore-patterns "original/consolidated.00.pth" --hf-token <HF_TOKEN>

To download the Llama3.1-70B-Instruct model:

.. code-block:: bash

    tune download meta-llama/Meta-Llama-3.1-70B-Instruct --output-dir /tmp/Meta-Llama-3.1-70B-Instruct --ignore-patterns "original/consolidated*" --hf-token <HF_TOKEN>

To download the Llama3.1-405B-Instruct model:

.. code-block:: bash

    tune download meta-llama/Meta-Llama-3.1-405B-Instruct --ignore-patterns "original/consolidated*" --hf-token <HF_TOKEN>

To download the Llama3 weights of the above models, you can instead download from `Meta-Llama-3-8B-Instruct` and
`Meta-Llama-3-70B-Instruct`.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    llama3.llama3
    llama3.lora_llama3
    llama3.llama3_8b
    llama3.lora_llama3_8b
    llama3.qlora_llama3_8b
    llama3.llama3_70b
    llama3.lora_llama3_70b
    llama3.qlora_llama3_70b
    llama3.llama3_tokenizer

    |

    llama3_1.llama3_1
    llama3_1.lora_llama3_1
    llama3_1.llama3_1_8b
    llama3_1.lora_llama3_1_8b
    llama3_1.qlora_llama3_1_8b
    llama3_1.llama3_1_70b
    llama3_1.lora_llama3_1_70b
    llama3_1.qlora_llama3_1_70b
    llama3_1.llama3_1_405b
    llama3_1.lora_llama3_1_405b
    llama3_1.qlora_llama3_1_405b


.. note::

    The Llama3.1 tokenizer reuses the `llama3.llama3_tokenizer` builder class.

llama2
------

All models from the `Llama2 family <https://llama.meta.com/llama2/>`_.

Important: You need to request access on `Hugging Face <https://huggingface.co/meta-llama/Llama-2-7b-hf>`__ before downloading it.

To download the Llama2-7B model:

.. code-block:: bash

   tune download meta-llama/Llama-2-7b-hf --output-dir /tmp/Llama-2-7b-hf --hf-token <HF_TOKEN>

To download the Llama2-13B model:

.. code-block:: bash

    tune download meta-llama/Llama-2-13b-hf --output-dir /tmp/Llama-2-13b-hf --hf-token <HF_TOKEN>

To download the Llama2-70B model:

.. code-block:: bash

    tune download meta-llama/Llama-2-70b-hf --output-dir /tmp/Llama-2-70b-hf --hf-token <HF_TOKEN>

.. autosummary::
    :toctree: generated/
    :nosignatures:

    llama2.llama2
    llama2.lora_llama2
    llama2.llama2_7b
    llama2.lora_llama2_7b
    llama2.qlora_llama2_7b
    llama2.llama2_13b
    llama2.lora_llama2_13b
    llama2.qlora_llama2_13b
    llama2.llama2_70b
    llama2.lora_llama2_70b
    llama2.qlora_llama2_70b
    llama2.llama2_tokenizer
    llama2.llama2_reward_7b
    llama2.lora_llama2_reward_7b
    llama2.qlora_llama2_reward_7b
    llama2.Llama2ChatTemplate


code llama
----------

Models from the `Code Llama family <https://arxiv.org/pdf/2308.12950>`_.

Important: You need to request access on `Hugging Face <https://huggingface.co/meta-llama/CodeLlama-7b-hf>`__ before downloading it.

To download the CodeLlama-7B model:

.. code-block:: bash

    tune download meta-llama/CodeLlama-7b-hf --output-dir /tmp/CodeLlama-7b-hf --hf-token <HF_TOKEN>

.. autosummary::
    :toctree: generated/
    :nosignatures:

    code_llama2.code_llama2_7b
    code_llama2.lora_code_llama2_7b
    code_llama2.qlora_code_llama2_7b
    code_llama2.code_llama2_13b
    code_llama2.lora_code_llama2_13b
    code_llama2.qlora_code_llama2_13b
    code_llama2.code_llama2_70b
    code_llama2.lora_code_llama2_70b
    code_llama2.qlora_code_llama2_70b

qwen-2
------

Models of size 0.5B, 1.5B, and 7B from the `Qwen2 family <https://huggingface.co/collections/Qwen/qwen2-6659360b33528ced941e557f>`_.

To download the Qwen2 1.5B model, for example:

.. code-block:: bash

    tune download Qwen/Qwen2-1.5B-Instruct --output-dir /tmp/Qwen2-1.5B-Instruct --ignore-patterns None

.. autosummary::
    :toctree: generated/
    :nosignatures:

    qwen2.qwen2
    qwen2.lora_qwen2
    qwen2.qwen2_7b
    qwen2.qwen2_0_5b
    qwen2.qwen2_1_5b
    qwen2.lora_qwen2_7b
    qwen2.lora_qwen2_0_5b
    qwen2.lora_qwen2_1_5b
    qwen2.qwen2_tokenizer

phi-3
-----

Models from the `Phi-3 mini family <https://news.microsoft.com/source/features/ai/the-phi-3-small-language-models-with-big-potential/>`_.

To download the Phi-3 Mini 4k instruct model:

.. code-block:: bash

    tune download microsoft/Phi-3-mini-4k-instruct --output-dir /tmp/Phi-3-mini-4k-instruct --ignore-patterns None --hf-token <HF_TOKEN>

.. autosummary::
    :toctree: generated/
    :nosignatures:

    phi3.phi3
    phi3.lora_phi3
    phi3.phi3_mini
    phi3.lora_phi3_mini
    phi3.qlora_phi3_mini
    phi3.phi3_mini_tokenizer

mistral
-------

All models from `Mistral AI family <https://mistral.ai/technology/#models>`_.

Important: You need to request access on `Hugging Face <https://huggingface.co/mistralai/Mistral-7B-v0.1>`__ to download this model.

To download the Mistral 7B v0.1 model:

.. code-block:: bash

    tune download mistralai/Mistral-7B-v0.1 --output-dir /tmp/Mistral-7B-v0.1 --hf-token <HF_TOKEN>

.. autosummary::
    :toctree: generated/
    :nosignatures:

    mistral.mistral
    mistral.lora_mistral
    mistral.mistral_classifier
    mistral.lora_mistral_classifier
    mistral.mistral_7b
    mistral.lora_mistral_7b
    mistral.qlora_mistral_7b
    mistral.mistral_reward_7b
    mistral.lora_mistral_reward_7b
    mistral.qlora_mistral_reward_7b
    mistral.mistral_tokenizer
    mistral.MistralChatTemplate


gemma
-----

Models of size 2B and 7B from the `Gemma family <https://blog.google/technology/developers/gemma-open-models/>`_.

Important: You need to request access on `Hugging Face <https://huggingface.co/google/gemma-2b>`__ to use this model.

To download the Gemma 2B model (not Gemma2):

.. code-block:: bash

    tune download google/gemma-2b --ignore-patterns "gemma-2b.gguf"  --hf-token <HF_TOKEN>

To download the Gemma 7B model:

.. code-block:: bash

    tune download google/gemma-7b --ignore-patterns "gemma-7b.gguf"  --hf-token <HF_TOKEN>

.. autosummary::
    :toctree: generated/
    :nosignatures:

    gemma.gemma
    gemma.lora_gemma
    gemma.gemma_2b
    gemma.lora_gemma_2b
    gemma.qlora_gemma_2b
    gemma.gemma_7b
    gemma.lora_gemma_7b
    gemma.qlora_gemma_7b
    gemma.gemma_tokenizer


clip
-----

Vision components to support multimodality using `CLIP encoder <https://arxiv.org/abs/2103.00020>`_.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    clip.clip_vision_encoder
    clip.TokenPositionalEmbedding
    clip.TiledTokenPositionalEmbedding
    clip.TilePositionalEmbedding
