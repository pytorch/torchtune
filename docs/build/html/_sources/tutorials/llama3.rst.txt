.. _llama3_label:

========================
Meta Llama3 in torchtune
========================

.. grid:: 2

    .. grid-item-card:: :octicon:`mortar-board;1em;` You will learn how to:

      * Download the Llama3-8B-Instruct weights and tokenizer
      * Fine-tune Llama3-8B-Instruct with LoRA and QLoRA
      * Evaluate your fine-tuned Llama3-8B-Instruct model
      * Generate text with your fine-tuned model
      * Quantize your model to speed up generation

    .. grid-item-card:: :octicon:`list-unordered;1em;` Prerequisites

      * Be familiar with :ref:`torchtune<overview_label>`
      * Make sure to :ref:`install torchtune<install_label>`


Llama3-8B
---------

`Meta Llama 3 <https://llama.meta.com/llama3>`_ is a new family of models released by Meta AI that improves upon the performance of the Llama2 family
of models across a `range of different benchmarks <https://huggingface.co/meta-llama/Meta-Llama-3-8B#base-pretrained-models>`_.
Currently there are two different sizes of Meta Llama 3: 8B and 70B. In this tutorial we will focus on the 8B size model.
There are a few main changes between Llama2-7B and Llama3-8B models:

- Llama3-8B uses `grouped-query attention <https://arxiv.org/abs/2305.13245>`_ instead of the standard multi-head attention from Llama2-7B
- Llama3-8B has a larger vocab size (128,256 instead of 32,000 from Llama2 models)
- Llama3-8B uses a different tokenizer than Llama2 models (`tiktoken <https://github.com/openai/tiktoken>`_ instead of `sentencepiece <https://github.com/google/sentencepiece>`_)
- Llama3-8B uses a larger intermediate dimension in its MLP layers than Llama2-7B
- Llama3-8B uses a higher base value to calculate theta in its `rotary positional embeddings <https://arxiv.org/abs/2104.09864>`_

|

Getting access to Llama3-8B-Instruct
------------------------------------

For this tutorial, we will be using the instruction-tuned version of Llama3-8B. First, let's download the model from Hugging Face. You will need to follow the instructions
on the `official Meta page <https://github.com/meta-llama/llama3/blob/main/README.md>`_ to gain access to the model.
Next, make sure you grab your Hugging Face token from `here <https://huggingface.co/settings/tokens>`_.


.. code-block:: bash

    tune download meta-llama/Meta-Llama-3-8B-Instruct \
        --output-dir <checkpoint_dir> \
        --hf-token <ACCESS TOKEN>

|

Fine-tuning Llama3-8B-Instruct in torchtune
-------------------------------------------

torchtune provides `LoRA <https://arxiv.org/abs/2106.09685>`_, `QLoRA <https://arxiv.org/abs/2305.14314>`_, and full fine-tuning
recipes for fine-tuning Llama3-8B on one or more GPUs. For more on LoRA in torchtune, see our :ref:`LoRA Tutorial <lora_finetune_label>`.
For more on QLoRA in torchtune, see our :ref:`QLoRA Tutorial <qlora_finetune_label>`.

Let's take a look at how we can fine-tune Llama3-8B-Instruct with LoRA on a single device using torchtune. In this example, we will fine-tune
for one epoch on a common instruct dataset for illustrative purposes. The basic command for a single-device LoRA fine-tune is

.. code-block:: bash

    tune run lora_finetune_single_device --config llama3/8B_lora_single_device

.. note::
    To see a full list of recipes and their corresponding configs, simply run ``tune ls`` from the command line.

We can also add :ref:`command-line overrides <cli_override>` as needed, e.g.

.. code-block:: bash

    tune run lora_finetune_single_device --config llama3/8B_lora_single_device \
        checkpointer.checkpoint_dir=<checkpoint_dir> \
        tokenizer.path=<checkpoint_dir>/tokenizer.model \
        checkpointer.output_dir=<checkpoint_dir>

This will load the Llama3-8B-Instruct checkpoint and tokenizer from ``<checkpoint_dir>`` used in the :ref:`tune download <tune_download_label>` command above,
then save a final checkpoint in the same directory following the original format. For more details on the
checkpoint formats supported in torchtune, see our :ref:`checkpointing deep-dive <understand_checkpointer>`.

.. note::
    To see the full set of configurable parameters for this (and other) configs we can use :ref:`tune cp <tune_cp_cli_label>` to copy (and modify)
    the default config. :ref:`tune cp <tune_cp_cli_label>` can be used with recipe scripts too, in case you want to make more custom changes
    that cannot be achieved by directly modifying existing configurable parameters. For more on :ref:`tune cp <tune_cp_cli_label>` see the section on
    :ref:`modifying configs <tune_cp_label>` in our ":ref:`finetune_llama_label`" tutorial.

Once training is complete, the model checkpoints will be saved and their locations will be logged. For
LoRA fine-tuning, the final checkpoint will contain the merged weights, and a copy of just the (much smaller) LoRA weights
will be saved separately.

In our experiments, we observed a peak memory usage of 18.5 GB. The default config can be trained on a consumer GPU with 24 GB VRAM.

If you have multiple GPUs available, you can run the distributed version of the recipe.
torchtune makes use of the `FSDP <https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html>`_ APIs from PyTorch Distributed
to shard the model, optimizer states, and gradients. This should enable you to increase your batch size, resulting in faster overall training.
For example, on two devices:

.. code-block:: bash

    tune run --nproc_per_node 2 lora_finetune_distributed --config llama3/8B_lora

Finally, if we want to use even less memory, we can leverage torchtune's QLoRA recipe via:

.. TODO (SalmanMohammadi) ref qlora recipe page

.. code-block:: bash

    tune run lora_finetune_single_device --config llama3/8B_qlora_single_device

Since our default configs enable full bfloat16 training, all of the above commands can be run with
devices having at least 24 GB of VRAM, and in fact the QLoRA recipe should have peak allocated memory
below 10 GB. You can also experiment with different configurations of LoRA and QLoRA, or even run a full fine-tune.
Try it out!

|

Evaluating fine-tuned Llama3-8B models with EleutherAI's Eval Harness
---------------------------------------------------------------------

Now that we've fine-tuned our model, what's next? Let's take our LoRA-finetuned model from the
preceding section and look at a couple different ways we can evaluate its performance on the tasks we care about.

First, torchtune provides an integration with
`EleutherAI's evaluation harness <https://github.com/EleutherAI/lm-evaluation-harness>`_
for model evaluation on common benchmark tasks.

.. note::
    Make sure you've first installed the evaluation harness via :code:`pip install "lm_eval==0.4.*"`.

For this tutorial we'll use the `truthfulqa_mc2 <https://github.com/sylinrl/TruthfulQA>`_ task from the harness.
This task measures a model's propensity to be truthful when answering questions and
measures the model's zero-shot accuracy on a question followed by one or more true
responses and one or more false responses. First, let's copy the config so we can point the YAML
file to our fine-tuned checkpoint files.

.. code-block:: bash

    tune cp eleuther_evaluation ./custom_eval_config.yaml

Next, we modify ``custom_eval_config.yaml`` to include the fine-tuned checkpoints.

.. code-block:: yaml

    model:
      _component_: torchtune.models.llama3.llama3_8b

    checkpointer:
      _component_: torchtune.training.FullModelMetaCheckpointer

      # directory with the checkpoint files
      # this should match the output_dir specified during
      # fine-tuning
      checkpoint_dir: <checkpoint_dir>

      # checkpoint files for the fine-tuned model. These will be logged
      # at the end of your fine-tune
      checkpoint_files: [
        meta_model_0.pt
      ]

      output_dir: <checkpoint_dir>
      model_type: LLAMA3

    # Make sure to update the tokenizer path to the right
    # checkpoint directory as well
    tokenizer:
      _component_: torchtune.models.llama3.llama3_tokenizer
      path: <checkpoint_dir>/tokenizer.model

Finally, we can run evaluation using our modified config.

.. code-block:: bash

    tune run eleuther_eval --config ./custom_eval_config.yaml

Try it for yourself and see what accuracy your model gets!

|

Generating text with our fine-tuned Llama3 model
------------------------------------------------

.. TODO (SalmanMohammadi) ref generate recipe page

Next, let's look at one other way we can evaluate our model: generating text! torchtune provides a
`recipe for generation <https://github.com/pytorch/torchtune/blob/main/recipes/generate.py>`_ as well.

Similar to what we did, let's copy and modify the default generation config.

.. code-block:: bash

    tune cp generation ./custom_generation_config.yaml

Now we modify ``custom_generation_config.yaml`` to point to our checkpoint and tokenizer.

.. code-block:: yaml

    model:
      _component_: torchtune.models.llama3.llama3_8b

    checkpointer:
      _component_: torchtune.training.FullModelMetaCheckpointer

      # directory with the checkpoint files
      # this should match the output_dir specified during
      # fine-tuning
      checkpoint_dir: <checkpoint_dir>

      # checkpoint files for the fine-tuned model. These will be logged
      # at the end of your fine-tune
      checkpoint_files: [
        meta_model_0.pt
      ]

      output_dir: <checkpoint_dir>
      model_type: LLAMA3

    # Make sure to update the tokenizer path to the right
    # checkpoint directory as well
    tokenizer:
      _component_: torchtune.models.llama3.llama3_tokenizer
      path: <checkpoint_dir>/tokenizer.model

Running generation with our LoRA-finetuned model, we see the following output:

.. code-block:: bash

    tune run generate --config ./custom_generation_config.yaml \
    prompt="Hello, my name is"

    [generate.py:122] Hello, my name is Sarah and I am a busy working mum of two young children, living in the North East of England.
    ...
    [generate.py:135] Time for inference: 10.88 sec total, 18.94 tokens/sec
    [generate.py:138] Bandwidth achieved: 346.09 GB/s
    [generate.py:139] Memory used: 18.31 GB

Faster generation via quantization
----------------------------------

We rely on `torchao <https://github.com/pytorch-labs/ao>`_ for `post-training quantization <https://github.com/pytorch/ao/tree/main/torchao/quantization#quantization>`_.
To quantize the fine-tuned model after installing torchao we can run the following command::

  # we also support `int8_weight_only()` and `int8_dynamic_activation_int8_weight()`, see
  # https://github.com/pytorch/ao/tree/main/torchao/quantization#other-available-quantization-techniques
  # for a full list of techniques that we support
  from torchao.quantization.quant_api import quantize_, int4_weight_only
  quantize_(model, int4_weight_only())

After quantization, we rely on torch.compile for speedups. For more details, please see `this example usage <https://github.com/pytorch/ao/blob/main/torchao/quantization/README.md#quantization-flow-example>`_.

torchao also provides `this table <https://github.com/pytorch/ao#inference>`_ listing performance and accuracy results for ``llama2`` and ``llama3``.

For Llama models, you can run generation directly in torchao on the quantized model using their ``generate.py`` script as
discussed in `this readme <https://github.com/pytorch/ao/tree/main/torchao/_models/llama>`_. This way you can compare your own results
to those in the previously-linked table.


This is just the beginning of what you can do with Meta Llama3 using torchtune and the broader ecosystem.
We look forward to seeing what you build!
