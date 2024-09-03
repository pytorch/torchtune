.. _e2e_flow:

==================================
End-to-End Workflow with torchtune
==================================

In this tutorial, we'll walk through an end-to-end example of how you can fine-tune,
evaluate, optionally quantize and then run generation with your favorite LLM using
torchtune. We'll also go over how you can use some popular tools and libraries
from the community seemlessly with torchtune.

.. grid:: 2

    .. grid-item-card:: :octicon:`mortar-board;1em;` What this tutorial will cover:

      * Different type of recipes available in torchtune beyond fine-tuning
      * End-to-end example connecting all of these recipes
      * Different tools and libraries you can use with torchtune

    .. grid-item-card:: :octicon:`list-unordered;1em;` Prerequisites

      * Be familiar with the :ref:`overview of torchtune<overview_label>`
      * Make sure to :ref:`install torchtune<install_label>`
      * Concepts such as :ref:`configs <config_tutorial_label>` and
        :ref:`checkpoints <understand_checkpointer>`


Overview
--------

Fine-tuning an LLM is usually only one step in a larger workflow. An example workflow that you
might have can look something like this:

- Download a popular model from `HF Hub <https://huggingface.co/docs/hub/en/index>`_
- Fine-tune the model using a relevant fine-tuning technique. The exact technique used
  will depend on factors such as the model, amount and nature of training data, your hardware
  setup and the end task for which the model will be used
- Evaluate the model on some benchmarks to validate model quality
- Run some generations to make sure the model output looks reasonable
- Quantize the model for efficient inference
- [Optional] Export the model for specific environments such as inference on a mobile phone

In this tutorial, we'll cover how you can use torchtune for all of the above, leveraging
integrations with popular tools and libraries from the ecosystem.

We'll use the Llama2 7B model for this tutorial. You can find a complete set of models supported
by torchtune `here <https://github.com/pytorch/torchtune/blob/main/README.md#introduction>`_.

|

Download Llama2 7B
------------------

In this tutorial, we'll use the Hugging Face model weights for the Llama2 7B mode.
For more information on checkpoint formats and how these are handled in torchtune, take a look at
this tutorial on :ref:`checkpoints <understand_checkpointer>`.

To download the HF format Llama2 7B model, we'll use the tune CLI.

.. code-block:: bash

  tune download \
  meta-llama/Llama-2-7b-hf \
  --output-dir <checkpoint_dir> \
  --hf-token <ACCESS TOKEN>

Make a note of ``<checkpoint_dir>``, we'll use this many times in this tutorial.

|

Finetune the model using LoRA
-----------------------------

For this tutorial, we'll fine-tune the model using LoRA. LoRA is a parameter efficient fine-tuning
technique which is especially helpful when you don't have a lot of GPU memory to play with. LoRA
freezes the base LLM and adds a very small percentage of learnable parameters. This helps keep
memory associated with gradients and optimizer state low. Using torchtune, you should be able to
fine-tune a Llama2 7B model with LoRA in less than 16GB of GPU memory using bfloat16 on a
RTX 3090/4090. For more information on how to use LoRA, take a look at our
:ref:`LoRA Tutorial <lora_finetune_label>`.

We'll fine-tune using our
`single device LoRA recipe <https://github.com/pytorch/torchtune/blob/main/recipes/lora_finetune_single_device.py>`_
and use the standard settings from the
`default config <https://github.com/pytorch/torchtune/blob/main/recipes/configs/llama2/7B_lora_single_device.yaml>`_.

This will fine-tune our model using a ``batch_size=2`` and ``dtype=bfloat16``. With these settings the model
should have a peak memory usage of ~16GB and total training time of around two hours for each epoch.
We'll need to make some changes to the config to make sure our recipe can access the
right checkpoints.

Let's look for the right config for this use case by using the tune CLI.

.. code-block:: bash

    tune ls

    RECIPE                                   CONFIG
    full_finetune_single_device              llama2/7B_full_low_memory
                                             mistral/7B_full_low_memory
    full_finetune_distributed                llama2/7B_full
                                             llama2/13B_full
                                             mistral/7B_full
    lora_finetune_single_device              llama2/7B_lora_single_device
                                             llama2/7B_qlora_single_device
                                             mistral/7B_lora_single_device
    ...


For this tutorial we'll use the ``llama2/7B_lora_single_device`` config.

The config already points to the HF Checkpointer and the right checkpoint files.
All we need to do is update the checkpoint directory for both the model and the
tokenizer. Let's do this using the overrides in the tune CLI while starting training!


.. code-block:: bash

    tune run lora_finetune_single_device \
    --config llama2/7B_lora_single_device \
    checkpointer.checkpoint_dir=<checkpoint_dir> \
    tokenizer.path=<checkpoint_dir>/tokenizer.model \
    checkpointer.output_dir=<checkpoint_dir>


Once training is complete, you'll see the following in the logs.

.. code-block:: bash

    [_checkpointer.py:473] Model checkpoint of size 9.98 GB saved to <checkpoint_dir>/hf_model_0001_0.pt

    [_checkpointer.py:473] Model checkpoint of size 3.50 GB saved to <checkpoint_dir>/hf_model_0002_0.pt

    [_checkpointer.py:484] Adapter checkpoint of size 0.01 GB saved to <checkpoint_dir>/adapter_0.pt


The final trained weights are merged with the original model and split across two checkpoint files
similar to the source checkpoints from the HF Hub
(see the :ref:`LoRA Tutorial <lora_finetune_label>` for more details).
In fact the keys will be identical between these checkpoints.
We also have a third checkpoint file which is much smaller in size
and contains the learnt LoRA adapter weights. For this tutorial, we'll only use the model
checkpoints and not the adapter weights.

|

.. _eval_harness_label:

Run Evaluation using EleutherAI's Eval Harness
----------------------------------------------

We've fine-tuned a model. But how well does this model really do? Let's run some Evaluations!

.. TODO (SalmanMohammadi) ref eval recipe docs

torchtune integrates with
`EleutherAI's evaluation harness <https://github.com/EleutherAI/lm-evaluation-harness>`_.
An example of this is available through the
``eleuther_eval`` recipe. In this tutorial, we're going to directly use this recipe by
modifying its associated config ``eleuther_evaluation.yaml``.

.. note::
    For this section of the tutorial, you should first run :code:`pip install lm_eval==0.4.*`
    to install the EleutherAI evaluation harness.

Since we plan to update all of the checkpoint files to point to our fine-tuned checkpoints,
let's first copy over the config to our local working directory so we can make changes. This
will be easier than overriding all of these elements through the CLI.

.. code-block:: bash

    tune cp eleuther_evaluation ./custom_eval_config.yaml \

For this tutorial we'll use the `truthfulqa_mc2 <https://github.com/sylinrl/TruthfulQA>`_ task from the harness.
This task measures a model's propensity to be truthful when answering questions and
measures the model's zero-shot accuracy on a question followed by one or more true
responses and one or more false responses. Let's first run a baseline without fine-tuning.


.. code-block:: bash

    tune run eleuther_eval --config ./custom_eval_config.yaml
    checkpointer.checkpoint_dir=<checkpoint_dir> \
    tokenizer.path=<checkpoint_dir>/tokenizer.model

    [evaluator.py:324] Running loglikelihood requests
    [eleuther_eval.py:195] Eval completed in 121.27 seconds.
    [eleuther_eval.py:197] truthfulqa_mc2: {'acc,none': 0.388...

The model has an accuracy around 38.8%. Let's compare this with the fine-tuned model.


First, we modify ``custom_eval_config.yaml`` to include the fine-tuned checkpoints.

.. code-block:: yaml

    checkpointer:
        _component_: torchtune.training.FullModelHFCheckpointer

        # directory with the checkpoint files
        # this should match the output_dir specified during
        # finetuning
        checkpoint_dir: <checkpoint_dir>

        # checkpoint files for the fine-tuned model. This should
        # match what's shown in the logs above
        checkpoint_files: [
            hf_model_0001_0.pt,
            hf_model_0002_0.pt,
        ]

        output_dir: <checkpoint_dir>
        model_type: LLAMA2

    # Make sure to update the tokenizer path to the right
    # checkpoint directory as well
    tokenizer:
        _component_: torchtune.models.llama2.llama2_tokenizer
        path: <checkpoint_dir>/tokenizer.model


Now, let's run the recipe.

.. code-block:: bash

    tune run eleuther_eval --config ./custom_eval_config.yaml


The results should look something like this.

.. code-block:: bash

    [evaluator.py:324] Running loglikelihood requests
    [eleuther_eval.py:195] Eval completed in 121.27 seconds.
    [eleuther_eval.py:197] truthfulqa_mc2: {'acc,none': 0.489 ...

Our fine-tuned model gets ~48% on this task, which is ~10 points
better than the baseline. Great! Seems like our fine-tuning helped.

|

Generation
-----------

We've run some evaluations and the model seems to be doing well. But does it really
generate meaningful text for the prompts you care about? Let's find out!

For this, we'll use the
`generate recipe <https://github.com/pytorch/torchtune/blob/main/recipes/generate.py>`_
and the associated
`config <https://github.com/pytorch/torchtune/blob/main/recipes/configs/generation.yaml>`_.


Let's first copy over the config to our local working directory so we can make changes.

.. code-block:: bash

    tune cp generation ./custom_generation_config.yaml

Let's modify ``custom_generation_config.yaml`` to include the following changes.

.. code-block:: yaml

    checkpointer:
        _component_: torchtune.training.FullModelHFCheckpointer

        # directory with the checkpoint files
        # this should match the output_dir specified during
        # finetuning
        checkpoint_dir: <checkpoint_dir>

        # checkpoint files for the fine-tuned model. This should
        # match what's shown in the logs above
        checkpoint_files: [
            hf_model_0001_0.pt,
            hf_model_0002_0.pt,
        ]

        output_dir: <checkpoint_dir>
        model_type: LLAMA2

    # Make sure to update the tokenizer path to the right
    # checkpoint directory as well
    tokenizer:
        _component_: torchtune.models.llama2.llama2_tokenizer
        path: <checkpoint_dir>/tokenizer.model


Once the config is updated, let's kick off generation! We'll use the
default settings for sampling with ``top_k=300`` and a
``temperature=0.8``. These parameters control how the probabilities for
sampling are computed. These are standard settings for Llama2 7B and
we recommend inspecting the model with these before playing around with
these parameters.

We'll use a different prompt from the one in the config

.. code-block:: bash

    tune run generate --config ./custom_generation_config.yaml \
    prompt="What are some interesting sites to visit in the Bay Area?"


Once generation is complete, you'll see the following in the logs.


.. code-block:: bash

    [generate.py:92] Exploratorium in San Francisco has made the cover of Time Magazine,
                     and its awesome. And the bridge is pretty cool...

    [generate.py:96] Time for inference: 11.61 sec total, 25.83 tokens/sec
    [generate.py:99] Memory used: 15.72 GB


Indeed, the bridge is pretty cool! Seems like our LLM knows a little something about the
Bay Area!

|

Speeding up Generation using Quantization
-----------------------------------------

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

|

Using torchtune checkpoints with other libraries
------------------------------------------------

As we mentioned above, one of the benefits of handling of the checkpoint
conversion is that you can directly work with standard formats. This helps
with interoperability with other libraries since torchtune doesn't add yet
another format to the mix.

Let's take a look at an example of how this would work with a popular codebase
used for running performant inference with LLMs -
`gpt-fast <https://github.com/pytorch-labs/gpt-fast/tree/main>`_. This section
assumes that you've cloned that repository on your machine.

``gpt-fast`` makes some assumptions about the checkpoint and the availability of
the key-to-file mapping i.e. a file mapping parameter names to the files containing them.
Let's satisfy these assumptions, by creating this mapping
file. Let's assume we'll be using ``<new_dir>/Llama-2-7B-hf`` as the directory
for this. ``gpt-fast`` assumes that the directory with checkpoints has the
same format at the HF repo-id.

.. code-block:: python

    import json
    import torch

    # create the output dictionary
    output_dict = {"weight_map": {}}

    # Load the checkpoints
    sd_1 = torch.load('<checkpoint_dir>/hf_model_0001_0.pt', mmap=True, map_location='cpu')
    sd_2 = torch.load('<checkpoint_dir>/hf_model_0002_0.pt', mmap=True, map_location='cpu')

    # create the weight map
    for key in sd_1.keys():
        output_dict['weight_map'][key] =  "hf_model_0001_0.pt"
    for key in sd_2.keys():
        output_dict['weight_map'][key] =  "hf_model_0002_0.pt"

    with open('<new_dir>/Llama-2-7B-hf/pytorch_model.bin.index.json', 'w') as f:
        json.dump(output_dict, f)


Now that we've created the weight_map, let's copy over our checkpoints.

.. code-block:: bash

    cp  <checkpoint_dir>/hf_model_0001_0.pt  <new_dir>/Llama-2-7B-hf/
    cp  <checkpoint_dir>/hf_model_0002_0.pt  <new_dir>/Llama-2-7B-hf/
    cp  <checkpoint_dir>/tokenizer.model     <new_dir>/Llama-2-7B-hf/

Once the directory structure is setup, let's convert the checkpoints and run inference!

.. code-block:: bash

    cd gpt-fast/

    # convert the checkpoints into a format readable by gpt-fast
    python scripts/convert_hf_checkpoint.py \
    --checkpoint_dir <new_dir>/Llama-2-7B-hf/ \
    --model 7B

    # run inference using the converted model
    python generate.py \
    --compile \
    --checkpoint_path <new_dir>/Llama-2-7B-hf/model.pth \
    --device cuda

The output should look something like this:

.. code-block:: bash

    Hello, my name is Justin. I am a middle school math teacher
    at WS Middle School ...

    Time for inference 5: 1.94 sec total, 103.28 tokens/sec
    Bandwidth achieved: 1391.84 GB/sec


And thats it! Try your own prompt!

Uploading your model to the Hugging Face Hub
--------------------------------------------

Your new model is working great and you want to share it with the world. The easiest way to do this
is utilizing the `huggingface-cli <https://huggingface.co/docs/huggingface_hub/en/guides/cli>`_ command, which works seamlessly with torchtune. Simply point the CLI
to your finetuned model directory like so:

.. code-block:: bash

    huggingface-cli upload <hf-repo-id> <checkpoint-dir>

The command should output a link to your repository on the Hub. If the repository doesn't exist yet, it will
be created automatically:

.. code-block:: text

    https://huggingface.co/<hf-repo-id>/tree/main/.

.. note::

    Before uploading, make sure you are `authenticated with Hugging Face <https://huggingface.co/docs/huggingface_hub/quick-start#authentication>`_ by running ``huggingface-cli login``.

For more details on the ``huggingface-cli upload`` feature check out the `Hugging Face docs <https://huggingface.co/docs/huggingface_hub/en/guides/cli#huggingface-cli-upload>`_.

|

Hopefully this tutorial gave you some insights into how you can use torchtune for
your own workflows. Happy Tuning!
