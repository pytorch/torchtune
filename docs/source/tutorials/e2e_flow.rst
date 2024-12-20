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

We'll use the Llama-3.2-3B-Instruct model for this tutorial. You can find a complete set of models supported
by torchtune `here <https://github.com/pytorch/torchtune/blob/main/README.md#introduction>`_.

|

Download Llama-3.2-3B-Instruct
------------------------------

For more information on checkpoint formats and how these are handled in torchtune, take a look at
this tutorial on :ref:`checkpoints <understand_checkpointer>`.

To download the HF format Llama-3.2-3B-Instruct, we'll use the tune CLI.

.. code-block:: bash

  tune download meta-llama/Llama-3.2-3B-Instruct \
    --output-dir /tmp/Llama-3.2-3B-Instruct \
    --ignore-patterns "original/consolidated.00.pth"

Make a note of ``<checkpoint_dir>``, we'll use this many times in this tutorial.

|

Finetune the model using LoRA
-----------------------------

For this tutorial, we'll fine-tune the model using LoRA. LoRA is a parameter efficient fine-tuning
technique which is especially helpful when you don't have a lot of GPU memory to play with. LoRA
freezes the base LLM and adds a very small percentage of learnable parameters. This helps keep
memory associated with gradients and optimizer state low. Using torchtune, you should be able to
fine-tune a Llama-3.2-3B-Instruct model with LoRA in less than 16GB of GPU memory using bfloat16 on a
RTX 3090/4090. For more information on how to use LoRA, take a look at our
:ref:`LoRA Tutorial <lora_finetune_label>`.

We'll fine-tune using our
`single device LoRA recipe <https://github.com/pytorch/torchtune/blob/main/recipes/lora_finetune_single_device.py>`_
and use the standard settings from the
`default config <https://github.com/pytorch/torchtune/blob/main/recipes/configs/llama3_2/3B_lora_single_device.yaml>`_.

This will fine-tune our model using a ``batch_size=2`` and ``dtype=bfloat16``. With these settings the model
should have a peak memory usage of ~16GB and total training time of around two hours for each epoch.
We'll need to make some changes to the config to make sure our recipe can access the
right checkpoints.

Let's look for the right config for this use case by using the tune CLI.

.. code-block:: bash

    tune ls

    RECIPE                                  CONFIG
    full_finetune_single_device             llama2/7B_full_low_memory
                                            code_llama2/7B_full_low_memory
                                            llama3/8B_full_single_device
                                            llama3_1/8B_full_single_device
                                            llama3_2/1B_full_single_device
                                            llama3_2/3B_full_single_device
                                            mistral/7B_full_low_memory
                                            phi3/mini_full_low_memory
                                            qwen2/7B_full_single_device
                                            ...


    full_finetune_distributed               llama2/7B_full
                                            llama2/13B_full
                                            llama3/8B_full
                                            llama3_1/8B_full
                                            llama3_2/1B_full
                                            llama3_2/3B_full
                                            mistral/7B_full
                                            gemma2/9B_full
                                            gemma2/27B_full
                                            phi3/mini_full
                                            qwen2/7B_full
                                            ...

    lora_finetune_single_device             llama2/7B_lora_single_device
                                            llama2/7B_qlora_single_device
                                            llama3/8B_lora_single_device
    ...


For this tutorial we'll use the ``llama3_2/3B_lora_single_device`` config.

The config already points to the HF Checkpointer and the right checkpoint files.
All we need to do is update the checkpoint directory for both the model and the
tokenizer. Let's do this using the overrides in the tune CLI while starting training!


.. code-block:: bash

    tune run lora_finetune_single_device --config llama3_2/3B_lora_single_device


Preparing your artifacts for inference
--------------------------------------

Congrats for getting this far! You have loaded your weights, trained your model, now it's time to visualize
the outputs. A simple way of doing this is by running `tree -a path/to/outputdir`, which should show something like the tree below.
There are 4 types of folders:

1) **recipe_state**: Holds recipe_state.pt with the information necessary to restart the last intermediate epoch. For more information, please check our deep-dive :ref:`Checkpointing in torchtune <understand_checkpointer>`.;
2) **logs**: Defined in your config in metric_logger;
3) **epoch_{}**: Contains your new trained model weights plus all original files of the model, except the checkpoints, making it easy for you to choose an specific epoch to run inference on or push to a model hub;

.. code-block:: bash

    >>> tree -a /tmp/torchtune/llama3_2_3B/lora_single_device
        /tmp/torchtune/llama3_2_3B/lora_single_device
        ├── epoch_0
        │   ├── adapter_config.json
        │   ├── adapter_model.pt
        │   ├── adapter_model.safetensors
        │   ├── config.json
        │   ├── ft-model-00001-of-00002.safetensors
        │   ├── ft-model-00002-of-00002.safetensors
        │   ├── generation_config.json
        │   ├── LICENSE.txt
        │   ├── model.safetensors.index.json
        │   ├── original
        │   │   ├── orig_params.json
        │   │   ├── params.json
        │   │   └── tokenizer.model
        │   ├── original_repo_id.json
        │   ├── README.md
        │   ├── special_tokens_map.json
        │   ├── tokenizer_config.json
        │   ├── tokenizer.json
        │   └── USE_POLICY.md
        ├── epoch_1
        │   ├── adapter_config.json
        │   ├── adapter_model.pt
        │   ├── adapter_model.safetensors
        │   ├── config.json
        │   ├── ft-model-00001-of-00002.safetensors
        │   ├── ft-model-00002-of-00002.safetensors
        │   ├── generation_config.json
        │   ├── LICENSE.txt
        │   ├── model.safetensors.index.json
        │   ├── original
        │   │   ├── orig_params.json
        │   │   ├── params.json
        │   │   └── tokenizer.model
        │   ├── original_repo_id.json
        │   ├── README.md
        │   ├── special_tokens_map.json
        │   ├── tokenizer_config.json
        │   ├── tokenizer.json
        │   └── USE_POLICY.md
        ├── logs
        │   └── log_1734652101.txt
        └── recipe_state
            └── recipe_state.pt

Let's understand the files:

- `adapter_model.safetensors` and `adapter_model.pt` are your LoRA trained adapter weights. We save a duplicated .pt version of it to facilitate resuming from checkpoint.
- `ft-model-{}-of-{}.safetensors` are your trained full model weights (not adapters). When LoRA finetuning, these are only present if we set ``save_adapter_weights_only=False``. In that case, we merge the merged base model with trained adapters, making inference easier.
- `adapter_config.json` is used by Huggingface PEFT when loading an adapter (more on that later);
- `model.safetensors.index.json` is used by Huggingface .from_pretrained when loading the model weights (more on that later)
- All other files were originally in the checkpoint_dir. They are automatically copied during training. Files over 100MiB and ending on .safetensors, .pth, .pt, .bin are ignored, making it lightweight.

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
let's first copy over the config to our local working directory so we can make changes.

.. code-block:: bash

    tune cp eleuther_evaluation ./custom_eval_config.yaml \

Then, in your config, you only need to replace two fields: ``output_dir`` and ``checkpoint_files``. Notice
that we are using the merged weights, and not the LoRA adapters.

.. code-block:: yaml

    # TODO: update to your desired epoch
    output_dir: /tmp/torchtune/llama3_2_3B/lora_single_device/epoch_0

    # Tokenizer
    tokenizer:
        _component_: torchtune.models.llama3.llama3_tokenizer
        path: ${output_dir}/original/tokenizer.model

    model:
        # Notice that we don't pass the lora model. We are using the merged weights,
        _component_: torchtune.models.llama3_2.llama3_2_3b

    checkpointer:
        _component_: torchtune.training.FullModelHFCheckpointer
        checkpoint_dir: ${output_dir}
        checkpoint_files: [
            ft-model-00001-of-00002.safetensors,
            ft-model-00002-of-00002.safetensors,
        ]
        output_dir: ${output_dir}
        model_type: LLAMA3_2

    ### OTHER PARAMETERS -- NOT RELATED TO THIS CHECKPOINT

    # Environment
    device: cuda
    dtype: bf16
    seed: 1234 # It is not recommended to change this seed, b/c it matches EleutherAI's default seed

    # EleutherAI specific eval args
    tasks: ["truthfulqa_mc2"]
    limit: null
    max_seq_length: 4096
    batch_size: 8
    enable_kv_cache: True

    # Quantization specific args
    quantizer: null

For this tutorial we'll use the `truthfulqa_mc2 <https://github.com/sylinrl/TruthfulQA>`_ task from the harness.

This task measures a model's propensity to be truthful when answering questions and
measures the model's zero-shot accuracy on a question followed by one or more true
responses and one or more false responses


.. code-block:: yaml

    tune run eleuther_eval --config ./custom_eval_config.yaml

    [evaluator.py:324] Running loglikelihood requests

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

Let's modify ``custom_generation_config.yaml`` to include the following changes. Again, you only need
 to replace two fields: ``output_dir`` and ``checkpoint_files``

.. code-block:: yaml

    output_dir: /tmp/torchtune/llama3_2_3B/lora_single_device/epoch_0

    # Tokenizer
    tokenizer:
        _component_: torchtune.models.llama3.llama3_tokenizer
        path: ${output_dir}/original/tokenizer.model
        prompt_template: null

    model:
        # Notice that we don't pass the lora model. We are using the merged weights,
        _component_: torchtune.models.llama3_2.llama3_2_3b

    checkpointer:
        _component_: torchtune.training.FullModelHFCheckpointer
        checkpoint_dir: ${output_dir}
        checkpoint_files: [
            ft-model-00001-of-00002.safetensors,
            ft-model-00002-of-00002.safetensors,
        ]
        output_dir: ${output_dir}
        model_type: LLAMA3_2

    ### OTHER PARAMETERS -- NOT RELATED TO THIS CHECKPOINT

    device: cuda
    dtype: bf16

    seed: 1234

    # Generation arguments; defaults taken from gpt-fast
    prompt:
    system: null
    user: "Tell me a joke. "
    max_new_tokens: 300
    temperature: 0.6 # 0.8 and 0.6 are popular values to try
    top_k: 300

    enable_kv_cache: True

    quantizer: null

Once the config is updated, let's kick off generation! We'll use the
default settings for sampling with ``top_k=300`` and a
``temperature=0.8``. These parameters control how the probabilities for
sampling are computed. We recommend inspecting the model with these before playing around with
these parameters.

.. code-block:: bash

    tune run generate --config ./custom_generation_config.yaml \
    prompt="tell me a joke. "


Once generation is complete, you'll see the following in the logs.


.. code-block::

    Tell me a joke. Here's a joke for you:

    What do you call a fake noodle?

    An impasta!

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

Let's start with huggingface

**Case 1: HF using BASE MODEL + trained adapter**

Here we load the base model from HF model hub. Then we load the adapters on top of it using PeftModel.
It will look for the files adapter_model.safetensors for the weights and adapter_config.json for where to insert them.

.. code-block:: python

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    #TODO: update it to your chosen epoch
    trained_model_path = "/tmp/torchtune/llama3_2_3B/lora_single_device/epoch_0"

    # Define the model and adapter paths
    original_model_name = "meta-llama/Llama-3.2-1B-Instruct"

    model = AutoModelForCausalLM.from_pretrained(original_model_name)

    # huggingface will look for adapter_model.safetensors and adapter_config.json
    peft_model = PeftModel.from_pretrained(model, trained_model_path)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(original_model_name)

    # Function to generate text
    def generate_text(model, tokenizer, prompt, max_length=50):
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=max_length)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    prompt = "tell me a joke: '"
    print("Base model output:", generate_text(peft_model, tokenizer, prompt))

**Case 2: HF using merged full+adapter weights**

In this case, HF will check in model.safetensors.index.json which files it should load.

.. code-block:: python

    from transformers import AutoModelForCausalLM, AutoTokenizer

    #TODO: update it to your chosen epoch
    trained_model_path = "/tmp/torchtune/llama3_2_3B/lora_single_device/epoch_0"

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=trained_model_path,
    )

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(trained_model_path, safetensors=True)


    # Function to generate text
    def generate_text(model, tokenizer, prompt, max_length=50):
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=max_length)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)


    prompt = "Complete the sentence: 'Once upon a time...'"
    print("Base model output:", generate_text(model, tokenizer, prompt))

**Case 3: vLLM using merged full+adapter weights**

It will load any .safetensors file. Since here we mixed both the full model weights and adapter weights, we have to delete the
adapter weights to succesfully load it.

.. code-block:: bash

    rm /tmp/torchtune/llama3_2_3B/lora_single_device/base_model/adapter_model.safetensors

Now we can run the script

.. code-block:: python

    from vllm import LLM, SamplingParams

    def print_outputs(outputs):
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        print("-" * 80)

    #TODO: update it to your chosen epoch
    llm = LLM(
        model="/tmp/torchtune/llama3_2_3B/lora_single_device/epoch_0",
        load_format="safetensors",
        kv_cache_dtype="auto",
    )
    sampling_params = SamplingParams(max_tokens=16, temperature=0.5)

    conversation = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hello! How can I assist you today?"},
        {
            "role": "user",
            "content": "Write an essay about the importance of higher education.",
        },
    ]
    outputs = llm.chat(conversation, sampling_params=sampling_params, use_tqdm=False)
    print_outputs(outputs)

Uploading your model to the Hugging Face Hub
--------------------------------------------

Your new model is working great and you want to share it with the world. The easiest way to do this
is utilizing the `huggingface_hub <https://huggingface.co/docs/huggingface_hub/guides/upload>`_.

.. code-block:: python

    import huggingface_hub
    api = huggingface_hub.HfApi()

    #TODO: update it to your chosen epoch
    trained_model_path = "/tmp/torchtune/llama3_2_3B/lora_single_device/epoch_0"

    username = huggingface_hub.whoami()["name"]
    repo_name = "my-model-trained-with-torchtune"

    # if the repo doesn't exist
    repo_id = huggingface_hub.create_repo(repo_name).repo_id

    # if it already exists
    repo_id = f"{username}/{repo_name}"

    api.upload_folder(
        folder_path=trained_model_path,
        repo_id=repo_id,
        repo_type="model",
        create_pr=False
    )

If you prefer, you can also try the cli version `huggingface-cli upload <https://huggingface.co/docs/huggingface_hub/en/guides/cli#huggingface-cli-upload>`_.

|

Hopefully this tutorial gave you some insights into how you can use torchtune for
your own workflows. Happy Tuning!
