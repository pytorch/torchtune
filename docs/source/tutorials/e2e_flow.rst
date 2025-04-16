.. _e2e_flow:

==================================
End-to-End Workflow with torchtune
==================================

In this tutorial, we'll walk through an end-to-end example of how you can fine-tune,
evaluate, optionally quantize and then run generation with your favorite LLM using
torchtune. We'll also go over how you can use some popular tools and libraries
from the community seamlessly with torchtune.

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


Finetune your model
-------------------

First, let's download a model using the tune CLI. The following command will download the `Llama3.2 3B Instruct <https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/>`_
model from the Hugging Face Hub and save it to the local filesystem. Hugging Face uploaded the original
weights (``consolidated.00.pth``) and the weights compatible with the `from_pretrained() <https://huggingface.co/docs/huggingface_hub/main/en/guides/integrations#frompretrained>`_ API (``*.safetensors``).
We don't need both so we'll ignore the original weights when downloading.

.. code-block:: text

    $ tune download meta-llama/Llama-3.2-3B-Instruct --ignore-patterns "original/consolidated.00.pth"
    Successfully downloaded model repo and wrote to the following locations:
    /tmp/Llama-3.2-3B-Instruct/.cache
    /tmp/Llama-3.2-3B-Instruct/.gitattributes
    /tmp/Llama-3.2-3B-Instruct/LICENSE.txt
    /tmp/Llama-3.2-3B-Instruct/README.md
    /tmp/Llama-3.2-3B-Instruct/USE_POLICY.md
    /tmp/Llama-3.2-3B-Instruct/config.json
    /tmp/Llama-3.2-3B-Instruct/generation_config.json
    /tmp/Llama-3.2-3B-Instruct/model-00001-of-00002.safetensors
    ...

.. note::

    For a list of all other models you can finetune out-of-the-box with torchtune, check out
    our :ref:`models page<models>`.

For this tutorial, we'll fine-tune the model using LoRA. LoRA is a parameter efficient fine-tuning
technique which is especially helpful when you don't have a lot of GPU memory to play with. LoRA
freezes the base LLM and adds a very small percentage of learnable parameters. This helps keep
memory associated with gradients and optimizer state low. Using torchtune, you should be able to
fine-tune a Llama-3.2-3B-Instruct model with LoRA in less than 16GB of GPU memory using bfloat16 on a
RTX 3090/4090. For more information on how to use LoRA, take a look at our
:ref:`LoRA Tutorial <lora_finetune_label>`.

Let's look for the right config for this use case by using the tune CLI.

.. code-block:: text

    $ tune ls
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


We'll fine-tune using our
:ref:`single device LoRA recipe <lora_finetune_recipe_label>`
and use the standard settings from the
`default config <https://github.com/pytorch/torchtune/blob/main/recipes/configs/llama3_2/3B_lora_single_device.yaml>`_.

This will fine-tune our model using a ``batch_size=4`` and ``dtype=bfloat16``. With these settings the model
should have a peak memory usage of ~16GB and total training time of around 2-3 hours for each epoch.

.. code-block:: text

    $ tune run lora_finetune_single_device --config llama3_2/3B_lora_single_device
    Setting manual seed to local seed 3977464327. Local seed is seed + rank = 3977464327 + 0
    Hint: enable_activation_checkpointing is True, but enable_activation_offloading isn't. Enabling activation offloading should reduce memory further.
    Writing logs to /tmp/torchtune/llama3_2_3B/lora_single_device/logs/log_1734708879.txt
    Model is initialized with precision torch.bfloat16.
    Memory stats after model init:
            GPU peak memory allocation: 6.21 GiB
            GPU peak memory reserved: 6.27 GiB
            GPU peak memory active: 6.21 GiB
    Tokenizer is initialized from file.
    Optimizer and loss are initialized.
    Loss is initialized.
    Dataset and Sampler are initialized.
    Learning rate scheduler is initialized.
    Profiling disabled.
    Profiler config after instantiation: {'enabled': False}
    1|3|Loss: 1.943998098373413:   0%|                    | 3/1617 [00:21<3:04:47,  6.87s/it]

Congrats on training your model! Let's take a look at the artifacts produced by torchtune. A simple way of doing this is by running :code:`tree -a path/to/outputdir`, which should show something like the tree below.
There are 3 types of folders:

1) **recipe_state**: Holds recipe_state.pt with the information necessary to restart the last intermediate epoch. For more information, please check our deep-dive :ref:`Checkpointing in torchtune <understand_checkpointer>`.;
2) **logs**: Contains all the logging output from your training run: loss, memory, exceptions, etc.
3) **epoch_{}**: Contains your trained model weights plus model metadata. If running inference or pushing to a model hub, you should use this folder directly.


.. code-block:: text

    $ tree -a /tmp/torchtune/llama3_2_3B/lora_single_device
    /tmp/torchtune/llama3_2_3B/lora_single_device
    ├── epoch_0
    │   ├── adapter_config.json
    │   ├── adapter_model.pt
    │   ├── adapter_model.safetensors
    │   ├── config.json
    │   ├── model-00001-of-00002.safetensors
    │   ├── model-00002-of-00002.safetensors
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
    │   ...
    ├── logs
    │   └── log_1734652101.txt
    └── recipe_state
        └── recipe_state.pt

Let's understand the files:

- ``adapter_model.safetensors`` and ``adapter_model.pt`` are your LoRA trained adapter weights. We save a duplicated .pt version of it to facilitate resuming from checkpoint.
- ``model-{}-of-{}.safetensors`` are your trained full model weights (not adapters). When LoRA finetuning, these are only present if we set ``save_adapter_weights_only=False``. In that case, we merge the base model with trained adapters, making inference easier.
- ``adapter_config.json`` is used by Huggingface PEFT when loading an adapter (more on that later);
- ``model.safetensors.index.json`` is used by Hugging Face ``from_pretrained()`` when loading the model weights (more on that later)
- All other files were originally in the checkpoint_dir. They are automatically copied during training. Files over 100MiB and ending in .safetensors, .pth, .pt, .bin are ignored, making it lightweight.

Evaluate your model
-------------------

We've fine-tuned a model. But how well does this model really do? Let's determine this through structured evaluation and playing with it.

.. _eval_harness_label:

Run evals using EleutherAI's Eval Harness
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. TODO (SalmanMohammadi) ref eval recipe docs

torchtune integrates with
`EleutherAI's evaluation harness <https://github.com/EleutherAI/lm-evaluation-harness>`_.
An example of this is available through the
`eleuther_eval <https://github.com/pytorch/torchtune/blob/main/recipes/eleuther_eval.py>`_ recipe. In this tutorial, we're going to directly use this recipe by
modifying its associated config `eleuther_evaluation.yaml <https://github.com/pytorch/torchtune/blob/main/recipes/configs/eleuther_evaluation.yaml>`_.

.. note::
    For this section of the tutorial, you should first run :code:`pip install lm_eval>=0.4.5`
    to install the EleutherAI evaluation harness.

Since we plan to update all of the checkpoint files to point to our fine-tuned checkpoints,
let's first copy over the config to our local working directory so we can make changes.

.. code-block:: bash

    $ tune cp eleuther_evaluation ./custom_eval_config.yaml
    Copied file to custom_eval_config.yaml

Notice that we are using the merged weights, and not the LoRA adapters.

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
            model-00001-of-00002.safetensors,
            model-00002-of-00002.safetensors,
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
responses and one or more false responses.

.. code-block:: text

    $ tune run eleuther_eval --config ./custom_eval_config.yaml
    [evaluator.py:324] Running loglikelihood requests
    ...

Generate some output
~~~~~~~~~~~~~~~~~~~~

We've run some evaluations and the model seems to be doing well. But does it really
generate meaningful text for the prompts you care about? Let's find out!

For this, we'll use the
`generate recipe <https://github.com/pytorch/torchtune/blob/main/recipes/generate.py>`_
and the associated
`config <https://github.com/pytorch/torchtune/blob/main/recipes/configs/generation.yaml>`_.

Let's first copy over the config to our local working directory so we can make changes.

.. code-block:: text

    $ tune cp generation ./custom_generation_config.yaml
    Copied file to custom_generation_config.yaml
    $ mkdir /tmp/torchtune/llama3_2_3B/lora_single_device/out

Let's modify ``custom_generation_config.yaml`` to include the following changes. Again, you only need
 to replace two fields: ``output_dir`` and ``checkpoint_files``

.. code-block:: yaml

    checkpoint_dir: /tmp/torchtune/llama3_2_3B/lora_single_device/epoch_0
    output_dir: /tmp/torchtune/llama3_2_3B/lora_single_device/out

    # Tokenizer
    tokenizer:
        _component_: torchtune.models.llama3.llama3_tokenizer
        path: ${checkpoint_dir}/original/tokenizer.model
        prompt_template: null

    model:
        # Notice that we don't pass the lora model. We are using the merged weights,
        _component_: torchtune.models.llama3_2.llama3_2_3b

    checkpointer:
        _component_: torchtune.training.FullModelHFCheckpointer
        checkpoint_dir: ${checkpoint_dir}
        checkpoint_files: [
            model-00001-of-00002.safetensors,
            model-00002-of-00002.safetensors,
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

.. code-block:: text

    $ tune run generate --config ./custom_generation_config.yaml prompt.user="Tell me a joke. "
    Tell me a joke. Here's a joke for you:

    What do you call a fake noodle?

    An impasta!

Introduce some quantization
~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

.. _use_model_in_wild:

Use your model in the wild
--------------------------

Let's say we're happy with how our model is performing at this point - we want to do something with it! Productionize it for serving, publish on the Hugging Face Hub, etc.
Since we handle checkpoint conversion, you can directly work with standard formats.

Use with Hugging Face ``from_pretrained()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Case 1: Hugging Face using base model + trained adapters**

Here we load the base model from Hugging Face model hub. Then we load the adapters on top of it using `PeftModel <https://huggingface.co/docs/peft/v0.6.1/en/package_reference/peft_model>`_.
It will look for the files ``adapter_model.safetensors`` for the weights and ``adapter_config.json`` for where to insert them.

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

**Case 2: Hugging Face using merged weights**

In this case, Hugging Face will check in ``model.safetensors.index.json`` for which files it should load.

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

Use with vLLM
~~~~~~~~~~~~~

`vLLM <https://docs.vllm.ai/en/latest/>`_ is a fast and easy-to-use library for LLM inference and serving. They include a lot of awesome features like
state-of-the-art serving throughput, continuous batching of incoming requests, quantization, and speculative decoding.

The library will load any .safetensors file. Since we already merged the full model weights and adapter weights, we can safely delete the
adapter weights (or move them) so that vLLM doesn't get confused by those files.

.. code-block:: python

    rm /tmp/torchtune/llama3_2_3B/lora_single_device/base_model/adapter_model.safetensors

Now we can run the following script:

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

Upload your model to the Hugging Face Hub
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
