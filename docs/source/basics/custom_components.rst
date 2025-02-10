.. _custom_components_label:

=============================
Custom Components and Recipes
=============================

torchtune lets you launch fine-tuning jobs directly from the command-line using both built-in and custom components,
such as datasets, models, recipes, and configs. This is done with the ``tune run`` command (see :ref:`cli_label`),
which can also be used from your project folder.

Setting up your torchtune project
---------------------------------
First, ensure that you have torchtune installed - see :ref:`install_label`. This will install the ``tune`` command
in your environment, so you can launch ``tune run`` from any directory. Let's create a new project directory and ensure
we can launch a built-in library recipe with a library config from that folder.

.. code-block:: bash

    mkdir ~/my_project
    cd ~/my_project
    # This downloads the Llama 3.2 1B Instruct model
    tune download meta-llama/Llama-3.2-1B-Instruct --output-dir /tmp/Llama-3.2-1B-Instruct --ignore-patterns "original/consolidated.00.pth"
    # This launches a lora finetuning run with the default single device config
    tune run lora_finetune_single_device --config llama3_2/1B_lora_single_device

Launching custom configs
------------------------
Often, you'll want to start with one of our default configs for a particular model and adjust a few training hyperparameters.
You can use the ``tune cp`` command to create a copy of a default config in your project directory so you can make modifications.

.. code-block:: bash

    # Show all the default model configs for each recipe
    tune ls
    # This makes a copy of a Qwen2 full finetune config
    tune cp qwen2/0.5B_full_single_device ~/my_project/config/qwen_config.yaml

Now, you can make modifications to the config directly in your project folder and launch the custom config. Make sure you are using
the correct recipe associated with the config and that you've downloaded the model. Even if you didn't start with copying a library
recipe, you can launch a completely custom config using the same command. Note that for custom configs, you must specify the file extension.

.. code-block:: bash

    mkdir ~/my_project/config
    tune run full_finetune_single_device --config ~/my_project/config/qwen_config.yaml
    # Or launch directly from the project directory with a relative path
    tune run full_finetune_single_device --config config/qwen_config.yaml

For a more detailed discussion on downloading models and modifying library configs, see :ref:`finetune_llama_label`.

Launching custom recipes
------------------------
torchtune's built-in recipes provide starting points for your fine-tuning workflows, but you can write your own training loop
with customized logic for your use case and launch training with ``tune run``. Similar to modifying library configs, you can
also copy one of our recipes as a starting point and modify, or write one completely from scratch. Note that for launching
custom recipes, you must specify the file extension.

.. code-block:: bash

    mkdir ~/my_project/recipes
    # Show all the default recipes
    tune ls
    # This makes a copy of the full finetune single device recipe locally
    tune cp full_finetune_single_device ~/my_project/recipes/single_device.py
    # Launch custom recipe with custom config from project directory
    tune run recipes/single_device.py --config config/qwen_config.yaml

If you are writing a new recipe from scratch, we recommend following the Python convention of defining a ``main()`` function
in your script and decorating it with the :func:`~torchtune.config.parse` decorator. This will enable you to launch the recipe
with ``tune run`` and pass in a yaml file for the ``--config`` argument.

.. code-block:: python

    from torchtune import config
    from omegaconf import DictConfig

    @config.parse
    def main(cfg: DictConfig):
        # Add all your recipe logic here, access config fields as attributes

    if __name__ == "__main__":
        # Config will be parsed from CLI, don't need to pass in here
        main()

Launching with custom components
--------------------------------
torchtune supports full experimentation with custom models, datasets, optimizers, or any fine-tuning component. You can define
these locally in your repo and use them in your recipes and configs that you can launch with ``tune run``.

We recommend following the "builder" pattern when making your components. This means creating "builder" functions that set up
the classes you need with a few high level parameters that can be modified easily from the config. For example, we can define custom
model and dataset builders in our project directory:

.. code-block:: python

    #
    # In models/custom_decoder.py
    #
    class CustomTransformerDecoder(nn.Module):
        # A custom architecture not present in torchtune

    # Builder function for the custom model
    def custom_model(num_layers: int, classification_head: bool = False):
        # Any setup for defining the class
        ...
        # Return the module you want to train
        return CustomTransformerDecoder(...)

This allows us to expose our custom model in a config friendly manner - rather than having to define every argument needed to
construct our custom model in our config, we only expose the arguments which we care about modifying. This is how we implement
our models in torchtune - see :func:`~torchtune.models.llama3_2_vision.llama3_2_vision_11b` as an example.

.. code-block:: python

    #
    # In datasets/custom_dataset.py
    #
    from torchtune.datasets import SFTDataset, PackedDataset
    from torchtune.data import InputOutputToMessages
    from torchtune.modules.transforms.tokenizers import ModelTokenizer

    # Example builder function for a custom code instruct dataset not in torchtune, but using
    # different dataset building blocks from torchtune
    def tiny_codes(tokenizer: ModelTokenizer, packed: bool = True):
        """
        Python subset of nampdn-ai/tiny-codes. Instruct and code response pairs.
        """
        ds = SFTDataset(
            model_transform=tokenizer,
            source="nampdn-ai/tiny-codes",
            message_transform=InputOutputToMessages(
                column_map={"input": "prompt", "output": "response"},
            ),
            filter_fn=lambda x: x["language"] == "python",
            split="train",
        )
        if packed:
            return PackedDataset(ds, max_seq_len=tokenizer.max_seq_len, split_across_pack=False)
        else:
            return ds

.. note::

    If you are using a default torchtune recipe with a custom dataset, you must define the first
    positional argument to be the tokenizer or model transform. These are automatically passed into
    dataset during instantiation and are defined separately in the config, not under the dataset field.

You can define the custom model and custom dataset in the config using the relative import path from where
you are launching with ``tune run``. It is best to define the path relative to your project root directory
and launch from there.

.. code-block:: yaml

    # In YAML file config/custom_finetune.yaml
    model:
      _component_: models.custom_decoder.custom_model
      num_layers: 32
      # this is an optional param, so you can also omit this from the config
      classification_head: False

    dataset:
      _component_: datasets.custom_dataset.tiny_codes
      # we don't need to define a tokenizer here as it's automatically passed in
      packed: True

.. code-block:: bash

    cd ~/my_project/
    tune run recipes/single_device.py --config config/custom_finetune.yaml

If your custom components are not being found or imported correctly, you can try to launch with ``tune run`` after
modifying the ``PYTHONPATH`` to ensure the files in your project directory are importable.

.. code-block:: bash

    PYTHONPATH=${pwd}:PYTHONPATH tune run recipes/single_device.py --config config/custom_finetune.yaml
