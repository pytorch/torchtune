.. _config_tutorial_label:

=================
All About Configs
=================

This deep-dive will guide you through writing configs for running recipes.

.. grid:: 2

    .. grid-item-card:: :octicon:`mortar-board;1em;` What this deep-dive will cover

      * How to write a YAML config and run a recipe with it
      * How to use :code:`instantiate` and :code:`parse` APIs
      * How to effectively use configs and CLI overrides for running recipes

    .. grid-item-card:: :octicon:`list-unordered;1em;` Prerequisites

      * Be familiar with the :ref:`overview of torchtune<overview_label>`
      * Make sure to :ref:`install torchtune<install_label>`
      * Understand the :ref:`fundamentals of recipes<recipe_deepdive>`


Where do parameters live?
-------------------------

There are two primary entry points for you to configure parameters: **configs** and
**CLI overrides**. Configs are YAML files that define all the
parameters needed to run a recipe within a single location. They are the single
source of truth for reproducing a run. The config parameters can be overridden on the
command-line using :code:`tune` for quick changes and experimentation without
modifying the config.


Writing configs
---------------
Configs serve as the primary entry point for running recipes in torchtune. They are
expected to be YAML files and they simply list out values for parameters you want to define
for a particular run.

.. code-block:: yaml

    seed: null
    shuffle: True
    device: cuda
    dtype: fp32
    enable_fsdp: True
    ...

Configuring components using :func:`instantiate<torchtune.config.instantiate>`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Many fields will require specifying torchtune objects with associated keyword
arguments as parameters. Models, datasets, optimizers, and loss functions are
common examples of this. You can easily do this using the :code:`_component_`
subfield. In :code:`_component_`, you need to specify the dotpath of the object
you wish to instantiate in the recipe. The dotpath is the exact path you would use
to import the object normally in a Python file. For example, to specify the
:class:`~torchtune.datasets.alpaca_dataset` in your config with custom
arguments:

.. code-block:: yaml

    dataset:
      _component_: torchtune.datasets.alpaca_dataset
      train_on_input: False

Here, we are changing the default value for :code:`train_on_input` from :code:`True`
to :code:`False`.

Once you've specified the :code:`_component_` in your config, you can create an
instance of the specified object in your recipe's setup like so:

.. code-block:: python

    from torchtune import config

    # Access the dataset field and create the object instance
    dataset = config.instantiate(cfg.dataset)

This will automatically use any keyword arguments specified in the fields under
:code:`dataset`.

As written, the preceding example will actually throw an error. If you look at the method for :class:`~torchtune.datasets.alpaca_dataset`,
you'll notice that we're missing a required positional argument, the tokenizer.
Since this is another configurable torchtune object, let's understand how to handle
this by taking a look at the :func:`~torchtune.config.instantiate` API.

.. code-block:: python

    def instantiate(
        config: DictConfig,
        *args: Tuple[Any, ...],
        **kwargs: Dict[str, Any],
    )

:func:`~torchtune.config.instantiate` also accepts positional arguments
and keyword arguments and automatically uses that with the config when creating
the object. This means we can not only pass in the tokenizer, but also add additional
keyword arguments not specified in the config if we'd like:

.. code-block:: yaml

    # Tokenizer is needed for the dataset, configure it first
    tokenizer:
      _component_: torchtune.models.llama2.llama2_tokenizer
      path: /tmp/tokenizer.model

    dataset:
      _component_: torchtune.datasets.alpaca_dataset

.. code-block:: python

    # Note the API of the tokenizer we specified - we need to pass in a path
    def llama2_tokenizer(path: str) -> Llama2Tokenizer:

    # Note the API of the dataset we specified - we need to pass in a model tokenizer
    # and any optional keyword arguments
    def alpaca_dataset(
        tokenizer: ModelTokenizer,
        train_on_input: bool = True,
        max_seq_len: int = 512,
    ) -> SFTDataset:

    from torchtune import config

    # Since we've already specified the path in the config, we don't need to pass
    # it in
    tokenizer = config.instantiate(cfg.tokenizer)
    # We pass in the instantiated tokenizer as the first required argument, then
    # we change an optional keyword argument
    dataset = config.instantiate(
        cfg.dataset,
        tokenizer,
        train_on_input=False,
    )

Note that additional keyword arguments will overwrite any duplicated keys in the
config.

Referencing other config fields with interpolations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Sometimes you need to use the same value more than once for multiple fields. You
can use *interpolations* to reference another field, and :func:`~torchtune.config.instantiate`
will automatically resolve it for you.

.. code-block:: yaml

    output_dir: /tmp/alpaca-llama2-finetune
    metric_logger:
      _component_: torchtune.training.metric_logging.DiskLogger
      log_dir: ${output_dir}

Validating your config
^^^^^^^^^^^^^^^^^^^^^^
We provide a convenient CLI utility, :ref:`tune validate<validate_cli_label>`, to quickly verify that
your config is well-formed and all components can be instantiated properly. You
can also pass in overrides if you want to test out the exact commands you will run
your experiments with. If any parameters are not well-formed, :ref:`tune validate<validate_cli_label>`
will list out all the locations where an error was found.

.. code-block:: bash

  tune cp llama2/7B_lora_single_device ./my_config.yaml
  tune validate ./my_config.yaml

Best practices for writing configs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Let's discuss some guidelines for writing configs to get the most out of them.

Airtight configs
""""""""""""""""
While it may be tempting to put as much as you can in the config to give you
maximum flexibility in switching parameters for your experiments, we encourage
you to only include fields in the config that will be used or instantiated in the
recipe. This ensures full clarity on the options a recipe was run with and will
make it significantly easier to debug.

.. code-block:: yaml

    # dont do this
    alpaca_dataset:
      _component_: torchtune.datasets.alpaca_dataset
    slimorca_dataset:
      ...

    # do this
    dataset:
      # change this in config or override when needed
      _component_: torchtune.datasets.alpaca_dataset

Use public APIs only
""""""""""""""""""""
If a component you wish to specify in a config is located in a private file, use
the public dotpath in your config. These components are typically exposed in their
parent module's :code:`__init__.py` file. This way, you can guarantee the stability
of the API you are using in your config. There should be no underscores in your
component dotpath.

.. code-block:: yaml

    # don't do this
    dataset:
      _component_: torchtune.datasets._alpaca.alpaca_dataset

    # do this
    dataset:
      _component_: torchtune.datasets.alpaca_dataset

.. _cli_override:

Command-line overrides
----------------------
Configs are the primary location to collect all your parameters to run a recipe,
but sometimes you may want to quickly try different values without having to update
the config itself. To enable quick experimentation, you can specify override values
to parameters in your config via the :code:`tune` command. These should be specified
as key-value pairs :code:`k1=v1 k2=v2 ...`

For example, to run the :ref:`LoRA single-device finetuning <lora_finetune_recipe_label>` recipe with custom model and tokenizer directories, you can provide overrides:

.. code-block:: bash

    tune run lora_finetune_single_device \
    --config llama2/7B_lora_single_device \
    checkpointer.checkpoint_dir=/home/my_model_checkpoint \
    checkpointer.checkpoint_files=['file_1','file_2'] \
    tokenizer.path=/home/my_tokenizer_path

Overriding components
^^^^^^^^^^^^^^^^^^^^^
If you would like to override a class or function in the config that is instantiated
via the :code:`_component_` field, you can do so by assigning to the parameter
name directly. Any nested fields in the components can be overridden with dot notation.

.. code-block:: yaml

    dataset:
      _component_: torchtune.datasets.alpaca_dataset

.. code-block:: bash

    # Change to slimorca_dataset and set train_on_input to True
    tune run lora_finetune_single_device --config my_config.yaml \
    dataset=torchtune.datasets.slimorca_dataset dataset.train_on_input=True

Removing config fields
^^^^^^^^^^^^^^^^^^^^^^
You may need to remove certain parameters from the config when changing components
through overrides that require different keyword arguments. You can do so by using
the `~` flag and specify the dotpath of the config field you would like to remove.
For example, if you want to override a built-in config and use the
`bitsandbytes.optim.PagedAdamW8bit <https://huggingface.co/docs/bitsandbytes/main/en/reference/optim/adamw#bitsandbytes.optim.PagedAdamW8bit>`_
optimizer, you may need to delete parameters like ``foreach`` which are
specific to PyTorch optimizers. Note that this example requires that you have `bitsandbytes <https://github.com/bitsandbytes-foundation/bitsandbytes>`_
installed.

.. code-block:: yaml

    # In configs/llama3/8B_full.yaml
    optimizer:
      _component_: torch.optim.AdamW
      lr: 2e-5
      foreach: False

.. code-block:: bash

    # Change to PagedAdamW8bit and remove fused, foreach
    tune run --nproc_per_node 4 full_finetune_distributed --config llama3/8B_full \
    optimizer=bitsandbytes.optim.PagedAdamW8bit ~optimizer.foreach
