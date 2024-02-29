.. _config_tutorial_label:

=================
Configs Deep-Dive
=================

This tutorial will guide you through writing configs for running recipes.

.. grid:: 2

    .. grid-item-card:: :octicon:`mortar-board;1em;` What you will learn

      * How to write a YAML config and run a recipe with it
      * How to use :code:`instantiate` and :code:`parse` APIs
      * How to effectively use configs and CLI overrides for running recipes

    .. grid-item-card:: :octicon:`list-unordered;1em;` Prerequisites

      * Be familiar with the :ref:`overview of TorchTune<overview_label>`
      * Make sure to :ref:`install TorchTune<install_label>`
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
Configs serve as the primary entry point for running recipes in TorchTune. They are
expected to be YAML files and they simply list out values for parameters you want to define
for a particular run.

.. code-block:: yaml

    seed: null
    shuffle: True
    device: cuda
    dtype: fp32
    enable_fsdp: True
    ...

Configuring components using :code:`instantiate`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Many fields will require specifying TorchTune objects with associated keyword
arguments as parameters. Models, datasets, optimizers, and loss functions are
common examples of this. You can easily do this using the :code:`_component_`
subfield. In :code:`_component_`, you need to specify the dotpath of the object
you wish to instantiate in the recipe. The dotpath is the exact path you would use
to import the object normally in a Python file. For example, to specify the
:class:`~torchtune.datasets._alpaca.AlpacaDataset` in your config with custom
arguments:

.. code-block:: yaml

    dataset:
      _component_: torchtune.datasets.AlpacaDataset
      train_on_input: False

Here, we are changing the default value for :code:`train_on_input` from :code:`True`
to :code:`False`.

.. note::
    :class:`~torchtune.datasets._alpaca.AlpacaDataset` is located in a private file,
    :code:`_alpaca.py`, but is exposed as public in :code:`torchtune/datasets/__init__.py`.
    When specifying dotpaths in your config, use the public path and not the private
    path for guarantee of API stability, i.e., :code:`torchtune.datasets.AlpacaDataset`
    and not :code:`torchtune.datasets._alpaca.AlpacaDataset`. There should not be
    underscores in your dotpath.

Once you've specified the :code:`_component_` in your config, you can create an
instance of the specified object in your recipe's setup like so:

.. code-block:: python

    from torchtune import config

    # Access the dataset field and create the object instance
    dataset = config.instantiate(cfg.dataset)

This will automatically use any keyword arguments specified in the fields under
:code:`dataset`.

As written, the preceding example will actually throw an error. If you look at the constructor for :class:`~torchtune.datasets._alpaca.AlpacaDataset`,
you'll notice that we're missing a required positional argument, the tokenizer.
Since this is another configurable TorchTune object, let's understand how to handle
this by taking a look at the :func:`~torchtune.config._instantiate.instantiate` API.

.. code-block:: python

    def instantiate(
        config: DictConfig,
        *args: Tuple[Any, ...],
        **kwargs: Dict[str, Any],
    )

:func:`~torchtune.config._instantiate.instantiate` also accepts positional arguments
and keyword arguments and automatically uses that with the config when creating
the object. This means we can not only pass in the tokenizer, but also add additional
keyword arguments not specified in the config if we'd like:

.. code-block:: python

    from torchtune import config

    tokenizer = config.instantiate(cfg.tokenizer)
    dataset = config.instantiate(
        cfg.dataset,
        tokenizer,
        use_clean=True,
    )

Note that additional keyword arguments will overwrite any duplicated keys in the
config.

Referencing other config fields with interpolations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Sometimes you need to use the same value more than once for multiple fields. You
can use *interpolations* to reference another field, and :func:`~torchtune.config._instantiate.instantiate`
will automatically resolve it for you.

.. code-block:: yaml

    output_dir: /tmp/alpaca-llama2-finetune
    metric_logger:
      _component_: torchtune.utils.metric_logging.DiskLogger
      log_dir: ${output_dir}


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
      _component_: torchtune.datasets.AlpacaDataset
      train_on_input: True
    slimorca_dataset:
      ...

    # do this
    dataset:
      # change this in config or override when needed
      _component_: torchtune.datasets.AlpacaDataset
      train_on_input: True

Order in config is order of instantiation
"""""""""""""""""""""""""""""""""""""""""
There will be many cases where a configured component requires another configured
component as an argument (for example, the tokenizer used with the dataset, or
the model parameters in the optimizer). The order of instantiation is handled in
the recipe code itself, but keeping the same order in the config can help communicate
this to other users and also help remind yourself when a configured component needs
to be passed in as an argument.

.. code-block:: yaml

    # Tokenizer is needed for the dataset, configure it first
    tokenizer:
      _component_: torchtune.models.llama2_tokenizer
      path: /tmp/tokenizer.model

    dataset:
      _component_: torchtune.datasets.AlpacaDataset
      train_on_input: True

Use public APIs only
""""""""""""""""""""
If a component you wish to specify in a config is located in a private file, use
the public dotpath in your config. These components are typically exposed in their
parent module's :code:`__init__.py` file. This way, you can guarantee the stability
of the API you are using in your config.

.. code-block:: yaml

    # don't do this
    dataset:
      _component_: torchtune.datasets._alpaca.AlpacaDataset
      train_on_input: True

    # do this
    dataset:
      _component_: torchtune.datasets.AlpacaDataset
      train_on_input: True


Command-line overrides
----------------------
Configs are the primary location to collect all your parameters to run a recipe,
but sometimes you may want to quickly try different values without having to update
the config itself. To enable quick experimentation, you can specify override values
to parameters in your config via the :code:`tune` command. These should be specified
with the flag :code:`--override k1=v1 k2=v2 ...`

For example, to run the :code:`full_finetune` recipe with custom model and tokenizer directories and using GPUs, you can provide overrides:

.. code-block:: bash

    tune full_finetune --config alpaca_llama2_full_finetune --override model_directory=/home/my_model_checkpoint tokenizer_directory=/home/my_tokenizer_checkpoint device=cuda
