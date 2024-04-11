.. _wandb_logging:

===========================
Logging to Weights & Biases
===========================

Torchtune supports logging your training runs to [Weights & Biases](https://wandb.ai).

.. note::

  You will need to install the `wandb`` package to use this feature.
  You can install it via pip:

  .. code-block:: bash

    pip install wandb

An example config for enabling Weights & Biases logging on the full llama2 7b finetune recipe is as follows:

.. code-block:: yaml

    # Tokenizer
    tokenizer:
      _component_: torchtune.models.llama2.llama2_tokenizer
      path: /tmp/tokenizer.model

    # Dataset
    dataset:
      _component_: torchtune.datasets.alpaca_dataset
    shuffle: True

    # Model Arguments
    model:
      _component_: torchtune.models.llama2.llama2_7b

    checkpointer:
      _component_: torchtune.utils.FullModelMetaCheckpointer
      checkpoint_dir: /tmp/llama2
      checkpoint_files: [consolidated.00.pth]
      recipe_checkpoint: null
      output_dir: /tmp/llama2
      model_type: LLAMA2
    resume_from_checkpoint: False

    # Fine-tuning arguments
    batch_size: 2
    epochs: 3
    optimizer:
      _component_: torch.optim.SGD
      lr: 2e-5
    loss:
      _component_: torch.nn.CrossEntropyLoss
    output_dir: /tmp/alpaca-llama2-finetune

    device: cuda
    dtype: bf16

    enable_activation_checkpointing: True
    
    log_every_n_steps: 1
    
    metric_logger:
      _component_: torchtune.utils.metric_logging.WandBLogger
      project: torchtune
      log_model: checkpoint



Metric Logger
-------------

The only change you need to make is to add the metric logger to your config. Weights & Biases will log the metrics and model checkpoints for you.

.. code-block:: python
    # enable logging to the built-in WandBLogger
    metric_logger:
      _component_: torchtune.utils.metric_logging.WandBLogger
      # the W&B project to log to
      project: torchtune
      # How often to log the model. Options are "null", "checkpoint" and "end"
      # "null" means no logging
      # "checkpoint" means log the model at each checkpoint (managed by the checkpointer)
      # "end" means log the model at the end of the run
      log_model: checkpoint

We automatically grab the config from the recipe you are running and log it to W&B. You can find it in the W&B overview tab and the actual file in the `Files` tab.

.. note::

  Click on this sample [project to see the W&B workspace](https://wandb.ai/capecape/torchtune)
  The config used to train the models can be found [here](https://wandb.ai/capecape/torchtune/runs/6053ofw0/files/torchtune_config_j67sb73v.yaml)

