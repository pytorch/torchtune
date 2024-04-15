.. _wandb_logging:

===========================
Logging to Weights & Biases
===========================

.. customcarditem::
   :header: Logging to Weights & Biases
   :card_description: Log metrics and model checkpoints to W&B
   :image: _static/img/torchtune_workspace.png
   :link: examples/wandb_logging.html
   :tags: logging,wandb


Torchtune supports logging your training runs to [Weights & Biases](https://wandb.ai).

.. note::

  You will need to install the `wandb`` package to use this feature.
  You can install it via pip:

  .. code-block:: bash

    pip install wandb


Metric Logger
-------------

The only change you need to make is to add the metric logger to your config. Weights & Biases will log the metrics and model checkpoints for you.

.. code-block:: python
    # enable logging to the built-in WandBLogger
    metric_logger:
      _component_: torchtune.utils.metric_logging.WandBLogger
      # the W&B project to log to
      project: torchtune


We automatically grab the config from the recipe you are running and log it to W&B. You can find it in the W&B overview tab and the actual file in the `Files` tab.

.. note::

  Click on this sample [project to see the W&B workspace](https://wandb.ai/capecape/torchtune)
  The config used to train the models can be found [here](https://wandb.ai/capecape/torchtune/runs/6053ofw0/files/torchtune_config_j67sb73v.yaml)

Logging Model Checkpoints to W&B
-------------------------------

You can also log the model checkpoints to W&B by modifying the desired script `save_checkpoint` method.

A suggested approach would be something like this:

.. code-block:: python

      def save_checkpoint(self, epoch: int) -> None:
        ...
        ## Let's save the checkpoint to W&B
        ## depending on the Checkpointer Class the file will be named differently
        ## Here is an example for the full_finetune case
        checkpoint_file = Path.joinpath(
            self._checkpointer._output_dir, f"torchtune_model_{epoch}"
        ).with_suffix(".pt")
        wandb_at = wandb.Artifact(
          name=f"torchtune_model_{epoch}",
          type="model",
          # description of the model checkpoint
          description="Model checkpoint",
          # you can add whatever metadata you want as a dict
          metadata={
            utils.SEED_KEY: self.seed,
            utils.EPOCHS_KEY: self.epochs_run,
            utils.TOTAL_EPOCHS_KEY: self.total_epochs,
            utils.MAX_STEPS_KEY: self.max_steps_per_epoch,
            }
        )
        wandb_at.add_file(checkpoint_file)
        wandb.log_artifact(wandb_at)
