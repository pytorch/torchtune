.. _understand_checkpointer:

==============================
Understanding the Checkpointer
==============================

This tutorial will walk you through the design and behavior of the checkpointer and associated
utilities.


Overview
--------

TorchTune checkpointers are designed to be composable components which can be plugged
into any recipe - training, evaluation or generation. Each checkpointer supports a
set of models and scenarios making these easy to understand, debug and extend.

TorchTune is designed to be "state-dict invariant".

- At the input, TorchTune accepts checkpoints from multiple sources in multiple formats.
  For Llama2 this includes both the HF Hub and the Meta Llama website. Model users don't
  have to worry about explicitly converting checkpoints every time they run a recipe.

- At the output, TorchTune produces checkpoints in the same format as the source. This includes
  converting the state_dict back into the original form and splitting the keys and weights
  across the same number of files. As a result, users should be able to use these fine-tuned
  checkpoints with any post-training tool (quantization, eval, inference) which supports the
  source format.

To be "state-dict invariant", the ``load_checkpoint`` and
``save_checkpoint`` methods make use of the weight convertors available
`here <https://github.com/pytorch/torchtune/blob/main/torchtune/models/convert_weights.py>`_.


|

Checkpoint Formats
------------------

TorchTune supports three different
`checkpointers <https://github.com/pytorch/torchtune/blob/main/torchtune/utils/_checkpointing/_checkpointer.py>`_,
each of which supports a different checkpoint format.


**HFCheckpointer**

This checkpointer reads and writes checkpoints in a format which is compatible with the transformers
framwork from Hugging Face. This is the most popular format within the Hugging Face Model Hub and is
the default format in every TorchTune config. Examples include the llama2 models with the "hf" suffix
in the repo-id, such as `Llama-2-7b-hf <https://huggingface.co/meta-llama/Llama-2-7b-hf>`_.

For this checkpointer to work correctly, we assume that checkpoint_dir contains the necessary checkpoint
and json files. The easiest way to make sure everything works correctly is to use the following flow:

- Download the model from the HF repo using tune download. By default, this will ignore the "safetensors"
  files.

    |

    .. code-block:: bash

        tune download meta-llama/Llama-2-7b-hf
        --output-dir <checkpoint_dir>
        --hf-token <hf-token>

- Use ``output_dir`` above as the ``checkpoint_dir`` for the checkpointer.

|

The following snippet explains how the HFCheckpointer is setup in TorchTune config files.

.. code-block:: yaml

    checkpointer:

        # checkpointer to use
        _component_: torchtune.utils.FullModelHFCheckpointer

        # directory with the checkpoint files
        # this should match the output_dir above
        checkpoint_dir: <checkpoint_dir>

        # checkpoint files. For the llama2-7b-hf model we have
        # 2 .bin files. The checkpointer takes care of sorting
        # by id and so the order here does not matter
        checkpoint_files: [
            pytorch_model-00001-of-00002.bin,
            pytorch_model-00002-of-00002.bin,
        ]

        # if we're restarting a previous run, we need to specify
        # the file with the checkpoint state. More on this in the
        # next section
        recipe_checkpoint: null

        # dir for saving the output checkpoints. Usually set
        # to be the same as checkpoint_dir
        output_dir: <checkpoint_dir>

        # model_type which specifies how to convert the state_dict
        # into a format which TorchTune understands
        model_type: LLAMA2

    # set to True if restarting training
    resume_from_checkpoint: False

.. note::
    Checkpoint conversion to and from HF's format requires access to model params which are
    read directly from the "config.json" file. This helps ensure we either load the weights
    correctly or error out in case of discrepancy between the HF checkpoint file and TorchTune's
    model implementations. This json file is downloaded from the hub along with the model checkpoints.

|

**MetaCheckpointer**

This checkpointer reads and writes checkpoints in a format which is compatible with the original meta-llama
github repository. Examples include the llama2 models without the "hf" suffix in the repo-id,
such as `Llama-2-7b <https://huggingface.co/meta-llama/Llama-2-7b>`_.


For this checkpointer to work correctly, we assume that checkpoint_dir contains the necessary checkpoint
and json files. The easiest way to make sure everything works correctly is to use the following flow:

- Download the model from the HF repo using tune download. By default, this will ignore the "safetensors"
  files.

    |

    .. code-block:: bash

        tune download meta-llama/Llama-2-7b
        --output-dir <checkpoint_dir>
        --hf-token <hf-token>

- Use ``output_dir`` above as the ``checkpoint_dir`` for the checkpointer.

|

The following snippet explains how the MetaCheckpointer is setup in TorchTune config files.

.. code-block:: yaml

    checkpointer:

        # checkpointer to use
        _component_: torchtune.utils.FullModelMetaCheckpointer

        # directory with the checkpoint files
        # this should match the output_dir above
        checkpoint_dir: <checkpoint_dir>

        # checkpoint files. For the llama2-7b model we have
        # a single .pth file
        checkpoint_files: [consolidated.00.pth]

        # if we're restarting a previous run, we need to specify
        # the file with the checkpoint state. More on this in the
        # next section
        recipe_checkpoint: null

        # dir for saving the output checkpoints. Usually set
        # to be the same as checkpoint_dir
        output_dir: <checkpoint_dir>

        # model_type which specifies how to convert the state_dict
        # into a format which TorchTune understands
        model_type: LLAMA2

    # set to True if restarting training
    resume_from_checkpoint: False

|

**TorchTuneCheckpointer**

This checkpointer reads and writes checkpoints in a format that is compatible with TorchTune's
model definition. This does not perform any state_dict conversions and is currently used either
for testing or for loading quantized models for generation.

|


Intermediate vs Final Checkpoints
---------------------------------

TorchTune Checkpointers support two checkpointing scenarios:

**End-of-training Checkpointing**

The model weights at the end of a completed training
run are written out to file. The checkpointer ensures that the output checkpoint
files have the same keys as the input checkpoint file used to begin training. The
checkpointer also ensures that the keys are paritioned across the same number of
files as the original checkpoint. The output state dict has the following
standard format:

  .. code-block:: python

            {
                "key_1": weight_1,
                "key_2": weight_2,
                ...
            }


**Mid-training Chekpointing**.

If checkpointing in the middle of training, the output checkpoint needs to store additional
information to ensure that subsequent training runs can be correctly restarted. In addition to
the model checkpoint files, we output a ``recipe_state.pt`` file for intermediate
checkpoints. These are currently output at the end of each epoch, and contain information
such as optimizer state, number of epochs completed etc.

To prevent us from flooding ``output_dir`` with checkpoint files, the recipe state is
overwritten at the end of each epoch.

The output state dicts have the following formats:

    .. code-block:: python

        Model:
            {
                "key_1": weight_1,
                "key_2": weight_2,
                ...
            }

        Recipe State:
            {
                "optimizer": ...,
                "epoch": ...,
                ...
            }

To restart from a previous checkpoint file, you'll need to make the following changes
to the config file

.. code-block:: yaml

    checkpointer:

        # checkpointer to use
        _component_: torchtune.utils.FullModelHFCheckpointer

        # directory with the checkpoint files
        # this should match the output_dir above
        checkpoint_dir: <checkpoint_dir>

        # checkpoint files. These refer to intermediate checkpoints
        # and will always have a ".pt" extensions
        checkpoint_files: [
            hf_model_0001_0.pt,
            hf_model_0002_0.pt,
        ]

        # if we're restarting a previous run, we need to specify
        # the file with the checkpoint state
        recipe_checkpoint: recipe_state.pt

        # dir for saving the output checkpoints. Usually set
        # to be the same as checkpoint_dir
        output_dir: <checkpoint_dir>

        # model_type which specifies how to convert the state_dict
        # into a format which TorchTune understands
        model_type: LLAMA2

    # set to True if restarting training
    resume_from_checkpoint: True


Checkpointing for LoRA
----------------------

In TorchTune, we output both the adapter weights and the full model "merged" weights
for LoRA. The "merged" checkpoint can be used just like you would use the source
checkpoint with any post-training tools.

The primary difference between the two use cases is when you want to resume training
from a checkpoint. In this case, the checkpointer needs access to both the initial frozen
base model weights as well as the learnt adapter weights. The config for this scenario
looks something like this:


.. code-block:: yaml

    checkpointer:

        # checkpointer to use
        _component_: torchtune.utils.FullModelHFCheckpointer

        # directory with the checkpoint files
        # this should match the output_dir above
        checkpoint_dir: <checkpoint_dir>

        # checkpoint files. This is the ORIGINAL frozen checkpoint
        # and NOT the merged checkpoint output during training
        checkpoint_files: [
            pytorch_model-00001-of-00002.bin,
            pytorch_model-00002-of-00002.bin,
        ]

        # this refers to the adapter weights learnt during training
        adapter_checkpoint: adapter_0.pt

        # the file with the checkpoint state
        recipe_checkpoint: recipe_state.pt

        # dir for saving the output checkpoints. Usually set
        # to be the same as checkpoint_dir
        output_dir: <checkpoint_dir>

        # model_type which specifies how to convert the state_dict
        # into a format which TorchTune understands
        model_type: LLAMA2

    # set to True if restarting training
    resume_from_checkpoint: True
