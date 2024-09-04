.. _understand_checkpointer:

==========================
Checkpointing in torchtune
==========================

This deep-dive will walk you through the design and behavior of the checkpointer and
associated utilities.

.. grid:: 1

    .. grid-item-card:: :octicon:`mortar-board;1em;` What this deep-dive will cover:

      * Checkpointer design for torchtune
      * Checkpoint formats and how we handle them
      * Checkpointing scenarios: Intermediate vs Final and LoRA vs Full-finetune


Overview
--------

torchtune checkpointers are designed to be composable components which can be plugged
into any recipe - training, evaluation or generation. Each checkpointer supports a
set of models and scenarios making these easy to understand, debug and extend.

Before we dive into the checkpointer in torchtune, let's define some concepts.

|

Checkpoint Format
^^^^^^^^^^^^^^^^^

In this deep-dive, we'll talk about different checkpoint formats and how torchtune handles them.
Let's take a close look at these different formats.

Very simply put, the format of a checkpoint is dictated by the state_dict and how this is stored
in files on disk. Each weight is associated with a string key that identifies it in the state dict.
If the string identifier of the keys in the stored checkpoints don't match up
exactly with those in the model definition, you'll either run into explicit errors (loading the
state dict will raise an exception) or worse - silent errors (loading will succeed but training or
inference will not work as expected). In addition to the keys lining up, you also need the shapes
of the weights (values in the state_dict) to match up exactly with those expected by the model
definition.

Let's look at the two popular formats for Llama2.

**Meta Format**

This is the format supported by the official Llama2 implementation. When you download the Llama2 7B model
from the `meta-llama website <https://llama.meta.com/llama-downloads>`_, you'll get access to a single
``.pth`` checkpoint file. You can inspect the contents of this checkpoint easily with ``torch.load``

.. code-block:: python

    >>> import torch
    >>> state_dict = torch.load('consolidated.00.pth', mmap=True, weights_only=True, map_location='cpu')
    >>> # inspect the keys and the shapes of the associated tensors
    >>> for key, value in state_dict.items():
    >>>    print(f'{key}: {value.shape}')

    tok_embeddings.weight: torch.Size([32000, 4096])
    ...
    ...
    >>> print(len(state_dict.keys()))
    292

The state_dict contains 292 keys, including an input embedding table called ``tok_embeddings``. The
model definition for this state_dict expects an embedding layer with ``32000`` tokens each having a
embedding with dim of ``4096``.


**HF Format**

This is the most popular format within the Hugging Face Model Hub and is
the default format in every torchtune config. This is also the format you get when you download the
llama2 model from the `Llama-2-7b-hf <https://huggingface.co/meta-llama/Llama-2-7b-hf>`_ repo.

The first big difference is that the state_dict is split across two ``.bin`` files. To correctly
load the checkpoint, you'll need to piece these files together. Let's inspect one of the files.

.. code-block:: python

    >>> import torch
    >>> state_dict = torch.load('pytorch_model-00001-of-00002.bin', mmap=True, weights_only=True, map_location='cpu')
    >>> # inspect the keys and the shapes of the associated tensors
    >>> for key, value in state_dict.items():
    >>>     print(f'{key}: {value.shape}')

    model.embed_tokens.weight: torch.Size([32000, 4096])
    ...
    ...
    >>> print(len(state_dict.keys()))
    241

Not only does the state_dict contain fewer keys (expected since this is one of two files), but
the embedding table is called ``model.embed_tokens`` instead of ``tok_embeddings``. This mismatch
in names will cause an exception when you try to load the state_dict. The size of this layer is the
same between the two, which is as expected.

|

As you can see, if you're not careful you'll likely end up making a number of errors just during
checkpoint load and save. The torchtune checkpointer makes this less error-prone by managing state dicts
for you. torchtune is designed to be "state-dict invariant".

- When loading, torchtune accepts checkpoints from multiple sources in multiple formats.
  You don't have to worry about explicitly converting checkpoints every time you run a recipe.

- When saving, torchtune produces checkpoints in the same format as the source. This includes
  converting the state_dict back into the original form and splitting the keys and weights
  across the same number of files.

One big advantage of being "state-dict invariant" is that you should be able to use
fine-tuned checkpoints from torchtune with any post-training tool (quantization, eval, inference)
which supports the source format, without any code changes OR conversion scripts. This is one of the
ways in which torchtune interoperates with the surrounding ecosystem.

.. note::

  To be state-dict "invariant" in this way, the ``load_checkpoint`` and ``save_checkpoint`` methods of each checkpointer
  make use of weight converters which correctly map weights between checkpoint formats. For example, when loading weights
  from Hugging Face, we apply a permutation to certain weights on load and save to ensure checkpoints behave exactly the same.
  To further illustrate this, the Llama family of models uses a
  `generic weight converter function <https://github.com/pytorch/torchtune/blob/898670f0eb58f956b5228e5a55ccac4ea0efaff8/torchtune/models/convert_weights.py#L113>`_
  whilst some other models like Phi3 have their own `conversion functions <https://github.com/pytorch/torchtune/blob/main/torchtune/models/phi3/_convert_weights.py>`_
  which can be found within their model folders.

|

Handling different Checkpoint Formats
-------------------------------------

torchtune supports three different
:ref:`checkpointers<checkpointing_label>`,
each of which supports a different checkpoint format.


:class:`HFCheckpointer <torchtune.training.FullModelHFCheckpointer>`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This checkpointer reads and writes checkpoints in a format which is compatible with the transformers
framework from Hugging Face. As mentioned above, this is the most popular format within the Hugging Face
Model Hub and is the default format in every torchtune config.

For this checkpointer to work correctly, we assume that ``checkpoint_dir`` contains the necessary checkpoint
and json files. The easiest way to make sure everything works correctly is to use the following flow:

- Download the model from the HF repo using tune download. By default, this will ignore the "safetensors"
  files.

    |

    .. code-block:: bash

        tune download meta-llama/Llama-2-7b-hf \
        --output-dir <checkpoint_dir> \
        --hf-token <hf-token>

- Use ``output_dir`` specified here as the ``checkpoint_dir`` argument for the checkpointer.

|

The following snippet explains how the HFCheckpointer is setup in torchtune config files.

.. code-block:: yaml

    checkpointer:

        # checkpointer to use
        _component_: torchtune.training.FullModelHFCheckpointer

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
        # into a format which torchtune understands
        model_type: LLAMA2

    # set to True if restarting training
    resume_from_checkpoint: False

.. note::
    Checkpoint conversion to and from HF's format requires access to model params which are
    read directly from the ``config.json`` file. This helps ensure we either load the weights
    correctly or error out in case of discrepancy between the HF checkpoint file and torchtune's
    model implementations. This json file is downloaded from the hub along with the model checkpoints.

|

:class:`MetaCheckpointer <torchtune.training.FullModelMetaCheckpointer>`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This checkpointer reads and writes checkpoints in a format which is compatible with the original meta-llama
github repository.


For this checkpointer to work correctly, we assume that ``checkpoint_dir`` contains the necessary checkpoint
and json files. The easiest way to make sure everything works correctly is to use the following flow:

- Download the model from the HF repo using tune download. By default, this will ignore the "safetensors"
  files.

    |

    .. code-block:: bash

        tune download meta-llama/Llama-2-7b \
        --output-dir <checkpoint_dir> \
        --hf-token <hf-token>

- Use ``output_dir`` above as the ``checkpoint_dir`` for the checkpointer.

|

The following snippet explains how the MetaCheckpointer is setup in torchtune config files.

.. code-block:: yaml

    checkpointer:

        # checkpointer to use
        _component_: torchtune.training.FullModelMetaCheckpointer

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
        # into a format which torchtune understands
        model_type: LLAMA2

    # set to True if restarting training
    resume_from_checkpoint: False

|

:class:`TorchTuneCheckpointer <torchtune.training.FullModelTorchTuneCheckpointer>`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This checkpointer reads and writes checkpoints in a format that is compatible with torchtune's
model definition. This does not perform any state_dict conversions and is currently used either
for testing or for loading quantized models for generation.

|


Intermediate vs Final Checkpoints
---------------------------------

torchtune Checkpointers support two checkpointing scenarios:

**End-of-training Checkpointing**

The model weights at the end of a completed training
run are written out to file. The checkpointer ensures that the output checkpoint
files have the same keys as the input checkpoint file used to begin training. The
checkpointer also ensures that the keys are partitioned across the same number of
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
        _component_: torchtune.training.FullModelHFCheckpointer

        checkpoint_dir: <checkpoint_dir>

        # checkpoint files. Note that you will need to update this
        # section of the config with the intermediate checkpoint files
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
        # into a format which torchtune understands
        model_type: LLAMA2

    # set to True if restarting training
    resume_from_checkpoint: True


Checkpointing for LoRA
----------------------

In torchtune, we output both the adapter weights and the full model "merged" weights
for LoRA. The "merged" checkpoint can be used just like you would use the source
checkpoint with any post-training tools. For more details, take a look at our
:ref:`LoRA Finetuning Tutorial <lora_finetune_label>`.Additionally, by setting the option "save_adapter_weights_only" to True when saving a checkpoint, you can choose to save only the adapter weights.

The primary difference between the two use cases is when you want to resume training
from a checkpoint. In this case, the checkpointer needs access to both the initial frozen
base model weights as well as the learnt adapter weights. The config for this scenario
looks something like this:


.. code-block:: yaml

    checkpointer:

        # checkpointer to use
        _component_: torchtune.training.FullModelHFCheckpointer

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
        # into a format which torchtune understands
        model_type: LLAMA2

    # set to True if restarting training
    resume_from_checkpoint: True

    # Set to True to save only the adapter weights
    save_adapter_weights_only: False

|

Putting this all together
-------------------------

Let's now put all of this knowledge together! We'll load some checkpoints,
create some models and run a simple forward.

For this section we'll use the Llama2 13B model in HF format.

.. code-block:: python

    import torch
    from torchtune.training import FullModelHFCheckpointer, ModelType
    from torchtune.models.llama2 import llama2_13b

    # Set the right directory and files
    checkpoint_dir = 'Llama-2-13b-hf/'
    pytorch_files = [
        'pytorch_model-00001-of-00003.bin',
        'pytorch_model-00002-of-00003.bin',
        'pytorch_model-00003-of-00003.bin'
    ]

    # Set up the checkpointer and load state dict
    checkpointer = FullModelHFCheckpointer(
        checkpoint_dir=checkpoint_dir,
        checkpoint_files=pytorch_files,
        output_dir=checkpoint_dir,
        model_type=ModelType.LLAMA2
    )
    torchtune_sd = checkpointer.load_checkpoint()

    # Setup the model and the input
    model = llama2_13b()

    # Model weights are stored with the key="model"
    model.load_state_dict(torchtune_sd["model"])
    <All keys matched successfully>

    # We have 32000 vocab tokens; lets generate an input with 70 tokens
    x = torch.randint(0, 32000, (1, 70))

    with torch.no_grad():
        model(x)

    tensor([[[ -6.3989,  -9.0531,   3.2375,  ...,  -5.2822,  -4.4872,  -5.7469],
        [ -8.6737, -11.0023,   6.8235,  ...,  -2.6819,  -4.2424,  -4.0109],
        [ -4.6915,  -7.3618,   4.1628,  ...,  -2.8594,  -2.5857,  -3.1151],
        ...,
        [ -7.7808,  -8.2322,   2.8850,  ...,  -1.9604,  -4.7624,  -1.6040],
        [ -7.3159,  -8.5849,   1.8039,  ...,  -0.9322,  -5.2010,  -1.6824],
        [ -7.8929,  -8.8465,   3.3794,  ...,  -1.3500,  -4.6145,  -2.5931]]])


You can do this with any model supported by torchtune. You can find a full list
of models and model builders :ref:`here <models>`.

We hope this deep-dive provided a deeper insight into the checkpointer and
associated utilities in torchtune. Happy tuning!
