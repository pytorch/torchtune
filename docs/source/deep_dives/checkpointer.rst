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

Let's look at the two popular formats for Llama 3.2.

**Meta Format**

This is the format supported by the official Llama 3.2 implementation. When you download the Llama 3.2 3B model
from the `meta-llama website <https://llama.meta.com/llama-downloads>`_, you'll get access to a single
``.pth`` checkpoint file. You can inspect the contents of this checkpoint easily with ``torch.load``

.. code-block:: python

    >>> import torch
    >>> state_dict = torch.load('consolidated.00.pth', mmap=True, weights_only=True, map_location='cpu')
    >>> # inspect the keys and the shapes of the associated tensors
    >>> for key, value in state_dict.items():
    >>>    print(f'{key}: {value.shape}')

    tok_embeddings.weight: torch.Size([128256, 3072])
    ...
    ...
    >>> print(len(state_dict.keys()))
    255

The state_dict contains 255 keys, including an input embedding table called ``tok_embeddings``. The
model definition for this state_dict expects an embedding layer with ``128256`` tokens each having a
embedding with dim of ``3072``.


**HF Format**

This is the most popular format within the Hugging Face Model Hub and is
the default format in every torchtune config. This is also the format you get when you download the
llama3.2 model from the `Llama-3.2-3B-Instruct <https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct>`_ repo.

The first big difference is that the state_dict is split across two ``.safetensors`` files. To correctly
load the checkpoint, you'll need to piece these files together. Let's inspect one of the files.

.. code-block:: python

    >>> from safetensors import safe_open
    >>> state_dict = {}
    >>> with safe_open("model-00001-of-00002.safetensors", framework="pt", device="cpu") as f:
    >>>     for k in f.keys():
    >>>         state_dict[k] = f.get_tensor(k)

    >>> # inspect the keys and the shapes of the associated tensors
    >>> for key, value in state_dict.items():
    >>>     print(f'{key}: {value.shape}')

    model.embed_tokens.weight: torch.Size([128256, 3072])
    ...
    ...
    >>> print(len(state_dict.keys()))
    187

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

- Download the model from the HF repo using tune download. This will ignore the "pth"
  files, since we will be loading the "safetensors".

    |

    .. code-block:: bash

       tune download meta-llama/Llama-3.2-3B-Instruct \
       --output-dir /tmp/Llama-3.2-3B-Instruct \
       --ignore-patterns "original/consolidated.00.pth"

- Use ``output_dir`` specified here as the ``checkpoint_dir`` argument for the checkpointer.

|

The following snippet explains how the HFCheckpointer is setup in torchtune config files.

.. code-block:: yaml

    checkpointer:

        # checkpointer to use
        _component_: torchtune.training.FullModelHFCheckpointer

        # directory with the checkpoint files
        # this should match the folder you used when downloading the model
        checkpoint_dir: /tmp/Llama-3.2-3B-Instruct

        # checkpoint files. For the Llama-3.2-3B-Instruct model we have
        # 2 .safetensor files. The checkpointer takes care of sorting
        # by id and so the order here does not matter
        checkpoint_files: [
            model-00001-of-00002.safetensors,
            model-00002-of-00002.safetensors,
        ]

        # dir for saving the output checkpoints
        output_dir: <output_dir>

        # model_type which specifies how to convert the state_dict
        # into a format which torchtune understands
        model_type: LLAMA3_2

    # set to True if restarting training. More on that later.
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

        tune download meta-llama/Llama-3.2-3B-Instruct \
        --output-dir /tmp/Llama-3.2-3B-Instruct \
        --ignore-patterns "*.safetensors"

- Use ``output_dir`` above as the ``checkpoint_dir`` for the checkpointer.

|

The following snippet explains how the MetaCheckpointer is setup in torchtune config files.

.. code-block:: yaml

    checkpointer:

        # checkpointer to use
        _component_: torchtune.training.FullModelMetaCheckpointer

        # directory with the checkpoint files
        # this should match the folder you used when downloading the model
        checkpoint_dir: <checkpoint_dir>

        # checkpoint files. For the llama3.2 3B model we have
        # a single .pth file
        checkpoint_files: [consolidated.00.pth]

        # dir for saving the output checkpoints.
        output_dir: <checkpoint_dir>

        # model_type which specifies how to convert the state_dict
        # into a format which torchtune understands
        model_type: LLAMA3_2

    # set to True if restarting training. More on that later.
    resume_from_checkpoint: False

|

:class:`TorchTuneCheckpointer <torchtune.training.FullModelTorchTuneCheckpointer>`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This checkpointer reads and writes checkpoints in a format that is compatible with torchtune's
model definition. This does not perform any state_dict conversions and is currently used either
for testing or for loading quantized models for generation.

|

Checkpoint Output
---------------------------------

Congrats for getting this far! Let's say you have followed our :ref:`End-to-End Workflow with torchtune <e2e_flow>` and trained a llama 3.2 3B using one of our LoRA recipes.

Now let's visualize the outputs. A simple way of doing this is by running :code:`tree -a path/to/outputdir`, which should show something like the tree below.
There are 3 types of folders:

1) **recipe_state**: Holds recipe_state.pt with the information necessary to restart your training run from the last intermediate epoch. More on that later;
2) **logs**: Outputs of your metric_logger, if any;
3) **epoch_{}**: Contains your trained model weights plus model metadata. If running inference or pushing to a model hub, you should use this folder directly;

.. note::
     For each epoch, we copy the contents of the original checkpoint folder, excluding the original checkpoints and large files.
     These files are lightweight, mostly configuration files, and make it easier for the user to use the epoch folders directly in downstream applications.

For more details about each file, please check the End-to-End tutorial mentioned above.

    .. code-block:: bash

        >>> tree -a /tmp/torchtune/llama3_2_3B/lora_single_device
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
        ├── logs
        │   └── log_1734652101.txt
        └── recipe_state
            └── recipe_state.pt


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

Resuming from checkpoint - Full Finetuning
------------------------------------------

Sometimes our training is interrupted for some reason. To restart training from a previous checkpoint file,
you'll need to **update** the following fields in your configs:

**resume_from_checkpoint**: Set it to True;

**checkpoint_files**: change the path to ``epoch_{YOUR_EPOCH}/model-{}-of-{}.safetensors``;

Notice that we do **not** change our checkpoint_dir or output_dir. Since we are resuming from checkpoint, we know
to look for it in the output_dir.

.. code-block:: yaml

    checkpointer:
        # [... rest of the config...]

        # checkpoint files. Note that you will need to update this
        # section of the config with the intermediate checkpoint files
        checkpoint_files: [
            epoch_{YOUR_EPOCH}/model-00001-of-00002.safetensors,
            epoch_{YOUR_EPOCH}/model-00001-of-00002.safetensors,
        ]

    # set to True if restarting training
    resume_from_checkpoint: True


Resuming from checkpoint - LoRA Finetuning
------------------------------------------

Similarly to full finetuning, we will also only need to modify two fields: ``resume_from_checkpoint``
and ``adapter_checkpoint``, which will be loaded from ``output_dir``. We do NOT have to modify ``checkpoint_files``,
because the base model being loaded is still the same. You can optionally leave ``adapter_checkpoint`` empty.
In this case, we will look for it in the last saved epoch folder.

.. code-block:: yaml

    checkpointer:
        # [... rest of the config...]

        # adapter_checkpoint. You will need to update this with the intermediate checkpoint files.
        # It can be empty if resuming from last epoch.
        adapter_checkpoint: epoch_{YOUR_EPOCH}/adapter_model.pt

    # set to True if restarting training
    resume_from_checkpoint: True

    # set to True to save only the adapter weights
    # it does not influence resuming_from_checkpointing
    save_adapter_weights_only: False

.. note::
    In torchtune, we output both the adapter weights and the full model merged weights
    for LoRA. The merged checkpoint is a convenience, since it can be used without having special
    tooling to handle the adapters. However, they should **not** be used when resuming
    training, as loading the merged weights + adapter would be an error. Therefore, when resuming for LoRA,
    we will take the original untrained weigths from checkpoint dir, and the trained
    adapters from output_dir. For more details, take a look at our :ref:`LoRA Finetuning Tutorial <lora_finetune_label>`.

.. note::
    Additionally, by setting the option :code:`save_adapter_weights_only`, you can choose to **only** save the adapter weights.
    This reduces the amount of storage and time needed to save the checkpoint, but has no influence over resuming from checkpoint.

|

Putting this all together
-------------------------

Let's now put all of this knowledge together! We'll load some checkpoints,
create some models and run a simple forward.

For this section we'll use the Llama-3.2-3B-Instruct model in HF format.

.. code-block:: python

    import torch
    from torchtune.models.llama3_2 import llama3_2_3b
    from torchtune.training import FullModelHFCheckpointer

    # Set the right directory and files
    checkpoint_dir = "/tmp/Llama-3.2-3B-Instruct/"
    output_dir = "/tmp/torchtune/llama3_2_3B/full_single_device"

    pytorch_files = [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]

    # Set up the checkpointer and load state dict
    checkpointer = FullModelHFCheckpointer(
        checkpoint_dir=checkpoint_dir,
        checkpoint_files=pytorch_files,
        output_dir=output_dir,
        model_type="LLAMA3_2",
    )
    torchtune_sd = checkpointer.load_checkpoint()

    # Setup the model and the input
    model = llama3_2_3b()

    # Model weights are stored with the key="model"
    model.load_state_dict(torchtune_sd["model"])
    model.to("cuda")

    # We have 128256 vocab tokens; lets generate an input with 24 tokens
    x = torch.randint(0, 128256, (1, 24), dtype=torch.long, device="cuda")

    tensor([[[ 1.4299,  1.1658,  4.2459,  ..., -2.3259, -2.3262, -2.3259],
            [ 6.5942,  7.2284,  2.4090,  ..., -6.0129, -6.0121, -6.0127],
            [ 5.6462,  4.8787,  4.0950,  ..., -4.6460, -4.6455, -4.6457],
            ...,
            [-0.4156, -0.0626, -0.0362,  ..., -3.6432, -3.6437, -3.6427],
            [-0.5679, -0.6902,  0.5267,  ..., -2.6137, -2.6138, -2.6127],
            [ 0.3688, -0.1350,  1.1764,  ..., -3.4563, -3.4565, -3.4564]]],
        device='cuda:0')


You can do this with any model supported by torchtune. You can find a full list
of models and model builders :ref:`here <models>`.

We hope this deep-dive provided a deeper insight into the checkpointer and
associated utilities in torchtune. Happy tuning!
