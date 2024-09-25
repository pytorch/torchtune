.. _finetune_vlm_label:

=========================================
Fine-Tune Your First VLM: Llama3.2-Vision
=========================================

In this tutorial, we will walk through fine-tuning Llama3.2-Vision-Instruct, a vision-language model (VLM), with a 
multimodal dataset in torchtune.

.. grid:: 2

    .. grid-item-card:: :octicon:`mortar-board;1em;` What you will learn

      * How to configure multimodal datasets
      * How to run inference on and evaluate a VLM

    .. grid-item-card:: :octicon:`list-unordered;1em;` Prerequisites

      * Install torchtune nightly build
      * Download the Llama3.2 model from Hugging Face

After you've downloaded the Llama3.2 model, you can start fine-tuning it right away with ``tune run``. Let's
launching a single device training job with the default dataset, the OCR-VQA subset of The Cauldron.

.. code-block:: bash

    tune run full_finetune_single_device --config llama3_2_vision/11B_full_single_device max_steps_per_epoch=100

You can see where the dataset is defined in the config file.

.. code-block:: yaml

    dataset:
      _component_: torchtune.datasets.multimodal.the_cauldron_dataset
      subset: ocrvqa

You can modify the config to use a different multimodal dataset. See :ref:`multimodal_dataset_usage_label` for available
built-in datasets in torchtune. Let's use the :func:`~torchtune.datasets.multimodal.llava_instruct_dataset` as an example.

.. code-block:: yaml

    # This requires downloading the COCO image dataset separately
    dataset:
      _component_: torchtune.datasets.multimodal.llava_instruct_dataset
      image_dir: /home/user/coco/train2017/

.. code-block:: bash

    tune cp llama3_2_vision/11B_full_single_device ./my_config.yaml
    # Make changes to my_config.yaml
    tune run full_finetune_single_device --config my_config.yaml max_steps_per_epoch=100

You can also use :func:`~torchtune.datasets.multimodal.multimodal_chat_dataset` to define your custom multimodal dataset.
See :ref:`example_multimodal` for more details.

|

Next steps
----------

Now that you have trained your model and set up your environment, let's take a look at what we can do with our
new model by checking out the ":ref:`E2E Workflow Tutorial<e2e_flow>`".
