========================
Llama2 Generation Recipe
========================

.. note::
    Ensure that the documentation version matches the installed TorchTune version

This recipe is used for generating text from a pre-trained or fine tuned Llama2 model. Given a model checkpoint and a prompt, it'll generate text.

This recipe uses :ref:`Generation Utilities<gen_label>`

To run the recipe directly, launch with

.. code-block:: bash

    tune alpaca_generate --config <generate_config>

Recipe
------

.. only:: builder_html or PyTorchdoc

    Copy the recipe directly into your own script or notebook to modify and edit for yourself.

.. literalinclude:: ../../../recipes/alpaca_generate.py
    :pyobject: recipe
