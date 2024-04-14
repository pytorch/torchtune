.. _install_label:

====================
Install Instructions
====================

There is currently only one way to download torchtune, which is locally.

.. code-block:: bash

    git clone https://github.com/pytorch/torchtune.git
    cd torchtune
    pip install -e .

To confirm that the package is installed correctly, you can run the following command:

.. code-block:: bash

    tune

And should see the following output:

::

    usage: tune [options] <recipe> [recipe_args]
    tune: error: the following arguments are required: recipe, recipe_args
