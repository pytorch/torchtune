.. _install_label:

====================
Install Instructions
====================

There is currently only one way to download TorchTune, which is locally.

.. code-block:: bash

    git clone https://github.com/pytorch-labs/torchtune.git
    cd torchtune
    pip install -e .

To confirm that the package is installed correctly, you can run the following command:

.. code-block:: bash

    tune recipe --help

And should see the following output:

::

    usage: tune recipe

    Utility for information relating to recipes

    positional arguments:

        list      List recipes
        cp        Copy recipe to local path

    options:
    -h, --help  show this help message and exit
