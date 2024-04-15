.. _install_label:

====================
Install Instructions
====================

torchtune requires PyTorch, so please install for your proper host and environment,
using `Start Locally <https://pytorch.org/get-started/locally/>`_ page.

The latest stable version of torchtune is hosted on PyPI and can be downloaded
with the following command:

.. code-block:: bash

    pip install torchtune

To confirm that the package is installed correctly, you can run the following command:

.. code-block:: bash

    tune

And should see the following output:

::

    usage: tune [-h] {download,ls,cp,run,validate} ...

    Welcome to the TorchTune CLI!

    options:
    -h, --help            show this help message and exit

    ...

If you want the latest and greatest features from torchtune or if you want to become a contributor,
you can also install the package locally with the following command.

.. code-block:: bash

    git clone https://github.com/pytorch/torchtune.git
    cd torchtune
    pip install -e .
