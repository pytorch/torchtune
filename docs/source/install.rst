.. _install_label:

====================
Install Instructions
====================

**Pre-requisites**: torchtune requires PyTorch, so please install for your proper host and environment
using the `Start Locally <https://pytorch.org/get-started/locally/>`_ page.

Install via PyPI
----------------

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

|

Install via ``git clone``
-------------------------

If you want the latest and greatest features from torchtune or if you want to become a contributor,
you can also install the package locally with the following command.

.. code-block:: bash

    git clone https://github.com/pytorch/torchtune.git
    cd torchtune
    pip install -e .

|

Install nightly build
---------------------

torchtune gets built every evening with the latest commits to ``main`` branch. If you want the latest updates
to the package *without* installing via ``git clone``, you can install with the following command:

.. code-block:: bash

    pip install --pre torchtune --extra-index-url https://download.pytorch.org/whl/test/cpu --no-cache-dir

.. note::

    ``--no-cache-dir`` will direct ``pip`` to not look for a cached version of torchtune, thereby overwriting
    your existing torchtune installation.

If you already have PyTorch installed, torchtune will default to using that version. However, if you want to
use the nightly version of PyTorch, you can append the ``--force-reinstall`` option to the above command. If you
opt for this install method, you will likely need to change the "cpu" suffix in the index url to match your CUDA
version. For example, if you are running CUDA 12, your index url would be "https://download.pytorch.org/whl/test/cu121".
