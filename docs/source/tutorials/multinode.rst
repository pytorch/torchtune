.. _multinode_tutorial:

=====================
Multi-node finetuning
=====================

Congratulations! You've finally escaped the struggles of being "GPU poor" and now have access to a multi-node setup.
You can bid farewell to the days of sweating over memory-efficient optimizations, but get ready for new challenges as you navigate the complexities of distributed computing.

.. grid:: 2

    .. grid-item-card:: :octicon:`mortar-board;1em;` You will learn:

      * Why multi-node training is useful
      * How to set up the torchtune package on a SLURM cluster
      * How to fine-tune a Llama3.3 70B model w/ full parameter updates (not LoRA)

    .. grid-item-card:: :octicon:`list-unordered;1em;` Prerequisites

      * Be familiar with distributed training in torchtune
      * Already know basic SLURM commands

.. _advantages_multi_node_label:

Advantages of multi-node training
---------------------------------

More machines means more memory! This is cool for several reasons:

1. **Bigger models**: With more memory, you can train larger models such as `Llama3.1 405B <https://ai.meta.com/blog/meta-llama-3-1/>`_, `Deepseek-V3 <https://www.deepseek.com/>`_, and more.
2. **Longer data**: For many fine-tuning tasks like writing code, it's helpful to have long context lengths; however longer context length means more memory needed for activations.
3. **Higher quality**: With more memory, you can do full parameter updates (not LoRA) and use optimizers like `AdamW <https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html>`_ (not low-precision optimizers), both of which can potentially improve the quality of your training.
4. **Faster training**: With the ability to fit more data in memory, you can use higher batch sizes *and* turn off memory optimizations like :ref:`activation checkpointing<glossary_act_ckpt>` thereby decreasing the time it takes for training to complete.

.. note::

    **Low inter-node bandwidth & FSDP** We utilize PyTorch's **Fully Sharded Data Parallel** to distribute models over multiple devices. In order to distribute training, FSDP runs an `all-gather <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#allgather>`_ operation
    for each forward pass and an all-gather (usually) plus a `reduce-scatter <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#reducescatter>`_ operation for each backwards pass. These operations (usually) block training from continuing until completed and with a slow
    inter-node connection, training speed may be reduced. For more on this, please refer to `this Github Issue <https://github.com/pytorch/pytorch/issues/102434>`_.

Training Llama3.3 70B on 2 nodes
--------------------------------

Let's get training! We'll be utilizing a common cluster workflow manager called `SLURM <https://slurm.schedmd.com/documentation.html>`_ and assume you have a decent working knowledge of SLURM for this tutorial.
First, we need to install torchtune. Although pretty much as straightforward as the :ref:`normal install instructions<install_label>`,
it's recommended that you install the package into a virtual environment that is accessible from all nodes in your cluster like a shared filesystem.

Next, we need to download the `Llama3.3 70B <https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct>`_ model to your shared filesystem. You'll need to make sure you have the correct credentials following the steps
outlined :ref:`here<tune_download_label>`.

.. code-block:: bash

    $ tune download meta-llama/Llama-3.3-70B-Instruct --ignore-patterns "consolidated/*.pth" --output-dir SHARED_FS/Llama-3.3-70B-Instruct

Now that we have a downloaded model, let's check out our example SLURM bash script.

.. literalinclude:: ../../../recipes/full_finetune_multinode.slurm

**There's a lot of information in this script but here are the high-level parts:**

* We utilize SLURM specific commands like number of nodes, tasks, CPUs available, etc.
* We are using `torchrun <https://pytorch.org/docs/stable/elastic/run.html>`_ and the `full_finetune_distributed <https://github.com/pytorch/torchtune/blob/main/recipes/full_finetune_distributed.py>`_ recipe to train just like on single node
* You can consider several cluster-specific environment variables (``NCCL_BUFFSIZE``, ``NCCL_DEBUG``, ``FI_PROVIDER``, etc.) in order to maximize GPU utilization, debug, and more.

.. note::

    We may need to explicitly set the network interface for distributed backends. You can read more about `PyTorch distributed backends here <https://pytorch.org/docs/stable/distributed.html#common-environment-variables>`_
    but it's also helpful to know that you can find your network interface by running `ipconfig <https://en.wikipedia.org/wiki/Ipconfig#:~:text=ipconfig%20(standing%20for%20%22Internet%20Protocol,ipconfig>`_ from a specific node.

After we update the shared filesystem in the bash script, we can launch using `sbatch <https://slurm.schedmd.com/sbatch.html>`_.

.. code-block:: bash

    sbatch full_finetune_multinode.slurm

And the output of `squeue <https://slurm.schedmd.com/squeue.html>`_ should show our job running:

.. code-block:: bash

    $ squeue
    JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
    1     train         torchtun slurm R       0:03      2 slurm-worker-[1-2]

Once training has completed, which should take roughly seven minutes in total (880 tok/s) with the default config, we can follow the :ref:`instructions here<use_model_in_wild>` in order to upload our beautiful new model to the Hugging Face Hub!

Future development
------------------

We've covered the basics of how to launch a fine-tuning job with SLURM on two nodes with FSDP. There's still more things we're cooking up,
including...

**2D parallelism**: Utilizing both FSDP *and* tensor parallelism in what is commonly referred to as `2D parallelism <https://pytorch.org/tutorials/intermediate/TP_tutorial.html>`_ will decrease memory requirements even further, allowing us to lean even harder
into the advantages listed :ref:`above<advantages_multi_node_label>`.

**Longer context (ring attention, etc)**: More memory and more machines means we can train on longer sequences and tag advantage of neat tricks like ring attention, where tokens are split across
GPUs. You can read more about our plans for torchtune in `this Github RFC <https://github.com/pytorch/torchtune/issues/1244>`_.

**Want other optimizations?** Feel free to let us know by `opening up a Github Issue <https://github.com/pytorch/torchtune/issues/new?q=sort%3Aupdated-desc+is%3Aissue+is%3Aopen&template=Blank+issue>`_ on our repo or `dropping us a line in Discord <https://discord.gg/Zsf8xgT7>`_!
