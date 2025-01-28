.. _multinode_tutorial:

=====================
Multi-node finetuning
=====================

Congratulations! After years of being "GPU poor", you've worked hard, saved your hard earned Bitcoin and
now have access to a proper multi-node cluster. You're part of the so-called "GPU middle class". In many ways,
your worries of yesteryear are gone: memory efficient training? Who cares! But in many other ways, your problems
are just starting because multi-node is a whole new beast. Come with me as I take you through your new life, complete with
a big backyard, new car, and of course - a nice rack of H100s.

.. grid:: 2

    .. grid-item-card:: :octicon:`mortar-board;1em;` You will learn:

      * How to set up the torchtune package on a SLURM cluster
      * How to fine-tune a Llama3.3 70B model w/ full parameter updates (not LoRA)
      * What common errors to lookout for

    .. grid-item-card:: :octicon:`list-unordered;1em;` Prerequisites

      * Be familiar with distributed training in torchtune
      * Already know basic SLURM commands


Advantages of multi-node training
---------------------------------

It's likely that if you're reading this tutorial, you don't need a refresher on the advantages of having
MORE compute, but let's go over it again so you can appreciate how lucky you are. Let's consider a simplified calculation
on how much memory is required to train a 70B parameter model in bfloat16.

.. code-block:: text

    Weights                            140 GB
    + Optim state (AdamW)              280 GB
    + Activations (bsz=8,seq_len=2048) XX
    ------------------------------------------
                                        280 GB

Right now the average GPU has 80GB of VRAM so definitely can't fit on a single GPU and even multiple GPUs won't be up to the task.
We have a ton of memory optimizations in torchtune that allow you to fit larger models in less resource.

Why might you want to use multi-node then?
* Larger models (like Llama 405B, Deepseek, etc)
* Potentially faster training via larger batch sizes, no activation checkpointing
* Potentially more accurate training with full parameter updates and non-approximate optimizers, etc

.. note::

    **Low inter-node bandwidth & FSDP**
    We utilize <FSDP> to distribute models over multiple devices. In order to distribute training, FSDP runs an all-gather operation for each forward pass and an all-gather plus a scatter-reduce
    operation for each backwards pass. These operations (usually) block training from continuing until completed and with a slow inter-node connection, training speed may be reduced.

Training Llama3.3 70B on 2 nodes
--------------------------------

With that background out of the way, let's get training! We'll be utilizing a common cluster setup called SLURM and we assume you have a decent working knowledge for this tutorial.
First, we need to install torchtune on your cluster. Although pretty much as straightforward as the <link> normal install instructions,
it's recommended that you install into a virtual environment that is accessible from nodes in your cluster - something like a shared filesystem.

Next, using the same idea as above, we need to download the Llama3.3 70B model to the shared fs. (You'll need to make sure you have the correct
credentials as noted before.)

.. code-block:: bash

    $ tune download meta-llama/Llama-3.3-70B-Instruct --ignore-patterns "consolidated/*.pth" --output-dir SHARED_FS/Llama-3.3-70B-Instruct

Now that we have a downloaded model, we can launch training. Although you can *technically* launch the multinode bash script from the tune CLI,
it's recommended that you copy the file to your machine.

.. code-block:: bash

    $ tune cp full_finetune_multinode .

And let's open it up to see what's inside:

.. literalinclude:: ../../../recipes/full_finetune_multinode.slurm

What are the high level parts?
* Uses `full_finetune_distributed` to launch training
* Can specify number of nodes, tasks, CPUs available, etc
* Should consider several cluster-specific environment variables

We just need to point to our checkpoint and output dir and get training!

> You may need to set your interface which you can find with ipconfig

Once we've trained, we can follow the instructions [here] in order to upload our beautiful new model to the Hugging Face Hub.

Future development
------------------

2D parallelism

Longer context (ring attention, etc)

What else do you want?

BLAH BLHAH BALSHD 很好
