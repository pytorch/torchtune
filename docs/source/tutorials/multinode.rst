.. _multinode_tutorial:

=====================
Multi-node finetuning
=====================

Congratulations! After years of being "GPU poor", you've worked hard, saved your hard earned Bitcoin and graduated to the
so-called **"GPU middle class"**. In many ways, your worries of yesteryear are gone (memory efficient training, who??).
But, new problems are on the horizon for you because multi-node is a whole new beast. Come with me as I take you
through your new life, complete with a big backyard, new car, and of course - a nice rack of H100s.

.. grid:: 2

    .. grid-item-card:: :octicon:`mortar-board;1em;` You will learn:

      * Why multi-node training is useful
      * How to set up the torchtune package on a SLURM cluster
      * How to fine-tune a Llama3.3 70B model w/ full parameter updates (not LoRA)

    .. grid-item-card:: :octicon:`list-unordered;1em;` Prerequisites

      * Be familiar with distributed training in torchtune
      * Already know basic SLURM commands


Advantages of multi-node training
---------------------------------

More machines means more memory! This is cool for several reasons:

1. **Bigger models**: With more memory, you can train larger models such as `Llama3.1 405B <https://ai.meta.com/blog/meta-llama-3-1/>`_, Deepseek-V3, and more.
2. **Longer data**: More many tasks like writing code, it's helpful to have long context lengths; however longer context length means more memory needed for activations.
3. **Higher quality**: With more memory, you can do full parameter updates (not LoRA) and use optimizers like AdamW (not low-precision optimizers),both of which can potentially improve the quality of your training.
4. **Faster training**: With the ability to fit more data in memory, you can use higher batch sizes *and* turn off memory optimizations like :ref:`activation checkpointing<glossary_act_ckpt>` thereby decreasing the time it takes for training to complete.

.. note::

    **Low inter-node bandwidth & FSDP** We utilize Fully Sharded Data Parallel to distribute models over multiple devices. In order to distribute training, FSDP runs an all-gather operation
    for each forward pass and an all-gather plus a scatter-reduce operation for each backwards pass. These operations (usually) block training from continuing until completed and with a slow
    inter-node connection, training speed may be reduced.

Training Llama3.3 70B on 2 nodes
--------------------------------

Let's get training! We'll be utilizing a common cluster setup called SLURM and assume you have a decent working knowledge of SLURM for this tutorial.
First, we need to install torchtune. Although pretty much as straightforward as the normal install instructions,
it's recommended that you install the package into a virtual environment that is accessible from all nodes in your cluster like a shared filesystem.

Next, we need to download the Llama3.3 70B model to the shared fs. (You'll need to make sure you have the correct credentials as noted before.)

.. code-block:: bash

    $ tune download meta-llama/Llama-3.3-70B-Instruct --ignore-patterns "consolidated/*.pth" --output-dir SHARED_FS/Llama-3.3-70B-Instruct

Now that we have a downloaded model, let's check out the bash script.

.. code-block:: bash

    $ tune cp full_finetune_multinode .

.. literalinclude:: ../../../recipes/full_finetune_multinode.slurm

**There's a lot of information in this script but here are the high-level parts:**

* We utilize SLURM specific commands like number of nodes, tasks, CPUs available, etc.
* We are using `torchrun` and the `full_finetune_distributed` recipe to train just like on single node
* Should consider several cluster-specific environment variables

.. note::

    We may need to explicitly set the network interface for distributed backends. You can read more about that [here]
    but it's also helpful to know that you can find your network interface by running `ipconfig` from a specific node.
    You'll see the output.

Once we update the shared filesystem in the bash script, we can launch using sbatch.

.. code-block:: bash

    sbatch full_finetune_multinode.slurm

And the output of `squeue` should show our job running:

.. code-block:: bash

    $ squeue
    JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
    1     train         torchtun slurm R       0:03      2 slurm-worker-[1-2]

Once training has completed, we can follow the instructions [here] in order to upload our beautiful new model to the Hugging Face Hub!

Future development
------------------

We've covered the basics of how to launch a fine-tuning job with SLURM on two nodes with FSDP. There's still more things we're cooking up,
including...

**2D parallelism**: Utilizing both FSDP *and* tensor parallelism will decrease memory requirements even further, allowing us to lean even harder
into the advantages listed <above>.

**Longer context (ring attention, etc)**:

**Want other optimizations?** Feel free to let us know by opening up a Github Issue on our repo or dropping us a line in Discord!
