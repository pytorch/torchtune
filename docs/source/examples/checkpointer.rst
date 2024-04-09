.. _understand_checkpointer:

==============================
Understanding the Checkpointer
==============================

This tutorial will walk you through the design and behavior of the checkpointer and the associated
utilities.

.. grid:: 1

    .. grid-item-card:: :octicon:`mortar-board;1em;` What you will learn

      * Deep-dive into the different checkpointers and how to correctly set them
      * Using the right checkpointer with the right model source
      * Differences between intermediate checkpoints and final checkpoints
      * Differences between checkpointing for full-finetune and LoRA


Overview
--------

TorchTune checkpointers are designed to be composable components which can be plugged
into any training recipe. Each checkpointer supports a specific set of models and training
scenarios making these easy to understand, debug and extend.

TorchTune is designed to be "state-dict invariant". This means the checkpointer
ensures that the output checkpoint has the same format as the source checkpoint i.e.
the output checkpoints have the same keys split across the same number of files as the original
checkpoint. Being "state-dict invariant" allows users to seamlessly use TorchTune checkpoints
with their favorite post-training tools from the open-source ecosystem without writing
TorchTune-specific convertors. To be "state-dict invariant", the ``load_checkpoint`` and
``save_checkpoint`` methods make use of the weight convertors available in
``torchtune/models/<model_folder>``.
