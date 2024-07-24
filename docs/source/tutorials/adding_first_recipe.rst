.. _custom_recipe_label:

==============================
Implementing Your First Recipe
==============================

**Author:** `Salman Mohammadi <https://github.com/SalmanMohammadi>`_

This guide will walk you through the process of implementing your first custom recipe in torchtune by following along as we implement a recipe for fine-tuning
an LLM for a preference modelling task.

.. grid:: 2

    .. grid-item-card:: :octicon:`mortar-board;1em;` Key learning points:

      * A crash course on reward modelling
      *
      * How to utilise torchtune's optimisation features for making your recipe fast and memory efficient
      * Tips and tricks for debugging your recipes

    .. grid-item-card:: :octicon:`list-unordered;1em;` Prerequisites:

      * Be familiar with the :ref:`overview of torchtune<overview_label>`
      * Get an overall feel for how recipes work :ref:`in our recipe deepdive<recipe_deepdive>`
      * Be familiar with :ref:`configuring datasets<dataset_tutorial_label>`
      * Make sure to :ref:`install torchtune<install_label>`


Preference modelling
--------------------

Reinforcement learning from human feedback (RLHF) is a vital step in fine-tuning language models to be `helpful and harmless <https://arxiv.org/abs/2204.05862>`.
Consider the diagram below `from OpenAI's InstructGPT <https://openai.com/index/instruction-following/>`; the RLHF procedure involves three steps:

.. image:: /_static/img/rlhf_diagram.png

#. 1) Fine-tune a base model for your task (e.g. chat, instruct, etc.).
#. 2) Using the fine-tuned model from step 1., train a reward model on a preference dataset.
#. 3) Train using RLHF with the fine tuned model from step 1., and the trained reward model from step 2.

We're interested in 2. - training a reward model on a preference dataset. Reward models are a cruical component in RLHF; they are responsible for
accurately modelling the preferences learned from their training data to ensure our final model is correctly aligned. The procedure outlined above is
common for chat or instruct models, where an initial supervised fine-tuning step can help bring models on-distribution.

The recipe template
-------------------


Setting things up
-----------------

The core training loop
----------------------

Starting small, iterating fast
------------------------------

Scaling up; going fast
----------------------

Next steps
----------
