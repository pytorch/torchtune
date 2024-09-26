.. _llama_kd_label:

============================
Distilling Llama3 8B into 1B
============================

This guide will teach you about knowledge distillation (KD) and show you how you can use torchtune to distill a Llama3.1 8B model into Llama3.2 1B.
If you already know what knowledge distillation is and want to get straight to running your own distillation in torchtune,
you can jump to knowledge distillation recipe in torchtune, `knowledge_distillation_single_device.py <https://github.com/pytorch/torchtune/blob/main/recipes/knowledge_distillation_single_device.py>`_.

.. grid:: 2

    .. grid-item-card:: :octicon:`mortar-board;1em;` What you will learn

      * What KD is and how it can help improve model performance
      * An overview of KD components in torchtune
      * How to distill from a teacher to student model using torchtune
      * How to experiment with different KD configurations

    .. grid-item-card:: :octicon:`list-unordered;1em;` Prerequisites

      * Be familiar with :ref:`torchtune<overview_label>`
      * Make sure to :ref:`install torchtune<install_label>`
      * Make sure you have downloaded the :ref:`Llama3 model weights<download_llama_label>`
      * Be familiar with :ref:`LoRA<lora_finetune_label>`

What is Knowledge Distillation?
-------------------------------

`Knowledge Distillation <https://arxiv.org/pdf/1503.02531>`_ is is a widely used compression technique
that transfers knowledge from a larger (teacher) model to a smaller (student) model. Larger models have
more parameters and capacity for knowledge, however, this larger capacity is also more computationally
expensive to deploy. Knowledge distillation can be used to compress the knowledge of a larger model into
a smaller model. The idea is that performance of smaller models can be improved by learning from larger
model's outputs.

How does Knowledge Distillation work?
-------------------------------------

Knowledge is transferred from the teacher to student model by training it on a transfer set and where the
student is trained to imitate the token-level probability distributions of the teacher. The diagram below
is a simplified representation of how KD works.

.. image:: /_static/img/kd-simplified.png
