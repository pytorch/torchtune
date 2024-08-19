.. _dpo_recipe_label:

====================================
Direct Preference Optimization
====================================

This recipe supports several `Direct Preference Optimization <https://arxiv.org/abs/2305.18290>`_ (DPO)-style fine-tuning techniques.

The core principle of DPO retains similarities to PPO (https://arxiv.org/abs/2009.01325), where it optimizes a policy
(language) model to align with human preferences, and regularizes the loss function using a baseline
reference (the frozen, initial language model) to prevent over-fitting to the preference dataset.
It differs from PPO by optimizing the policy model directly using labelled preference data, rather
than using an additional reward model to provide feedback.
This significantly simplifies training and reduces compute overhead.

The best way to get started with our recipes is through the :ref:`cli_label`, which allows you to
list all our recipes and configs, run recipes, copy configs and recipes, and validate configs
without touching a line of code!

For example, if you're interested in using this recipe with the latest `Llama models <https://llama.meta.com/>`_, you can fine-tune
in just two steps:


.. note::

    You may need to be granted access to the Llama model you're interested in. See
    :ref:`here <download_llama_label>` for details on accessing gated repositories.


.. code-block:: bash

    tune download meta-llama/Llama-2-7b-hf \
    --output-dir /tmp/Llama-2-7b-hf \
    --HF_TOKEN <HF_TOKEN>

    tune run lora_dpo_single_device --config llama2/7B_lora_dpo_single_device



Most of you will want to twist, pull, and bop all the different levers, buttons, and knobs we expose in our recipes. Check out our
:ref:`configs tutorial <config_tutorial_label>` to learn how to customize recipes to suit your needs.

For this recipe in particular, we include several different DPO-style losses, which  use :class:`preference datasets <torchtune.datasets.PreferenceDataset>` to
align language models:

* :class:`Direct Preference Optimization <torchtune.modules.rlhf.loss.DPOLoss>` (DPO) loss [1]_. The DPO loss function
  increases the relative log-probabilities of preferred to un-preferred responses, whilst using log probabilities
  from a reference model to prevent policy degradation during training. Alongside RLHF, this is the most commonly used
  alignment technique and is used to train a growing number of state-of-the-art LLMs e.g. Llama3.1, Gemma 2, Qwen2, etc.
  This is a good starting point for alignment fine-tuning.
* :class:`Statistical Rejection Sampling Optimization <torchtune.modules.rlhf.loss.RSOLoss>` (RSO) or "hinge" loss [2]_.
  RSO builds on concepts from support vector machines and DPO, applying a margin-based approach that penalizes
  low-quality responses while ensuring a significant gap between chosen and un-chosen log probabilities.
* :class:`Identity Preference Optimization <torchtune.modules.rlhf.loss.IPOLoss>` (IPO) loss [3]_. The IPO loss function
  maintains the identity of the model's original preferences by directly optimizing the difference between preferred and
  un-preferred responses. IPO focuses on preserving the intrinsic qualities of the base model while aligning
  it with user preferences.
* :class:`Simple Preference Optimization <torchtune.modules.rlhf.loss.SimPOLoss>` (SimPO) loss [4]_. SimPO simplifies
  preference optimization by using the average log probability of responses as an implicit reward,
  eliminating the need for a reference model, and also introduces a target reward margin to encourage a
  clear distinction between preferred and non-preferred outputs. Using SimPO should result in slightly
  faster training than other preference optimization techniques.

To use any of these, simply use the ``loss`` config entry or flag:

.. code-block:: bash

    tune run lora_dpo_single_device --config llama2/7B_lora_dpo_single_device \
    loss=torchtune.modules.loss.SimPOLoss \
    beta=2.0 \
    gamma=0.5


This recipe is also an example of parameter efficient fine-tuning (PEFT). To understand the different
levers you can pull, see our documentation for the different PEFT training paradigms we support:

.. * :ref:`glossary_lora`
.. * :ref:`glossary_qlora`.

.. As with all of our recipes, you can also:

.. * Adjust :ref:`model precision <glossary_precision>`.
.. * Use :ref:`activation checkpointing <glossary_act_ckpt>`.
.. * Enable :ref:`gradient accumulation <glossary_grad_accm>`.
.. * Use :ref:`lower precision optimizers <glossary_low_precision_opt>`.
..   However, note that since LoRA significantly reduces memory usage due to gradient state, you will likely not need this
..   feature.

.. .. and for distributed recipes

.. .. As with all our distributed recipes:

.. .. * `glossary_distrib`


.. If you're interested in an overview of our memory optimization features, check out our  :ref:`memory optimization overview<memory_optimization_overview_label>`!

.. [1] Rafailov, R., Sharma, A., Mitchell, E., Manning, C.D., Ermon, S. and Finn, C., 2024. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36.
.. [2] Liu, T., Zhao, Y., Joshi, R., Khalman, M., Saleh, M., Liu, P.J. and Liu, J., 2023. Statistical rejection sampling improves preference optimization. arXiv preprint arXiv:2309.06657.
.. [3] Azar, M.G., Guo, Z.D., Piot, B., Munos, R., Rowland, M., Valko, M. and Calandriello, D., 2024, April. A general theoretical paradigm to understand learning from human preferences. In International Conference on Artificial Intelligence and Statistics (pp. 4447-4455). PMLR.
.. [4] Meng, Y., Xia, M. and Chen, D., 2024. Simpo: Simple preference optimization with a reference-free reward. arXiv preprint arXiv:2405.14734.
