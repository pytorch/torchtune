.. _dpo_recipe_label:

====================================
Direct Preference Optimization
====================================

This recipe supports several `Direct Preference Optimization <https://arxiv.org/abs/2305.18290>`_ (DPO)-style fine-tuning techniques.
These techniques aim to steer (or `align <https://en.wikipedia.org/wiki/AI_alignment>`_) a model towards some desirable behaviours.
For example, a common goal is to train language models to produce safe and honest outputs,
or to be `helpful and harmless <https://arxiv.org/abs/2204.05862>`_.

To see the best results when using this recipe, it may be helpful to first fine-tune your model with using supervised fine-tuning to ensure your model is
on-distribution for the domain you're interested in. To do this, check out our other fine-tuning recipes in the :ref:`recipe overview <recipes_overview_label>` which
support a variety of SFT paradigms.

After supervised fine-tuning, here is an example of using either LoRA-based finetuning, or full-finetuning Llama 3.1 8B with DPO:

.. note::

    You may need to be granted access to the Llama model you're interested in. See
    :ref:`here <download_llama_label>` for details on accessing gated repositories.


.. code-block:: bash

    tune download meta-llama/Meta-Llama-3.1-8B-Instruct \
    --ignore-patterns "original/consolidated.00.pth"
    --HF_TOKEN <HF_TOKEN>

    # run lora dpo on a single device
    tune run lora_dpo_single_device --config llama3_1/8B_lora_dpo_single_device

    # run lora dpo on two gpus
    tune run --nproc_per_node 2 lora_dpo_distributed --config llama3_1/8B_lora_dpo

    # run full dpo on four gpus
    tune run --nproc_per_node 4 full_dpo_distributed --config llama3_1/8B_full_dpo

It's easy to get started with this recipe with your dataset of choice, including custom local datasets,
and datasets from Hugging Face. Check out our primer on :ref:`preference datasets <preference_dataset_usage_label>` to
see how to do this.

For this recipe we include different DPO-style losses:

* :class:`Direct Preference Optimization <torchtune.rlhf.loss.DPOLoss>` (DPO) loss [#]_. The DPO loss function
  increases the relative log-probabilities of preferred to un-preferred responses, whilst using log probabilities
  from a reference model to prevent policy degradation during training. Alongside RLHF, this is the most commonly used
  alignment technique and is used to train a growing number of state-of-the-art LLMs e.g. Llama3.1, Gemma 2, Qwen2, etc.
  This is a good starting point for alignment fine-tuning.
* :class:`Statistical Rejection Sampling Optimization <torchtune.rlhf.loss.RSOLoss>` (RSO) or "hinge" loss [#]_.
  RSO builds on concepts from support vector machines and DPO, applying a margin-based approach that penalizes
  low-quality responses while ensuring a significant gap between chosen and un-chosen log probabilities.

To use any of these, simply use the ``loss`` config entry or flag through the :ref:`cli_label`:

.. code-block:: bash

    tune run lora_dpo_single_device --config llama2/7B_lora_dpo_single_device \
    loss=torchtune.modules.loss.RSOLoss \
    gamma=0.5

Also, you can pass your custom loss in our recipe. Note that its `forward` method should align with the following signature:

.. code-block:: python

    def forward(self, policy_inputs: ChosenRejectedOutputs, reference_inputs: ChosenRejectedOutputs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ...

Here, `ChosenRejectedOutputs` is a dataclass obtained from `concatenated_forward``:

.. code-block:: python

  @dataclass
  class ChosenRejectedOutputs:
      chosen_logps: torch.Tensor
      rejected_logps: torch.Tensor
      chosen_logits: torch.Tensor
      rejected_logits: torch.Tensor

If this is not sufficient and you need to compute additional values from the logits, you can modify `concatenated_forward` directly. To do this, use `tune cp` to copy the desired recipe, and don’t forget to use your own dataclass!

Refer to the TRL library for reference implementations of the desired losses. In particular, you may find useful loss calculations in trainers.

For a deeper understanding of the different levers you can pull when using this recipe,
see our documentation for the different PEFT training paradigms we support:

* :ref:`glossary_lora`
* :ref:`glossary_qlora`
* :ref:`glossary_dora`

Many of our other memory optimization features can be used in this recipe. You can learn more about all of our memory optimization features in our :ref:`memory optimization overview<memory_optimization_overview_label>`.

.. rubric:: References:

.. [#] Rafailov, R., Sharma, A., Mitchell, E., Manning, C.D., Ermon, S. and Finn, C., 2024.
         Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36.
.. [#] Liu, T., Zhao, Y., Joshi, R., Khalman, M., Saleh, M., Liu, P.J. and Liu, J., 2023.
         Statistical rejection sampling improves preference optimization. arXiv preprint arXiv:2309.06657.
