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

We also support custom contrastive losses! But due to our philosophy related to the simplicity of the recipes, we do not support any of them directly in torchtune.
Instead, we provide a mechanism to make it possible to use a recipe with a custom loss without touching internals.

Here's how:

1. Introduce your loss in the following format: 

.. code-block:: python
  
  class SimPOLoss(nn.Module):
    """
    SimPO: Simple Preference Optimization with a Reference-Free Reward: https://arxiv.org/abs/2405.14734.
    Intuition from the paper:
        The effectiveness of SimPO is attributed to a key design: using the average log probability of a sequence as
        the implicit reward. Additionally, we introduce a target reward margin to the Bradley-Terry objective to
        encourage a larger margin between the winning and losing responses, further enhancing the algorithm's performance.
    Based on the TRL implementation:
    https://github.com/huggingface/trl/blob/98ad01ddfd1e1b67ec018014b83cba40e0caea66/trl/trainer/cpo_trainer.py#L603
    SimPO is pretty much identitcal to DPO but uses average logprobs to eliminate the need for a reference model to regularize
    the policy during training. It also uses a target reward margin to guide the policy towards better responses.
    This is kind of the same intuition as in :class:`~torchtune.rlhf.loss.IPOLoss`, but instead of optimizing against
    a margin between the reference policy and policy models, we're optimizing against a margin between the chosen and
    rejected responses.
    Args:
        beta (float): Equivalent temperature scaling parameter to DPO loss, typically in the range of 2.0 to 2.5. Default is 2.0.
        gamma (float): Target reward margin hyperparameter, typically we have ``gamma in (0, 1.5]``.
            Default is 0.5.
        label_smoothing (float): Parameter encoding uncertainty about the labels. Default is 0.
    """
    def __init__(
        self,
        beta: float = 2.0,
        gamma: float = 0.5,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the SimPO loss for a batch chosen and rejected average log probabilities.
        Args:
            policy_chosen_logps (torch.Tensor): Average log probabilities of the policy model
                for the chosen responses with shape [b,].
            policy_rejected_logps (torch.Tensor): Average log probabilities of the policy model
                for the rejected responses with shape [b,].
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]; A tuple of three tensors with shape [b,]:
                - losses: The SimPO loss for each example in the batch.
                - chosen_rewards: Rewards for the chosen responses.
                - rejected_rewards: Rewards for the rejected responses.
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        gamma_logratios = self.gamma / self.beta
        logits = pi_logratios - gamma_logratios
        losses = (
            -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
            - F.logsigmoid(-self.beta * logits) * self.label_smoothing
        )
        chosen_rewards = self.beta * (policy_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps).detach()
        return losses, chosen_rewards, rejected_rewards
    
    def concatenated_forward(
        self, model: nn.Module, batch: Tuple[torch.Tensor, torch.Tensor], _device, activations_handling_ctx
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run forward pass of the model with chosen and rejected samples concatenated.
        Args:
            model (nn.Module): The model to be used for the forward pass.
            batch (Tuple[torch.Tensor, torch.Tensor]): Tuple of input_ids and labels.
        Returns:
            Tuple of chosen log probs, rejected log probs, chosen logits, rejected logits.
        """
        concatenated_input_ids, concatenated_labels = batch
        concatenated_input_ids = concatenated_input_ids.to(_device)
        concatenated_labels = concatenated_labels.to(_device)
        # formed by concatenating an equal number of "chosen" and "rejected".
        len_chosen = concatenated_input_ids.shape[0] // 2
        with activations_handling_ctx:
            all_logits = model(concatenated_input_ids)
        all_log_probs = rlhf.get_batch_log_probs(all_logits, concatenated_labels)
        chosen_log_probs = all_log_probs[:len_chosen]
        rejected_log_probs = all_log_probs[len_chosen:]
        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]
        return (chosen_log_probs, rejected_log_probs, chosen_logits, rejected_logits)
2. Notice, that you need to provide both loss forward and concatenated_forward.
3. Create some module in your config directory with this loss, for instance `my_loss.py`
4. Finally, pass your custom loss through the config.

.. code-block:: yaml
  loss:
    _component_: my_loss.SimPOLoss

5. In the most cases you don't need reference logprobs, so you can disable calculation of them, through:

.. code-block:: yaml
  reference_model: false

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
