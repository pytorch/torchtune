.. _preference_dataset_usage_label:

===================
Preference Datasets
===================


Preference datasets are used for reward modelling, where the downstream task is to fine-tune a base model
to capture some underlying human preferences. Currently, these datasets are used in torchtune with the
Direct Preference Optimization (DPO) `recipe <https://github.com/pytorch/torchtune/blob/main/recipes/lora_dpo_single_device.py>`_,
which is an alignment technique that aims to steer the behaviour of a pre-trained LLM towards human-preferable behaviour.

The ground-truth in preference datasets is usually the outcome of a binary comparison between two completions for the same prompt,
and where a human annotator has indicated that one completion is more preferable than the other, according to some pre-set criterion.
These prompt-completion pairs could be instruct style (single-turn, optionally with a single prompt), chat style (multi-turn), or
some other set of interactions between a user and model.


Preference dataset format
-------------------------

Preference datasets are expected to have two columns: "chosen", which indicates the human annotator's preferred response, and "rejected", indicating
the human annotator's dis-preferred response. Each of these columns should contain a list of messages with an identical prompt, followed by a list of messages.
The list of messages could include a system prompt, an instruction, multiple turns between user and assistant, or tool calls/returns. Let's take a look at
Anthropic's helpfulness/harmlessness `dataset <https://huggingface.co/datasets/RLHFlow/HH-RLHF-Helpful-standard>`_ as an example:
