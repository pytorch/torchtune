## [RFC] Configuring low precision and QLoRAtraining in TorchTune

This is meant to be a brief note on how we can configure low precision training for both single and multi-device recipes in TorchTune. 

#### TL;DR
- Single device (both full finetune and LoRA) and multi device will support a flag, `dtype`, that can be either [bf16, fp32] to configure low precision training. This will initialize the model in the lower precision, so all parameters, activations, gradients, and opt. states will be in this precision to optimize memory savings. We will not enable torch.autocast.
- We will actively de-invest in fp16 training since most recent HW has support for bf16. In terms of consumer HW arches we'd like to support, 4090, 3090, A6000 support bf16, only T4 does not support bf16. This means in particular we won't have a memory efficient ( < 16GB) finetuning solution that runs reliably on T4's (but we will for the other  mentioned consumer GPUs).
- QLoRA MVP offering in TorchTune will be scoped to quantization of parameters in linear layers that we're applying LoRA on to 4 bits, and compute in bf16. We won't in particular enable QLoRA with a fp32 computation type.
- Quantizing to 4 bits will be coupled to using LoRA and training in bf16 as this is convention and allows us to most closely replicate baselines / papers. There are no technical blockers preventing us from extending this as needed in the future.
- We will enable QLoRA with a `quantize_base` flag that necesitates `dtype=torch.bfloat16`. This in particular means we won't support QLoRA on HW types that dont' support bf16. If this support is needed, we can extend QLoRA to support fp32 compute dtype.

##### Overview of low/mixed precision training

PyTorch offers a variety of ways to conduct reduced precision training using existing APIs:
- [torch.autocast](https://pytorch.org/docs/stable/amp.html), where model exists in full precision, and parameters and activations are dynamically cast to lower precision at dispatcher level based on an allow/deny list. This has potential to speed up training by running some ops that aren't as precision sensitive in lower precision, but does not save memory (in fact there have been some reports of increased memory when using this) since parameters/gradients/activations are not stored in the lower precision format (the lower precision is only used for compute).

- "full" low precision training (for lack of a better name) is where the entire model's parameters are moved to a lower precision format such as fp16 or bf16, via i.e. `model.to(torch.bfloat16)`. This will make all compute (fwd/backward) run in lower precision, activations, gradients, optimizer states will be in this lower precision. Historically, making this work for fp16 has been challenging even with gradient scaling etc, but bf16 low precision training is becoming increasingly popular and several LLMs, including llama2, have been successfully trained w/full bf16. This also has a large memory saving property, since all parameters, gradients, activations, and optimizer states are held in bf16.

- Mixed precision policies in parallelism APIs such as [FSDP's mixed precision](https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.MixedPrecision). This is somewhat of an "in between" of the above two options - in particular, parameters are stored in low precision, forward and backward compute happens in low precision, but gradients are upcast back to fp32 before optimizer step has run. This is because [historically](https://arxiv.org/abs/1710.03740) it's been discovered that optimizer states and optimizer step is the most sensitive to precision, so the tradeoff was made to run optimizer passes in fp32, but fwd/backward in the lower precision. This also has a memory saving property in that parameters and activations are held in the lower precision.


#### "Full" low precision training

In this section, I make the case to focus on (1) only "full" low precision training for the single device LLM finetuning use case, and (2) "full" low precision + FSDP mixed precision for the multi-device use case:

 
- In single device case, "full" low precision training offers significant memory improvements over torch.autocast (roughly halving the memory requirement for training). Since we're building for memory constrained, GPU-poor use cases, one of the most important properties we care about is memory efficiency gains enabled by lower precision training, and "full" low precision offers the most memory savings (the other most important property is correctness, which @kartikayk has filed an issue for: https://github.com/pytorch/torchtune/issues/467). The only potential advantage of torch.autocast is that it could automatically run some ops in lower precision via a static denylist. If we have operators in our training that are especially precision sensitive, this could be an advantage. However,  this is not a strong pro for me at the moment because (1) LLMs have been able to train well even in full bf16 (save for some issues around loss spikes), and (2) we haven't observed any discernable convergence issues in torchtune (see https://github.com/pytorch/torchtune/pull/391 which has similar convergence for bf16 vs fp32). There are also tools we can use / build to mitigate potential loss spikes (briefly discussed blow).
- In multi-device case, FSDP mixed precision offers memory savings over torch.autocast. The ops FSDP mixed precision runs in lower precision is a superset of those run by torch.autocast. Similar to single device, the only potential advantage of torch.autocast is that it could automatically run some ops in lower precision via a static denylist. In addition to the reasons given above, I don't think this is an especially strong argument since FSDP provides workarounds where users can specify to run certain submodules in fp32 if problematic operators are found.

##### A wrench: QLoRA
- We're building QLoRA in torchtune (https://github.com/pytorch/torchtune/pull/478), which holds parameters in 4-bit precision but runs computation in bf16. In this case, parameters are in 4 bit, but gradients, activations, and optimizer states are 16 bit, and computation occurs in 16 bit. 
- Holding these parameters in 4 bit will also be coupled to LoRA in torchtune's MVP / initial offering (this isn't strictly necessary, but it is not common to 4-bit quantize outside of LoRA technique, and the convergence of such training has not been well studied).
- *Our overall goal* for initial offering of QLoRA in TorchTune's MVP is to enable 4-bit quantization of base model parameters in linear layers that we're applying LoRA to (the qkv projections in attention, the output proj in attention, MLPs, and the overall output proj (but see caveat below), and run computation in (strictly) bf16. This can be relaxed as needed post-MVP, but is the minimum set of functionality needed to be at parity with other popular finetuning repositories and replicate the results in lightning AI's blog: https://lightning.ai/pages/community/lora-insights/ (which shows eval results for applying LoRA to all linear layers and running in bf16). 

#### Proposal: configuring low precision training in TorchTune

##### Single-device full finetune use case
- For the single device use case, *full finetuning* will expose a single configuration flag, `dtype`, that controls the `dtype` the model will be initialized in / converted to, and we will do "full" low precision training in this dtype. 
- The ONLY dtypes we'll allow are are [bf16, fp32]. We'll disallow fp16 for now due to issues with convergence. Alternatively, we could allow fp16 and log warnings that the convergence has not been tested (I would not advocate for this).
- Since we're prioritizing bf16 and fp32, we won't enable gradient scalers as they're not needed for fp32/bf16 training.

##### Single-device LoRA finetune use case
- The main complexity introduced over single device full finetune is the additional possibilty of running QLoRA.
- In TorchTune's initial MVP, we'll couple QLoRA to bf16 training since [the paper](https://arxiv.org/abs/2305.14314) and lit-gpt also necesitate this, and QLoRA with fp32 does not help memory efficiency and is not well studied. One tradeoff we make here is that we likely won't be able to finetune in < 16GB on T4's, as T4's do not support bf16 dtype. We could enable QLoRA w/fp32 as a fast follow on post MVP.
- We'll also only allow applying 4-bit quantization to layers LoRA is being applied on. This again isn't strictly necessary, but is the convention across papers & well-known repos. 
- My suggestion for configuration here would be keeping the `dtype` flag and allowing [fp32, bf16]. We then add another boolean flag, `quantize_base` to enable QLoRA. If `quantize_base=True`, we enforce `dtype` as bf16. We also add checks to ensure bf16 dtype is available and supported for the HW that we're training on.

##### Multi-device use case
- This is similar to the single device use case, except we can allow user to configure FSDP mixed precision. My take here is that we should only enable configuring FSDP mixed precision if `dtype` is fp32, otherwise FSDP mixed precision does not make sense / provide any additional value, in terms of perf/correctness/memory saving, if we're already running compute in a lower precision. This in particular means that for full bf16 training or QLoRA, we won't allow user to configure FSDP mixed precision (and throw an error explaining why).

