# E2E User Flow

This design doc attempts to describe the end-to-end user flow for the "hobbyist"
user for alpha release up to MVP. This **does not** seek to cover the
"hero" user, nor does it attempt to cover every possible use case. In addition, there
are many great features that we should think about post-MVP. These features will not be
discussed here, but as always, people are encouraged to open detailed Github Issues for
others to reference later.

## Prologue: User hears about Torchtune

There are a few channels that make sense to direct users towards Torchtune: blogs,
X, the PyTorch website, and word-of-mouth. This is out of scope for this RFC and more in
line w/ a GTM strategy. Simply mentioning it here to capture the entire scope of a
user journey.

***

> I'm adding a checkbox for each of these individual items indicating
completeness.

## 1) User meets Torchtune

We want to create a seamless and unified user onboarding experience. While we cannot control
third-party blogs or forks of our repo, we have two ways in which users can get started
with Torchtune: Github and the PyTorch docs. While our Github README should be incredibly clear
and include minimal quickstart information b/c **inevitably users will come to our Github page directly**,
we should direct users to the PyTorch docs as the source of truth. That way, we aren't rigourously
attempting to keep two surfaces updated. See how [PyTorch does this](https://github.com/pytorch/pytorch).

### 1a) Github README Proposal
To be concrete, I propose the following sections to be included in our Github:
* [ ] Introduction
    - **Description**: What is Torchtune in 1-2 lines?
* [ ] Why Torchtune?
    - **Description**: What are the selling points of Torchtune > Axolotl, Lit-GPT, etc?
    - **Reasoning**: There exists several finetuning LLM frameworks. Why are we better? We
    get some clout by being PyTorch affiliated but what can we quickly communicate
    to potential users about why they should switch from their e.g. Axolotl setup to us?
* [ ] Quickstart:
    - **Description**: Minimal command to getting the package `pip` installed and running a **LoRA finetune**,
    include a link to our more comprehensive "Getting Started" tutorial on the official docs page.
    - **Inspo**: [Axolotl's Quickstart Guide](https://github.com/OpenAccess-AI-Collective/axolotl?tab=readme-ov-file#quickstart-)
* [ ] Supported Models & Datasets
    - **Description**: Crystal clear communication of which models + datasets they can pick up and run with OOTB
    - **Inspo**: [Lit-GPT](https://github.com/Lightning-AI/lit-gpt/tree/main?tab=readme-ov-file#-lit-gpt-1), [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/tree/main?tab=readme-ov-file#axolotl-supports), HF
    - **Reasoning**: People want to use the best models possible and will be checking this as one of the first things to see what they can
    accomplish with minimal effort
* [ ] Debugging & Getting Help
    - **Description**: Directions to our Discord channel, link to an OOM/Memory management tutorial, link to a selecting model + datasets tutorial,
    link to common errors page [SEE 1b for more details on these tutorials]
    - **Reasoning**: We don't want to slam users w/ debugging information on the Github page, rather just link to comprehensive
    docs
* [ ] Contributing
    - Full-fledged `CONTRIBUTING.md` guide. Deciding the contents of this guide are out-of-scope for this RFC.
* [ ] Citation
    - Reasoning: Credit :)
* [ ] License

### 1b) Official Torchtune Docs Proposal
Our official documentation on the PyTorch website has to be held to an even higher bar. @NicolasHug has done
a ton of great work to get us off the ground in that respect. In order for us to be ready for the alpha + MVP,
I propose the following:

* Entry Page
    - [ ]Core concepts of Torchtune
        - Extensible configs
        - ...
    - [ ]Torchtune's advantages in the OSS community

* API Reference
    - [ ] Every single public class and method properly documented (lowest bar) At the time of writing this RFC,
    not every single public class and method **is** documented and rendered.

* Recipes
    - [ ] Full finetune recipe
    - [ ] LoRA finetune recipe
    - [ ] Generate recipe

* Tutorials
    - [ ] Getting started akin to ExecuTorch's [Getting Started](https://pytorch.org/executorch/stable/getting-started-setup.html)
    - [ ] OOM/Memory Management tutorial akin to [Lit-GPT's OOM Guide](https://github.com/Lightning-AI/lit-gpt/blob/main/tutorials/oom.md)
    - [ ] How to choose the things for your HW + arch, idea from @lrouesnel's post in Workplace
    - [ ] Common Errors akin to [HF's Troubleshooting Guide](https://huggingface.co/docs/transformers/troubleshooting)

***

## 2) User finetunes a model

By now the user has been able to read through some or all of our docs on Github + our official documentation.

### 2a) User downloads the model

The primary model we will support out of the box is Llama2. There are currently two ways to get Llama2 weights.

The first of these is to:
1. Navigate to the [official page](https://ai.meta.com/llama/)
2. Accept the terms and conditions
3. Wait for approval (takes <10 mins for a Meta email, which is a big, big caveat. I am trying to confirm how long it takes w/ some fake affiliations, etc.)
4. Receive approval in the form of an email from Meta AI
5. Check your email from a message from Meta w/ further instructions
6. Visit the [Llama repo](https://github.com/facebookresearch/llama)
7. Checkout the `download.sh` script
8. Enter unique custom URL
9. Select which model weights to download
10. Profit? Take a nap?

The second option is the same up til step 4. Then the user would do the following:

5. Navigate to Huggingface's [meta-llama](https://huggingface.co/meta-llama/Llama-2-7b-hf) page
6. Request access (takes <5 seconds if you've already been approved)
7. Download weights how you would normally with Huggingface

While both have their limitations, **aligning ourselves with HuggingFace for model downloading is more aligned with our future goals**
re: downloading other models like Mistral, whose official release is on Huggingface.

Additional roadblocks to this approach: The user will have to have `transformers` installed to download the weights.

Last thought here: The conversion script will allow a user to specify an output dir, but we set the default and propogate this default
through to our configs so that downstream scripts run OOTB. See [Lit-GPT]'s setup for this seamless start.

My ideal command looks like the following:

```python scripts/download_and_convert_checkpoints.py --model meta-llama/llama2-7b-hf --output-dir /tmp/llama2-7b```

### 2b) User launches a finetune

At this point, we've directed the user to checkout our first finetuning script, which will be
**LoRA + FSDP on an Alpaca-style dataset**.

I've started creating Github Issues with "[Alpaca comparison]" detailing what we need to reach parity.
For now those include:
- [ ] [Gradient accumulation](https://github.com/pytorch-labs/torchtune/issues/252)
- [ ] [LR scheduling](https://github.com/pytorch-labs/torchtune/issues/242)

They've looked through the recipe and the config trying
to understand what knobs they can tweak.

- [ ] Where do we document a core recipe's capabilities?
- [ ] Where do we document all the possible options for a config?

The above are likely going to be answered in [Rafi's Config RFC](https://github.com/pytorch-labs/torchtune/pull/230),
but I'm leaving them here for reference.

Ideal command:

```tune --nnodes 1 --nproc_per_node 4 lora_finetune --config llama2_lora_finetune.yml```

### 2c) User hits an error and debugs

Starting now, we will keep track of any errors or complications that come up when a "new" user
(e.g. Laurence, Matt, etc.) tries to do a clean install and running of the default recipe.
These will go into the common errors documentation section above. These should be tagged with "startup error"
tag in Github Issues.

### 2d) User re-starts a finetune

For the scenario in which a user starts a finetune, gets 75% of the way through and then runs into a
hardware issue, they should be able to restart from the latest checkpoint, specified by an interval,
with proper "data rewinded". The following open PR is the last step before we are able to hit that
goal.
- [ ] [Dataloader, epoch, seed checkpoint integration + full determinism resume](https://github.com/pytorch-labs/torchtune/pull/247)

***

## 3) User evaluates a model

### 3a) User tries a domain-specific prompt on newly fine-tuned model

The user will be able to take the model they just finetuned and take it out for a spin! It should be fun and easy. To start, we can
utilize the generate recipe. In a follow-up, we can enable an integration with [Gradio](https://www.gradio.app/).

```tune generate --model lora_llama2 --checkpoint <NEW-CKPT> --prompt "Which apple is best: Honeycrisp or Gala?"```

### 3b) User runs automated suite of benchmarks

This is the best indicator of whether or not a freshly finetuned model is any good. User will be able to choose to evaluate
against [Eluther's Eval Harness](https://github.com/EleutherAI/lm-evaluation-harness).
- [ ] Integration w/ Eluther Eval

```tune evaluate --model lora_llama2 --checkpoint <NEW-CKPT> --tests bigbench hellaswag```

***

## 4) User publishes a model

Part of the "viral" loop that we want to encourage is the publishing of cool new models. As such,
we want to have a "upload to HuggingFace hub" script. The script should have the following capabilities:
- [ ] Modifiable model card w/ datsets trained on & default license
- [ ] "Trained w/ Torchtune" svg similar to [Axolotl's](https://github.com/OpenAccess-AI-Collective/axolotl?tab=readme-ov-file#badge-%EF%B8%8F)

```tune push_to_hub --config lora_llama2_final.yml --hf-auth-token ABCDEFG```

Future iterations could do a conversion to [safetensors](https://huggingface.co/docs/safetensors/index), which would open us up to participating in Open LLM
competitions and creater extensibility with HuggingFace. There's already an API reference for how to do this [here](https://huggingface.co/docs/safetensors/api/torch).

***

## Epilogue: User comes back with friends
