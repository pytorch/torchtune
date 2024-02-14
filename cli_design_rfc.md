# CLI Design RFC

## Outline
* Importance of a good CLI tool
* Current problems w/ CLI tool
* Addressing Immediate Feedback
* MVP Proposal (The Vision)
* Research

***

## Importance of a good CLI tool

There are two high-level ways a user can interact with TorchTune:
1. Import components from TorchTune
2. Run recipes w/ tune CLI

Option 1 will be reserved for those users we consider "Heroes" - essentially advanced users.
They come with their own potentially complex setup for experimentation and launching training runs.
They will have minimal to zero interaction with the `tune` CLI.

Option 2, on the other hand, applies to our "Hobbyist" and "Hacker" user types. They will
**need** to use the `tune` CLI.

> See [here](https://github.com/pytorch-labs/torchtune/issues/53) for more information about these user types

A quote from Laurence on how we should think about positioning
our CLI: "Hey when you're getting started, just use the CLI tool, it's super quick and easy". Regardless
of which user someone ends up being, the way they start will almost always be the `tune CLI`.

## Current problems with our CLI tool

We did a limited release internally, and also collected targeted feedback from some
people close to the project: Laurence & Matt. We directed them to [our documentation](https://deploy-preview-368--torchtune-preview.netlify.app/examples/first_finetune_tutorial)
and had them go from overview -> install -> first finetune. Some of Matt's direct feedback
can be found in our [Issues](https://github.com/pytorch-labs/torchtune/issues/372), but other feedback was delivered via video recording and in person. Please ping me for access to the former.
While they had plenty of feedback on the whole onboarding experience, this RFC will focus on the specific feedback for the CLI.

1. Confusion about command to run a recipe
    - Unfamiliarity w/ torchrun commands. What does `--nproc_per_node` mean?
    - "I don't use torchrun regularly and `accelerate` just handles this for me"
2. Confusion about recipe + config manipulation
3. Confusion about total capabilities of the tool
    - "Can I do XYZ? Where can I see all that the CLI can do?"

## Addressing Immediate Feedback

In addressing the above feedback, I also did extensive research on current best practices for building a CLI tool,
read through other engineer's efforts to build a CLI tool, and also examined CLI tools I liked and tried to determine
what exactly they did well. This can all be found in the "Research" section of this RFC (at the bottom).

### 1. Confusion about running a recipe

We made the [decision](https://github.com/pytorch-labs/torchtune/pull/138) to "shadow" the `torchrun` CLI command so that we were quickly able to launch multi-node training using an existing framework. This decision allowed us to
validate our training and get both LoRA and FFT landed. Unfortunately, without prior knowledge of `torchrun` and/or extensive documentation, launching multi-process training is confusing, unintuitive. The user should be able to specify
a number of GPUs that they want to use and not have to worry about what exactly is launching the distributed training
under the hood.

#### Proposed Change

Key Points:
- Renaming to `run` reflects "running" any recipe
- `num_gpus` specification is easy to understand and mirrors other popular framworks like [Ray](https://docs.ray.io/en/latest/cluster/kubernetes/user-guides/gpu.html#gpus-and-ray)

```
$ tune run --help
Run a recipe

USAGE
    $ tune run RECIPE [OPTIONS]

OPTIONS
    --config        Specify a config file for a given recipe
    -h, --help      Show help message and exit

EXAMPLES
    $ tune run full_finetune.py --config alpaca_llama2_full_finetune.yaml
    Starting finetuning run...

A user can also override any config option for a given recipe
through the command line, e.g. `$ tune run example.py --config example_config.yaml --num_gpus 4`

Take a look at the recipes documentation page for all possible overrides: <LINK>
or for a list of all builtin recipes/configs you can run:
    $ tune ls
```

**How does this work?**
Instead of calling `torchrun` via the entrypoint script laid out [here](https://github.com/pytorch/pytorch/blob/main/torch/distributed/run.py), we translate `num_gpus` and any other
distributed params to [`elastic_launch`](https://github.com/pytorch/pytorch/blob/main/torch/distributed/launcher/api.py#L101), which `torchrun` is calling under the hood and which we already use in [our tests](https://github.com/search?q=repo%3Apytorch-labs%2Ftorchtune%20elastic_launch&type=code).
- Do we lose any functionality? No, this is the same underlying process being called.
- How will we specify multi-process on CPU? (This was previously done by specifying `--nproc_per_node X` on a
non-GPU backend)
Open to ideas here, but so far I see very little need for flexibility here.
My honest idea would be to get the max cores on the machine and utilize all of them if someone is running
training on CPU.
On that note, we don't actuallysay we fully support running on CPU (not until CPUOffload is fixed) so
it's a bit of a null-issue.
- Will we expose *all* options for `torchrun` via the CLI if run is specified? No, we can decide on a core set of
features we want to expose to start and address other concerns as they come up. "It's easier to add a feature than to
claw one back that's been already open to users" - Nicolas Hug.
- Is hiding this functionality leading us too close to the "HuggingFace magic" route? To be fair, this is a tight line
to walk. For the purposes of this utility, being upfront that our underlying code is just `torchrun` and directing them
to those docs is less obfuscation than found in HuggingFace. At the end of the day; however, we want to understand what
knowledge our users may have and meet them where they are at. If `num_gpus` makes more sense than `nproc_per_node`, I
will opt for that ever time. We shouldn't make them have to learn new terminology unless absolutely necessary.

**Unsolved UX Issues**
- How to easily show users what config options are available for a given recipe. I consider this to be
out of scope for this RFC, but needs to be addressed. Trial and error cannot be the only course of action
available.

### 2. Recipe + config manipulation

This was mentioned less specifically in feedback, but is highly relevant to discussions we're having
today in the TorchTune Core team - See #364 and #362.

**The problem**: It is not clear how users can discover or interact with built-in recipes, and when they
go to modify a recipe or config, it's not clear how to interact with it using the CLI tool.

#### Proposed Change

Key Points:
- This lists **only** the built-in recipes and configs, meant only to get
the user off the ground and able to modify
- Pairs recipes visually with their available configs

```
$ tune ls --help
List all built-in recipes or configs

USAGE
    $ tune ls

EXAMPLES
    $ tune ls
    RECIPE                  CONFIGS
    full_finetune.py        alpaca_llama2_full_finetune.yaml
                            alpaca_mistral_full_finetune.yaml
    lora_finetune.py        alpaca_llama2_lora_finetune.yaml
    alpaca_generate.py      <>

To get started you can run:
$ tune run full_finetune.py --config alpaca_llama2_full_finetune.yaml
```


Key Points
- User still needs some way to copy a config or recipe into their local dir in
order to modify

```
$ tune cp --help
Copy a recipe or config to local

USAGE
    $ tune cp RECIPE/CONFIG [OPTIONS]

OPTIONS
    --to            Where to copy the file to
    -h, --help      Print help message and exit

EXAMPLES
    # Copy a recipe
    $ tune cp full_finetune.py --to my_project/full_finetune.py

    # Copy a config
    $ tune cp alpaca_llama2_full_finetune.yaml --to .

Need to see all possible configs to copy? Try running `tune ls`
```

### 3. Confusion about total capabilities of the tool

This one requires minimal functionality changes to the CLI tool itself. Instead it necessitates
extensive documentation both within the CLI tool itself in the form of `--help` commands, examples, and in
the live docs.

From the CLIG: Guidelines:

"Provide web-based documentation. People need to be able to search online for your tool’s documentation, and to link other people to specific parts. The web is the most inclusive documentation format available.

Provide terminal-based documentation. Documentation in the terminal has several nice properties: it’s fast to access, it stays in sync with the specific installed version of the tool, and it works without an internet connection."

This informatino was simply missing from our Alpha launch and has to be better moving forward.

***

## MVP Proposal (The Vision)

**tl;dr**: Our CLI should be:
1. **Comprehensive**: A user should be able to do all the *basic* functionality without leaving the CLI
2. **Intuitive, Human First**: A user should understand how to use the CLI based on their other experiences
with CLI tools and should just "make sense"
3. **Consistent**: All actions and grouping is the same across our CLI commands and options

This is a **starting point** for the MVP. The most important feedback will be that which we
get from our first users. Based on their experience, we will adapt the CLI. This proposal
gives us a foundation upon which users can experiment easily.

```
$ tune
tune: finetune your local LLM with ease

USAGE
    $ tune [COMMAND] ...

COMMANDS
    ls                  List all built-in recipes/configs.
    cp                  Copy a built-in recipe or config to a local path.
    download            Download a base large language model from the HuggingFace Hub.
    convert-ckpt        Convert a model checkpoint into a format supported by TorchTune.
    run                 Run a recipe e.g. finetune a model, evaluate a model.
    upload              Upload a model to the HuggingFace Hub.
    share               Share an idempotent representation of a finetuning job.

OPTIONS
    -h, --help          Show help message and exit
    -V, --version       Show version number and exit.

EXAMPLES
    # Download a Llama2 model
    $ tune download llama2
    Fetching 10 files: 100%|████████████████████| 10/10 [00:00<00:00, 100.72it/s]
    Succesfully downloaded model repo and wrote to the following locations:
    /tmp/model/.gitattributes
    /tmp/model/LICENSE.txt
    /tmp/model/Responsible-Use-Guide.pdf
    /tmp/model/README.md
    /tmp/model/checklist.chk
    /tmp/model/params.json
    /tmp/model/USE_POLICY.md
    /tmp/model/tokenizer_checklist.chk
    /tmp/model/tokenizer.model
    /tmp/model/consolidated.00.pth

    # Or launch a finetuning run
    $ tune launch full_finetune.py --config my_config
    Starting run...

View all documentation for commands at https://pytorch.org/torchtune/docs.
```

### `ls`

Key Points:
- This lists **only** the built-in recipes and configs, meant only to get
the user off the ground and able to modify
- Pairs recipes visually with their available configs

```
$ tune ls --help
List all built-in recipes or configs

USAGE
    $ tune ls

EXAMPLES
    $ tune ls
    RECIPE                  CONFIGS
    full_finetune.py        alpaca_llama2_full_finetune.yaml
                            alpaca_mistral_full_finetune.yaml
    lora_finetune.py        alpaca_llama2_lora_finetune.yaml
    alpaca_generate.py      <>

To get started you can run:
$ tune run full_finetune.py --config alpaca_llama2_full_finetune.yaml
```

### `cp`

Key Points
- User still needs some way to copy a config or recipe into their local dir in
order to modify

```
$ tune cp --help
Copy a recipe or config to local

USAGE
    $ tune cp RECIPE/CONFIG [OPTIONS]

OPTIONS
    --to            Where to copy the file to
    -h, --help      Print help message and exit

EXAMPLES
    # Copy a recipe
    $ tune cp full_finetune.py --to my_project/full_finetune.py

    # Copy a config
    $ tune cp alpaca_llama2_full_finetune.yaml --to .

Need to see all possible configs to copy? Try running `tune ls`
```

### `download`

Key Points:
- User generally doesn't care where the model comes from, just that it's easy to get the correct model,
e.g. `ollama run llama2`
- **Why do I need to convert the model?** was a question that came up. This automatically
converts the model to a format supported by TorchTune

```
$ tune download --help
Download a model checkpoint

USAGE
    $ tune download MODEL [OPTIONS]

MODELS
    llama2                  Llama2 7B model from Meta
    mistral                 Mistral 7B model from Mistral AI.

OPTIONS
    --hf-token              HuggingFace authentication token, for gated models
    --convert               Bool to convert checkpoint into format supported by TorchTune, default True
    --output-dir            Output directory to write model files to
    -h, --help              Print help message and exit

EXAMPLES
    $ tune download llama2
    Fetching 10 files: 100%|█████████████████████| 10/10 [00:00<00:00, 100.72it/s]
    Succesfully downloaded model repo and wrote to the following locations:
    /tmp/model/.gitattributes
    /tmp/model/LICENSE.txt
    /tmp/model/Responsible-Use-Guide.pdf
    /tmp/model/README.md
    /tmp/model/checklist.chk
    /tmp/model/params.json
    /tmp/model/USE_POLICY.md
    /tmp/model/tokenizer_checklist.chk
    /tmp/model/tokenizer.model
    /tmp/model/model.pt


For a full list of supported models, see https://pytorch.org/torchtune/docs/models
```

### `run`

Key Points:
- Shadowing `torchrun` was an effective way to get up and running quickly, but
caused confusion and does not mirror how other repositories handle distributed,
user should be able to specify a number of GPUs <= 8 and it should run without
any other questions
- Renaming to `run` reflects "running" any recipe

```
$ tune run --help
Run a recipe

USAGE
    $ tune run RECIPE [OPTIONS]

OPTIONS
    --config        Specify a config file for a given recipe

A user can also override any config option for a given recipe
through the command line, e.g. `$ tune run example.py --config example_config.yaml --num_gpus 4`

Take a look at the recipes documentation page for all possible overrides: <LINK>.
```

### `upload`

```
$ tune upload --help
Upload a finetuned model to the HuggingFace Hub

USAGE
    $ tune upload CKPT [OPTIONS]

CKPT
    Model checkpoint

OPTIONS
    --hf-token          HuggingFace authentication token
    -f, --force         Force write, overwriting any previously existing ckpt
    -h, --help

EXAMPLES
    $ tune upload finetuned-model.pt
    Succesfully authenticated!
    Uploading to the hub: 100%|███████████████████| 1/1 [00:00<00:00, 1100.72it/s]
    You can find your model at https://huggingface.co/hub/my-model
```

### `share`

Key Points:
- We need an easy and quick way to share for debugging and exploratory reasons,
Axolotl does a great job of this
- Needs functionality that doesn't exist currently: imagining creating a output dir
that copies the recipe, defaults, config, and logs into a single directory with a
run ID that can be easily shared


```
$ tune share --help
Share all information related to a recipe run

USAGE
    $ tune share RUN_DIR [OPTIONS]

RUN_DIR
    Directory containing all information related to a specific run

OPTIONS
    --compress          Use zip to compress the file's contents
    --redact            Redact all keys located in YAML config file
    --email             List of emails to send this to
    -h, --help          Show help message and exit

EXAMPLES
    $ tune share ./ --compress
    Zipping run_dir...
    Ready to share!
```

## Research

Highly recommend reading through the following resources on your own time or if you have specific ideas
you'd like to see implemented in the CLI tool.

- Design guidelines
    - [Overview and philosophies](https://clig.dev/)
    - [Revamping a CLI](https://uxdesign.cc/user-experience-clis-and-breaking-the-world-baed8709244f)
    - [On naming within a CLI](https://smallstep.com/blog/the-poetics-of-cli-command-names/)
    - Heroku devs [on building a CLI](https://medium.com/@jdxcode/12-factor-cli-apps-dd3c227a0e46)
    - Zapier devs [on building a CLI](https://zapier.com/engineering/how-to-cli/)
- General inspiration
    - Neal Stephenson [story on the command line](https://web.stanford.edu/class/cs81n/command.txt)
    - [git](https://git-scm.com/book/en/v2/Getting-Started-The-Command-Line)
    - [docker-compose](https://docs.docker.com/compose/reference/)
    - [jq](https://jqlang.github.io/jq/)
    - [ollama](https://ollama.com/)
- Finetuning specific inspiration
    - HuggingFace `accelerate` [CLI](https://huggingface.co/docs/accelerate/en/package_reference/cli)
