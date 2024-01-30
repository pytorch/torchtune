# [RFC] Config system for TorchTune
## Why do we need configs?
TorchTune provides keystone recipes for fine-tuning foundational LLM models that
are configurable to user’s needs. The primary entry point for most users to run
fine-tuning recipes will be via CLI using tune. In order to provide a set of
default parameters to run these recipes, we use YAML configs as the interaction
point between the CLI and the recipe script. Configs play the role of describing
the high level details of what is needed to run a specific recipe, such as which
model to fine-tune, which dataset to use, training hyperparameters, etc. More
advanced users may want to create their own configs to further customize recipes
for their needs. Since configs are an important entry point for users,
they must strike a fine balance between flexible yet accessible and easy to
read. TorchTune aims to place a high bar on configs for the greater LLM
fine-tuning community.

## Approach 1: Python dataclasses as configs (HuggingFace)
Instead of using yaml files directly, this approach relies on Python dataclasses
to handle argos for a certain model or category. For example: [LlamaConfig](https://huggingface.co/docs/transformers/v4.36.1/en/model_doc/llama2#transformers.LlamaConfig),
[PEFTConfig](https://huggingface.co/docs/peft/v0.7.1/en/package_reference/config#peft.PeftConfig), [TrainingArguments](https://huggingface.co/docs/transformers/v4.36.1/en/main_classes/trainer#transformers.TrainingArguments). If we adopt this approach,
we will have to parse CLI arguments with something like argparser and manually
create these config objects in the recipe code. Alternatively, we can create
methods to load from dict or json such that we can link it with some yaml
parsing library.
```
# Define in script
fine_tune = FineTuneLLMConfig(
    batch_size=2,
    lr=2e-5,
    epochs=3,
    ...
)

# From CLI
parser.add_argument(...)
parser.add_argument(...)
args = parser.parse_args()
# This could be a helper method
fine_tune = FineTuneLLMConfig(
    batch_size=args.batch_size,
    lr=args.lr,
    epochs=args.epochs,
    ...
)

# From yaml - this could all be a separate util method
cfg = OmegaConf.load('config.yaml')
fine_tune = FineTuneLLMConfig(
    batch_size=cfg.batch_size,
    lr=cfg.lr,
    epochs=cfg.epochs,
    ...
)

recipe(fine_tune)
```
**Pros:** Config validation is automatically handled via Python linting and type
checking, anything else can be added to the dataclass itself
**Cons:** TorchTune specific dataclasses that users need to ramp up to and
understand to read the code, additional layer of abstraction that we need to
maintain

## Approach 2: CLI only (FastChat, LitGPT)
Instead of having yaml files, this approach skips the config layer entirely and
lets the CLI control the recipe directly. This means that the CLI will have to
contain all the arguments that are normally placed in a config. While the
current approach has this ability, it’s not mandatory for users to specify all
the parameters via CLI. Defaults can be set in argparse add_argument but this is
more difficult to read. It also necessitates that we have a CLI parser like
argparse, and we’ll need to maintain all the add_argument calls.

```
# Run on command line
tune finetune_llm --batch_size 2 --lr 2e-5 --epochs 3 ...

# Parse CLI
parser.add_argument(...)
parser.add_argument(...)
args = parser.parse_args()

recipe(args)
```

**Pros:** No intrusive configs and one less layer of abstraction
**Cons:** Additional parsing logic, shareability is more challenging
(can work around with util functions or saving commands in .sh scripts, but this
is a higher barrier), massively long CLI commands if you want to tune a lot of
knobs which is more error prone.

## Approach 3: CLI + YAML configs
The above methods are not necessarily mutually exclusive. In fact, the current
approach uses a combination of CLI and YAML configs as two sources of parameters
to leverage both CLI’s ease of quick experimentation and
shareability/self-documentation of configs. Currently, we use argparse to
process both arguments from CLI and YAML and feed them directly into the recipe.

```
# Config yaml file
dataset: alpaca
model: llama2_7b
...

# Run on command line
tune finetune_llm --config alpaca_llama2_finetune

# Parse CLI and yaml
parser = utils.TuneArgumentParser(...) # config is read here
parser.add_argument(...)
parser.add_argument(...)
args = parser.parse_args()

recipe(args)
```
**Pros:** All parameters can be easily seen in config file but retain ease of
specifying parameters via CLI
**Cons:** Laundry list of add_arguments that needs to be maintained separately
for every recipe, although this can be hidden from user

## Proposed changes
We can actually play to the advantages of all three aproaches if we tightly
define each of their roles.

The flow of parameters will be the following:

Config & CLI &rarr; Parser &rarr; Dataclass &rarr; Recipe

### Configs
Configs will play the role as the primary entry point for users. They will
contain all the user-facing parameters of the recipe and provide a quick high
level snapshot of the most important parameters for a recipe. These are vetted
to be parameters that are commonly configured by beginner and intermediate
users. The config file itself can be structured the exact same way as it is now.

```
# Config yaml file
dataset: alpaca
model: llama2_7b
...
```

### CLI
CLI is another entry point for users. We maintain two entry points to maximize
flexibility, as CLI enables quick experimentation that configs do not provide
and many users will feel most comfortable interacting with recipes via CLI.
CLI overrides take precedence over config parameters, as is convention.
```
tune finetune_llm --config test.yaml --override batch_size=4 ...
```

### Parser
The `TuneArgumentParser` is an extension of `argparse`'s parser that enables the
ability of loading a YAML config file and overriding parameters with anything
specified via CLI. Here, we still use argparse, but instead of using it as a
second location to define a recipe's parameters we simply source it from the
configs or CLI via `--config` and `--override`. This way, the parser doesn't
play the role of both parsing and defining the contract for the recipe. This
approach also removes the clutter of numerous `add_argument` calls.

### Dataclass
The parsed arguments are then funneled into a dataclass that is specific for the
recipe. We use a dataclass instead of flattened parameters in the recipe
signature because it provides a singular location to define the contract of the
recipe and validate user inputs via built-in type checking and custom validation
logic. It also contains the full list of inputs into a recipe, including
experimental parameters and others that only advanced users would want to
configure. This way, we maintain full flexibility and functionality for advanced
users while enabling accessible entry points via configs for beginner users.
The dataclass serves as an enforcer of the recipe's contract, ensuring provided
user inputs via config or CLI is compliant. Thus, the order of precedence for
overrides is as follows:

CLI &rarr; Config &rarr; Dataclass defaults

### Recipe
This is the primary script that is run with user specified parameters. It should
only have a singular dataclass as its arguments, with the dataclass defining
all the individual parameters.

```
# Current approach

# Read from yaml
parser = utils.TuneArgumentParser()
# Read from CLI
parser.add_argument(
"--model",
type=str,
choices=models.list_models(),
help="Model to finetune.",
)
parser.add_argument(
"--tokenizer",
type=str,
choices=models.list_tokenizers(),
help="Model tokenizer.",
)
args = parser.parse_args()
recipe(args)

# Proposed approach

# Read from yaml
cfg = OmegaConf.load('config.yaml')
# Parse command-line arguments and merge with the configuration file
cli_cfg = OmegaConf.from_cli(sys.argv[1:])
cfg = OmegaConf.merge(cfg, cli_cfg)
# Pass the arguments into dataclass for validation
params = RecipeParams(**cfg)
# Execute recipe
recipe(params)
```

### Testing configs in CI
We add unit tests for all configs in `recipes/configs`. The test simply loads
in the config and ensures that the corresponding recipe dataclass can be
instantiated successfully and pass any validation checks. Any time a new config
is added, it is incumbent on the contributor to include this config in the test.

## Design principles
General rules you should follow when creating a user config:
- Keys should be clear and understandable. Avoid acronyms, unless the parameter
is more known by its acronym than its full name (ex: fsdp, lr for learning_rate
[although this one is debatable])
- Values should only be strings, booleans, or numerical

### Fields that should be included in config
These are the criteria that should be used when considering adding a field to
the config.
1. Hyperparameters or fixed values that are used to configure the recipe itself.
Examples include fine-tuning specific parameters such as batch size, learning
rate, number of epochs, optimizer/scheduler, loss, device, and distributed
related fields.
2. Directories for model checkpoint, tokenizer, output, logging, profiling
3. String-based fields that map to builder or class via a getter method.
Examples include model, tokenizer, dataset.

### Fields that should NOT be included in config
Strongly reconsider adding fields to the config if it violates any of these
criteria:
1. Singular parameter for a class or method
2. It is not consumed in the recipe layer (i.e., used in a getter, instantiated,
or used directly by the recipe script). Avoid passing down the instantiated
config object in its entirety in nested methods deeper than the recipe script.
This avoids scenarios where it becomes difficult to debug a parameter because
it is not clear where in the codebase it is actually used.
3. You won’t expect a savvy hobbyist or a beginner hacker to experiment with it.
If it rarely needs to change, it shouldn’t be in the config.

### User vs dev configs
To allow more flexibility with experimental features or for active development
on TorchTune, we can delegate a separate folder that contains configs that can
experiment with the full range of parameters for any recipe. Any field can be
added here for the purpose of development or to test feature requests. Once the
feature lands in a stable version of the library, the config field can be
graduated to user configs if it satisfies the above criteria. To validate dev
configs, devs can run the same test as discussed above and add any validation
logic necessary to the corresponding dataclass.

## FAQ
### How do I add a parameter?
The process for adding a user-controlled feature to a recipe is as follows:
- Add the functionality in the recipe body with corresponding testing
- Expose the parameter by updating the recipe's dataclass
- Add any validation logic needed if there are complex interactions
- Test the feature using a dev config
- Land the feature in a stable version of the library
- Graduate the parameter to user configs if it satisfies the criteria
