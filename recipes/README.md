# Training Recipes

&nbsp;

## What are Recipes?

Recipes are the primary entry points for TorchTune users. These can be thought of as end-to-end pipelines for training and optionally evaluating LLMs. Each recipe consists of three components:

- **Configurable parameters**, specified through yaml configs [example](https://github.com/pytorch-labs/torchtune/blob/main/recipes/configs/alpaca_llama2_full_finetune.yaml), command-line overrides and dataclasses
- **Recipe class**, core logic needed for training, exposed to users through a set of APIs [interface](https://github.com/pytorch-labs/torchtune/blob/main/recipes/interfaces.py)
- **Recipe script**, puts everything together including parsing and validating configs, setting up the environment, and correctly using the recipe class

&nbsp;

## Recipe Design

Recipes in TorchTune are:

1. **Simple**. Written fully in native-PyTorch.
2. **Correct**. Numerical parity verification for every component and extensive comparisons with reference implementations and benchmarks.
3. **Easy to Understand**. Each recipe provides a limited set of meaningful features, instead of every possible feature hidden behind 100s of flags. Code duplication is preferred over unnecessary abstractions.
4. **Easy to Extend**. No dependency on training frameworks and no implementation inheritance. Users don't need to go through layers-upon-layers of abstractions to figure out how to extend core functionality.
5. **Accessible to a spectrum of Users**. Users can decide how they want to interact with TorchTune Recipes:
    - Start training models by modifying existing configs
    - Modify existing recipes for custom cases
    - Directly use available building blocks to write completely new recipes/training paradigms

&nbsp;

## How to specify parameters for recipes

The arguments for a recipe are defined in a params object (such as `FullFinetuneParams`) that contains the full list of configurable parameters. These are either set to default values or sourced from the YAML file listed with `--config` and `--override` arguments in the `tune` CLI. The `TuneArgumentParser` class is responsible for parsing the provided config file and overrides and funneling it into the corresponding params object for the recipe the user wishes to run. The order of overrides from these parameter sources is as follows, with highest precedence first:

CLI &rarr; Config &rarr; Params defaults

The config is the primary entry point for users, with CLI overrides providing flexibility for quick experimentation.

### Examples

To run the `finetune_llm` recipe with the `alpaca_llama2_finetune.yaml` config, run this command:

On GPU (without PyTorch Distributed):
```
tune finetune_llm --config alpaca_llama2_finetune --override device=cuda
```

On multiple GPUs with FSDP:
```
tune --nnodes 1 --nproc_per_node 4 finetune_llm --config alpaca_llama2_finetune --override enable_fsdp=True enable_activation_checkpointing=False device=cuda
```

To run the generation recipe, run this command from inside the main `/torchtune` directory:
```
python -m recipes.alpaca_generate --native-checkpoint-path /tmp/finetune-llm/model_0.ckpt --tokenizer-path ~/llama/tokenizer.model --input "What is some cool music from the 1920s?"
```

&nbsp;

## Creating parameters for custom recipes
In general, you should expose the minimal amount of parameters you need to run and experiment with your recipes. These should be collected in a dataclass object that is passed into the recipe.
```
class FullFinetuneParams:
    # Model
    model: str = ""
    model_checkpoint: str = ""
```
In the dataclass, all fields should have defaults assigned to them. If a reasonable value cannot be assigned or it is a required argument, use the null value for that data type as the default and ensure that it is set by the user in the `__post_init__` (see Parameter Validation). The dataclass should go in the `recipes/params/` folder and the name of the file should match the name of the recipe file you are creating.

To link the dataclass object with config and CLI parsing, you can use the `TuneArgumentParser` object and funnel the parsed arguments into your dataclass.
```
if __name__ == "__main__":
    parser = utils.TuneArgumentParser(
        description=FullFinetuneParams.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Get user-specified args from config and CLI and create params for recipe
    args, _ = parser.parse_known_args()
    args = vars(args)
    params = FullFinetuneParams(**args)

    logger = utils.get_logger("DEBUG")
    logger.info(msg=f"Running finetune_llm.py with parameters {params}")

    recipe(params)
```

### Parameter validation
To validate user arguments for your dataclass and recipe, use the `__post_init__` method to house any checks and raised exceptions.
```
def __post_init__(self):
    for param in fields(self):
        if getattr(self, param.name) == "":
            raise TypeError(f"{param.name} needs to be specified")
```

### Write config
Now that you've set up the recipe, the parameters dataclass, and the parser, you can create a simple config in `recipes/configs/` that specifies values for any of the fields you defined in the dataclass. Anything that is not specified should have a default value in the dataclass.

### Testing configs
TorchTune has testing for every config added to the library, namely ensuring that it instantiates the dataclass and runs the recipe correctly. To add your config to this test suite, simply update the dictionary in `recipes/tests/configs/test_configs.py`.
```
config_to_params = {
    os.path.join(ROOT_DIR, "alpaca_llama2_full_finetune.yaml"): FullFinetuneParams,
    ...,
}
```

### Running your recipe
If everything is set up correctly, you should be able to run your recipe just like the existing library recipes using the `tune` command:
```
tune <recipe> --config <config> --override ...
```
