# Training Recipes

&nbsp;

## What are Recipes?

Recipes are the primary entry points for torchtune users. These can be thought of as end-to-end pipelines for training and optionally evaluating LLMs. Each recipe consists of three components:

- **Configurable parameters**, specified through yaml configs [example](https://github.com/pytorch/torchtune/blob/main/recipes/configs/llama2/7B_full.yaml) and command-line overrides
- **Recipe class**, core logic needed for training, exposed to users through a set of APIs [interface](https://github.com/pytorch/torchtune/blob/main/recipes/interfaces.py)
- **Recipe script**, puts everything together including parsing and validating configs, setting up the environment, and correctly using the recipe class

&nbsp;

## Recipe Design

Recipes in torchtune are:

1. **Simple**. Written fully in native-PyTorch.
2. **Correct**. Numerical parity verification for every component and extensive comparisons with reference implementations and benchmarks.
3. **Easy to Understand**. Each recipe provides a limited set of meaningful features, instead of every possible feature hidden behind 100s of flags. Code duplication is preferred over unnecessary abstractions.
4. **Easy to Extend**. No dependency on training frameworks and no implementation inheritance. Users don't need to go through layers-upon-layers of abstractions to figure out how to extend core functionality.
5. **Accessible to a spectrum of Users**. Users can decide how they want to interact with torchtune Recipes:
    - Start training models by modifying existing configs
    - Modify existing recipes for custom cases
    - Directly use available building blocks to write completely new recipes/training paradigms

&nbsp;

### Quantization and Sparsity

torchtune integrates with `torchao`(https://github.com/pytorch-labs/ao/) for architecture optimization techniques including quantization and sparsity. Currently only some quantization techniques are integrated, see the docstrings in the [quantization recipe](quantize.py) for more details.

#### Quantize
To quantize a model (default is int4 weight only quantization):
```
tune run quantize --config quantization
```

#### Eval
To evaluate a quantized model, make the following changes to the default [evaluation config](configs/eleuther_evaluation.yaml)


```yaml
# Currently we only support torchtune checkpoints when
# evaluating quantized models. For more details on checkpointing see
# https://pytorch.org/torchtune/main/examples/checkpointer.html
# Make sure to change the default checkpointer component
checkpointer:
  _component_: torchtune.utils.FullModeltorchtuneCheckpointer
  ..
  checkpoint_files: [<quantized_model_checkpoint>]

# Quantization specific args
quantizer:
  _component_: torchtune.utils.quantization.Int4WeightOnlyQuantizer
  groupsize: 256
```

and run evaluation:
```bash
tune run eleuther_eval --config eleuther_evaluation
```

#### Generate
To run inference using a quantized model, make the following changes to the default [generation config](configs/generation.yaml)


```yaml
# Currently we only support torchtune checkpoints when
# evaluating quantized models. For more details on checkpointing see
# https://pytorch.org/torchtune/main/examples/checkpointer.html
# Make sure to change the default checkpointer component
checkpointer:
  _component_: torchtune.utils.FullModeltorchtuneCheckpointer
  ..
  checkpoint_files: [<quantized_model_checkpoint>]

# Quantization Arguments
quantizer:
  _component_: torchtune.utils.quantization.Int4WeightOnlyQuantizer
  groupsize: 256
```

and run generation:
```bash
tune run generate --config generation
```
