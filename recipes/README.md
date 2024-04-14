# Training Recipes

&nbsp;

## What are Recipes?

Recipes are the primary entry points for torchtune users. These can be thought of as end-to-end pipelines for training and optionally evaluating LLMs. More information about recipes, their components and a design overview can be found in our [recipe deep-dive](https://pytorch.org/torchtune/main/examples/recipe_deepdive.html).

&nbsp;

# Quantization and Sparsity

torchtune integrates with [torchao](https://github.com/pytorch-labs/ao/) for architecture optimization techniques including quantization and sparsity. Currently only some quantization techniques are integrated, see the docstrings in the [quantization recipe](quantize.py) for more details.

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
  _component_: torchtune.utils.FullModelTorchTuneCheckpointer
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
  _component_: torchtune.utils.FullModelTorchTuneCheckpointer
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
