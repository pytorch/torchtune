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

### Architecture Optimization

torchtune integrates with `torchao`(https://github.com/pytorch-labs/ao/) for architecture optimization techniques including quantization and sparsity. Currently only some quantization techniques are integrated, see the docstrings in the [quantization recipe](quantize.py) for more details.

#### Quantize
To quantize a model (default is int4 weight only quantization):
```
tune run quantize --config quantization
```

#### Eval
To evaluate a quantized model, add the following to `recipes/configs/eleuther_eval.yaml`:


```
# make sure to change the checkpointer component
checkpointer:
  _component_: torchtune.utils.FullModeltorchtuneCheckpointer

# Quantization specific args
quantizer:
  _component_: torchtune.utils.quantization.Int4WeightOnlyQuantizer
  groupsize: 256
```

and run the eval command:
```
tune run eleuther_eval --config eleuther_evaluation
```

#### Generate
Changes in `recipes/configs/generation.yaml`
```
# Model arguments
checkpointer:
# make sure to change the checkpointer component
checkpointer:
  _component_: torchtune.utils.FullModeltorchtuneCheckpointer
  checkpoint_files: [meta_model_0-4w.pt]

# Quantization Arguments
quantizer:
  _component_: torchtune.utils.quantization.Int4WeightOnlyQuantizer
  groupsize: 256
```

and run generate command:
```
tune run generate --config generation
```

#### GPTQ

GPTQ is an algorithm to improve the accuracy of quantized model through optimizing the loss of (activation * weight) together, here are the changes that's needed to use it for int4 weight only quantization

`recipes/configs/quantization.yaml`

We'll publish doc pages for different quantizers in torchao a bit later. Please check `recipes/configs/quantized.yaml for how to use them for now.

```
quantizer:
  _component_: torchtune.utils.quantization.Int4WeightOnlyGPTQQuantizer
  blocksize: 128
  percdamp: 0.01
  groupsize: 256

tokenizer:
  _component_: torchtune.models.llama2.llama2_tokenizer
  path: /tmp/llama2/tokenizer.model
```

`recipes/quantize.py`

```
def quantize(self, cfg: DictConfig):
    from torchao.quantization.GPTQ import InputRecorder
    tokenizer = config.instantiate(cfg.tokenizer)
    calibration_seq_length = 100
    calibration_tasks = ['wikitext']
    inputs = InputRecorder(
        tokenizer,
        calibration_seq_length,
        vocab_size=self._model.tok_embeddings.weight.shape[0],
        device="cpu",
    ).record_inputs(
        calibration_tasks,
        5,  # calibration_limit
    ).get_inputs()
    t0 = time.perf_counter()
    self._model = self._quantizer.quantize(self._model, inputs)
    ....
```

Run quantize
```
tune run quantize --config quantization
```

`recipes/eleuther_eval.py`

```
# To skip running the full GPTQ quantization process that typically takes a long time,
# change model = quantizer.quantize(model) to:
model = quantizer._convert_for_runtime(model)
```

`recipes/configs/eleuther_eval.yaml`
```
quantizer:
  _component_: torchtune.utils.quantization.Int4WeightOnlyGPTQQuantizer
  blocksize: 128
  percdamp: 0.01
  groupsize: 256
```
