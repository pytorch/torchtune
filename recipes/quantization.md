# Quantization and Sparsity

torchtune integrates with [torchao](https://github.com/pytorch/ao/) for architecture optimization techniques including quantization and sparsity. Currently only some quantization techniques are integrated, see the docstrings in the [quantization recipe](quantize.py) and the [QAT recipe](qat_distributed.py) for more details.

#### Quantize
To quantize a model (default is int4 weight only quantization):
```
tune run quantize --config quantization
```

#### Quantization-Aware Training (QAT)

(PyTorch 2.4+)

QAT refers to applying fake quantization to weights and/or activations during finetuning,
which means simulating only the quantization math without actually casting the original
dtype to a lower precision. You can run QAT with finetuning using the following command:

```
tune run --nproc_per_node 4 qat_distributed --config llama3/8B_qat_full
```

This produces an unquantized model in the original data type. To get an actual quantized model,
follow this with `tune run quantize` while specifying the same quantizer in the config, e.g.

```yaml
# QAT specific args
quantizer:
  _component_: torchtune.utils.quantization.Int8DynActInt4WeightQATQuantizer
  groupsize: 256
```

Currently only `torchtune.utils.quantization.Int8DynActInt4WeightQATQuantizer`
is supported. This refers to int8 dynamic per token activation quantization
combined with int4 grouped per axis weight quantization. For more details,
please refer to the [torchao implementation](https://github.com/pytorch/ao/blob/950a89388e88e10f26bbbbe2ec0b1710ba3d33d1/torchao/quantization/prototype/qat.py#L22).


#### Eval
To evaluate a quantized model, make the following changes to the default [evaluation config](configs/eleuther_evaluation.yaml)


```yaml
# Currently we only support torchtune checkpoints when
# evaluating quantized models. For more details on checkpointing see
# https://pytorch.org/torchtune/main/deep_dives/checkpointer.html
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
# https://pytorch.org/torchtune/main/deep_dives/checkpointer.html
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
