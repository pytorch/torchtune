# Quantization and Sparsity

torchtune integrates with [torchao](https://github.com/pytorch/ao/) for QAT and QLoRA. Currently only some quantization techniques are integrated, see the docstrings in the [quantization recipe](quantize.py) and the [QAT recipe](qat_distributed.py) for more details.

For post training quantization, we recommend using `torchao` directly: https://github.com/pytorch/ao/blob/main/torchao/quantization/README.md to quantize their model
and do eval/benchmark in torchao as well: https://github.com/pytorch/ao/tree/main/torchao/_models/llama.

## Quantization-Aware Training (QAT)

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
  _component_: torchtune.training.quantization.Int8DynActInt4WeightQATQuantizer
  groupsize: 256
```

Currently only `torchtune.training.quantization.Int8DynActInt4WeightQATQuantizer`
is supported. This refers to int8 dynamic per token activation quantization
combined with int4 grouped per axis weight quantization. For more details,
please refer to the [torchao implementation](https://github.com/pytorch/ao/blob/950a89388e88e10f26bbbbe2ec0b1710ba3d33d1/torchao/quantization/prototype/qat.py#L22).

## Eval
To evaluate a quantized model, make the following changes to the default [evaluation config](configs/eleuther_evaluation.yaml)


```yaml
# Currently we only support torchtune checkpoints when
# evaluating quantized models. For more details on checkpointing see
# https://pytorch.org/torchtune/main/deep_dives/checkpointer.html
# Make sure to change the default checkpointer component
checkpointer:
  _component_: torchtune.training.FullModelTorchTuneCheckpointer
  ..
  checkpoint_files: [<quantized_model_checkpoint>]

# Quantization specific args
quantizer:
  _component_: torchtune.training.quantization.Int8DynActInt4WeightQuantizer
  groupsize: 256
```

Noet: we can use `Int8DynActInt4WeightQuantizer` to load a QAT quantized model since it's the same type of quantization.

and run evaluation:
```bash
tune run eleuther_eval --config eleuther_evaluation
```

## Generate
To run inference using a quantized model, make the following changes to the default [generation config](configs/generation.yaml)


```yaml
# Currently we only support torchtune checkpoints when
# evaluating quantized models. For more details on checkpointing see
# https://pytorch.org/torchtune/main/deep_dives/checkpointer.html
# Make sure to change the default checkpointer component
checkpointer:
  _component_: torchtune.training.FullModelTorchTuneCheckpointer
  ..
  checkpoint_files: [<quantized_model_checkpoint>]

# Quantization Arguments
quantizer:
  _component_: torchtune.training.quantization.Int8DynActInt4WeightQuantizer
  groupsize: 256
```

and run generation:
```bash
tune run generate --config generation
```
