# Training Recipes

&nbsp;

## What are Recipes?

Recipes are the primary entry points for TorchTune users. These can be thought of as end-to-end pipelines for training and optionally evaluating LLMs. Each recipe consists of three components:

- **Configurable parameters**, specified through yaml configs [example](https://github.com/pytorch/torchtune/blob/main/recipes/configs/llama2/7B_full.yaml) and command-line overrides
- **Recipe class**, core logic needed for training, exposed to users through a set of APIs [interface](https://github.com/pytorch/torchtune/blob/main/recipes/interfaces.py)
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

## Checkpoints and using the TorchTune Checkpointer

TorchTune supports multiple checkpoint formats. For Llama2 specifically, this includes:

&nbsp;

**Meta Format**. This refers to the checkpoints uploaded by the original authors. You can download this checkpoint from the
HF Hub, using the following command
```
tune download --repo-id meta-llama/Llama-2-7b \
--hf-token <HF_TOKEN> \
--output-dir /tmp/llama2
```
This should load in a single `consolidated.00.pth` file. You can use this checkpoint directly with TorchTune using the
`FullModelMetaCheckpointer`. When starting a fresh training run, the checkpointer component of the config looks like this:
```
checkpointer:
  _component_: torchtune.utils.FullModelMetaCheckpointer
  checkpoint_dir: /tmp/llama2
  checkpoint_files: [consolidated.00.pth]
  output_dir: /tmp/llama2
  model_type: LLAMA2
resume_from_checkpoint: False
```
The checkpointer will take care of converting the state_dict into a format compatible with TorchTune.

&nbsp;

**HF Format**. This refers to the HF-formatted llama2 checkpoints available in the HF repo. You can download this checkpoint from the HF Hub, using the following command
```
tune download --repo-id meta-llama/Llama-2-7b-hf \
--hf-token <HF_TOKEN> \
--output-dir /tmp/llama2
```
This should load in two checkpoint files: `pytorch_model-00001-of-00002.bin` and  `pytorch_model-00002-of-00002.bin`. You can use this checkpoint directly with TorchTune using the`FullModelHFCheckpointer`. When starting a fresh training run, the checkpointer component of the config looks like this:
```
checkpointer:
  _component_: torchtune.utils.FullModelHFCheckpointer
  checkpoint_dir: /tmp/llama2-hf
  checkpoint_files: [pytorch_model-00001-of-00002.bin, pytorch_model-00002-of-00002.bin]
  output_dir: /tmp/llama2-hf
  model_type: LLAMA2
resume_from_checkpoint: False
```
The checkpointer will take care of converting the state_dict into a format compatible with TorchTune.

&nbsp;

### Checkpoints created during Training

TorchTune recipes will output checkpoints in two scenarios:

&nbsp;

**Mid-training checkpoints**. Checkpoints are created at the end of each epoch. Mid-training, in addition to the model checkpoint, the checkpointer will output additional checkpoint files. These include:
- Recipe Checkpoint. The `recipe_state.pt` file contains information about training needed to restart training from that point onwards. This includes training seed, number of epochs completed, optimizer state etc.
- Adapter Checkpoint. If using PEFT like LoRA, the checkpointer additionally outputs the adapter weights needed to correctly intialize the LoRA parameters to restart training.

To correctly restart training, the checkpointer needs access to the Recipe Checkpoint and optionally to the Adapter Checkpoint (in case training LoRA). A sample config component for LoRA looks something like this:

```
checkpointer:
  _component_: torchtune.utils.FullModelHFCheckpointer
  checkpoint_dir: /tmp/llama2-hf
  checkpoint_files: [pytorch_model-00001-of-00002.bin, pytorch_model-00002-of-00002.bin]
  adapter_checkpoint: adapter_0.pt
  recipe_checkpoint: recipe_state.pt
  output_dir: /tmp/llama2-hf
  model_type: LLAMA2
```

Note: In case of PEFT (eg: LoRA), the checkpoint files should continue to point towards the original base model since the output model checkpoint file contains the merged weights which should not be used for restarting training.

&nbsp;

**End-of-training checkpoints**. Torchtune outputs checkpoints in the same format as the input checkpoint. This means that you can use the output checkpoint with the same set of tools that you could use with the input checkpoints. This includes evaluation harnesses, inference engines etc.

For example, to use the Meta format checkpoints with llama.cpp, you can directly convert these to GGUF format using the convertor in the llama.cpp code base. After you've followed the setup instructions in the llama.cpp README, you can run the following:

```
python3 convert.py /tmp/llama2/meta_model_0.pt --ctx 4096
```

This will output a gguf file in the same precision which can be used for running inference.

### Architecture Optimization

TorchTune integrates with `torchao`(https://github.com/pytorch-labs/ao/) for architecture optimization techniques including quantization and sparsity. Currently only some quantization techniques are integrated, see `receipes/configs/quantize.yaml` for more details.

#### Quantize
To quantize a model (default is int4 weight only quantization):
```
tune run quantize --config quantize
```

#### Eval
To evaluate a quantized model, add the following to `receipes/configs/eleuther_eval.yaml`:


```
# make sure to change the checkpointer component
checkpointer:
  _component_: torchtune.utils.FullModelTorchTuneCheckpointer

# Quantization specific args
quantizer:
  _component_: torchtune.utils.quantization.Int4WeightOnlyQuantizer
  groupsize: 256
```

and run the eval command:
```
tune run eleuther_eval --config eleuther_eval
```

#### Generate
Changes in `receipes/configs/generate.yaml`
```
# Model arguments
checkpointer:
# make sure to change the checkpointer component
checkpointer:
  _component_: torchtune.utils.FullModelTorchTuneCheckpointer
  checkpoint_files: [meta_model_0.4w.pt]

# Quantization Arguments
quantizer:
  _component_: torchtune.utils.quantization.Int4WeightOnlyQuantizer
  groupsize: 256
```

and run generate command:
```
tune run generate --config generate
```

#### GPTQ

GPTQ is an algorithm to improve the accuracy of quantized model through optimizing the loss of (activation * weight) together, here are the changes that's needed to use it for int4 weight only quantization

`receipes/configs/quantize.yaml`

We'll publish doc pages for different quantizers in torchao a bit later. Please check `receipes/configs/quantized.yaml for how to use them for now.

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

`receipes/quantize.py`

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
tune run quantize --config quantize
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
