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
