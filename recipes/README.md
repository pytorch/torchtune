## How to specify parameters for recipes

The arguments for a recipe are defined in a params object (such as `FullFinetuneParams`) that contains the full list of configurable parameters. These are either set to default values or sourced from the YAML file listed with `--config` and `--override` arguments in the `tune` CLI. The `TuneArgumentParser` class is responsible for parsing the provided config file and overrides and funneling it into the corresponding params object for the recipe the user wishes to run. The order of overrides from these parameter sources is as follows, with highest precedence first:

CLI &rarr; Config &rarr; Params defaults

The config is the primary entry point for users, with CLI overrides providing flexibility for quick experimentation.

## Examples

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
