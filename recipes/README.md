To run the finetune_llm recipe, run this command:

On GPU (without PyTorch Distributed):
```
tune finetune_llm --config alpaca_llama2_finetune --device cuda
```

On multiple GPUs with FSDP:
```
tune --nnodes 1 --nproc_per_node 4 finetune_llm --config alpaca_llama2_finetune --fsdp True --activation-checkpointing False  --device cuda
```

To run the generation recipe, run this command from inside the main `/torchtune` directory:
The recipe assumes you are running inference on a model finetuned like above, but if you are using a pre-trained model, replace native_state_dict["model"] with native_state_dict in L58.
```
python -m recipes.generate --native-checkpoint-path /tmp/finetune-llm/model_0.ckpt --tokenizer-path ~/llama/tokenizer.model
```
