To run the finetune_llm recipe, run this command from inside the main `/torchtune` directory:

On CPU (without PyTorch Distributed):
```
python -m recipes.finetune_llm --dataset alpaca --tokenizer llama2_tokenizer --tokenizer-checkpoint ~/llama/tokenizer.model --model llama2_7b --model-checkpoint /tmp/native_checkpoints/llama2-7b --batch-size 8 --device cpu
```

On multiple GPUs with FSDP:
```
torchrun --nnodes 1 --nproc_per_node 8 recipes/finetune_llm.py --dataset alpaca --tokenizer llama2_tokenizer --tokenizer-checkpoint ~/llama/tokenizer.model --model llama2_7b --fsdp True --activation-checkpointing False --model-checkpoint /tmp/native_checkpoints/llama2-7b --batch-size 8 --device cuda --autocast-precision bf16
```


To run the generation recipe, run this command from inside the main `/torchtune` directory:
The recipe assumes you are running inference on a model finetuned like above, but if you are using a pre-trained model, replace native_state_dict["model"] with native_state_dict in L58.
```
python -m recipes.generate --native-checkpoint-path /tmp/finetune-llm/model_0.ckpt --tokenizer-path ~/llama/tokenizer.model
```
