To run the finetune_llm recipe, run this command from inside the main `/torchtune` directory:

```
torchrun --nnodes 1 --nproc_per_node 8 recipes/finetune_llm.py --dataset alpaca --tokenizer-checkpoint ~/llama/tokenizer.model --model-checkpoint /tmp/native_checkpoints/llama2-7b --batch-size 8
```
