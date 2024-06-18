# Single GPU 
# Full 7b

Run_llama2-7b_single_device_full() {
    tune run full_finetune_single_device \
    --config llama2/7B_full_low_memory \
    device=xpu \
    optimizer._component_=torch.optim.AdamW \
    checkpointer.checkpoint_dir=/workspace1/huggingface/hub/llama2-7b \
    tokenizer.path=/workspace1/huggingface/hub/llama2-7b/tokenizer.model \
    checkpointer.output_dir=/workspace1/huggingface/hub \
    output_dir=/home/zhuhong/LLM/torchtune/recipes/alpaca-llama2-finetune
}


# Full 7b with low PagedAdamw

Run_llama2-7b_single_device_full_pagedadamw() {
    tune run full_finetune_single_device \
    --config llama2/7B_full_low_memory \
    device=xpu \
    checkpointer.checkpoint_dir=/workspace1/huggingface/hub/llama2-7b \
    tokenizer.path=/workspace1/huggingface/hub/llama2-7b/tokenizer.model \
    checkpointer.output_dir=/workspace1/huggingface/hub \
    output_dir=/home/zhuhong/LLM/torchtune/recipes/alpaca-llama2-finetune
}



# LoRA 7b
Run_llama2-7b_single_device_lora() {
    tune run lora_finetune_single_device \
    --config llama2/7B_lora_single_device \
    device=xpu \
    checkpointer.checkpoint_dir=/workspace1/huggingface/hub/llama2-7b \
    tokenizer.path=/workspace1/huggingface/hub/llama2-7b/tokenizer.model \
    checkpointer.output_dir=/workspace1/huggingface/hub \
    output_dir=/home/zhuhong/LLM/torchtune/recipes/alpaca-llama2-finetune
}


# # QLora 7b
Run_llama2-7b_single_device_qlora() {
    tune run lora_finetune_single_device \
    --config llama2/7B_qlora_single_device \
    checkpointer.checkpoint_dir=/data2/zhuhong/huggingface/llama2-7b \
    tokenizer.path=/data2/zhuhong/huggingface/llama2-7b/tokenizer.model \
    checkpointer.output_dir=/data2/zhuhong/huggingface \
    output_dir=/home/pt-gpu/zhuhong/torchtune/recipes/alpaca-llama2-finetune
}

# distributed full
Run_llama2-7b_distributed_full() {
    tune run --nproc_per_node 4 full_finetune_distributed \
    --config llama2/7B_full \
    device=xpu \
    checkpointer.checkpoint_dir=/workspace1/huggingface/hub/llama2-7b \
    tokenizer.path=/workspace1/huggingface/hub/llama2-7b/tokenizer.model \
    checkpointer.output_dir=/workspace1/huggingface/hub \
    output_dir=/home/zhuhong/LLM/torchtune/recipes/alpaca-llama2-finetune
}


# distributed lora
Run_llama2-7b_distributed_lora() {
    tune run --nproc_per_node 4 lora_finetune_distributed \
    --config llama2/7B_lora \
    device=xpu \
    checkpointer.checkpoint_dir=/workspace1/huggingface/hub/llama2-7b \
    tokenizer.path=/workspace1/huggingface/hub/llama2-7b/tokenizer.model \
    checkpointer.output_dir=/workspace1/huggingface/hub \
    output_dir=/home/zhuhong/LLM/torchtune/recipes/alpaca-llama2-finetune
}


main() {
  # Run_llama2-7b_single_device_full
  # Run_llama2-7b_single_device_full_pagedadamw
  # Run_llama2-7b_single_device_lora
  # Run_llama2-7b_single_device_qlora
  # Run_llama2-7b_distributed_full
  Run_llama2-7b_distributed_lora
}

main
