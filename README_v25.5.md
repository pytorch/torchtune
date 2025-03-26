


# Torchtune




## Overview 📚


torchtune is a PyTorch library for easily authoring, post-training, and experimenting with LLMs. It provides:

- Hackable training recipes for SFT, knowledge distillation, DPO, PPO, GRPO, and quantization-aware training
- Simple PyTorch implementations of popular LLMs like Llama, Gemma, Mistral, Phi, Qwen, and more
- Best-in-class memory efficiency, performance improvements, and scaling, utilizing the latest PyTorch APIs
- YAML configs for easily configuring training, evaluation, quantization or inference recipes

&nbsp;

# Finetuning
This section describes finetuning llama-3.1-70b using wikitext dataset on a single node using [Torchtune](https://pytorch.org/torchtune/stable/index.html) utility.

### Environment setup

```bash
docker run -it --device /dev/dri --device /dev/kfd --network host --ipc host --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged    -v  $HOME/.ssh:/root/.ssh  -v /home/amd:/home/amd --shm-size 128G --name YOUR_NAME_HERE DOCKER_IMAGE

# This is the main branch
git clone https://github.com/AMD-AIG-AIMA/torchtune.git --branch Torchtune_AMD

cd torchtune

# If you don't have access to this model, see the section below for an alternative source.
huggingface-cli login
huggingface-cli download meta-llama/Llama-3.3-70B-Instruct --local-dir ./models/Llama-3.3-70B-Instruct --exclude 'original/*.pth'
# Note: Finetune part of the tutorial requires 140GB of disk space for model + dataset.

# Test the llama-3.3-70B full fineutning with wikitext dataset
# To enable, packed=true, set the sequence length to 512, 1024, etc.
# If you want to run for a complete epoch, remove MAX_STEPS
MODEL_DIR=./models/Llama-3.3-70B-Instruct COMPILE=True PACKED=False SEQ_LEN=null CPU_OFFLOAD=False ACTIVATION_CHECKPOINTING=True MBS=64 GAS=1 EPOCHS=1 SEED=42 MAX_STEPS=20 bash run_llama_3_3_full_wiki.sh

# Test the llama-3.3-70B LoRA fineutning with wikitext dataset
# To enable, packed=true, set the sequence length to 512, 1024, etc.
# If you want to run for a complete epoch, remove MAX_STEPS
MODEL_DIR=./models/Llama-3.3-70B-Instruct COMPILE=True PACKED=False SEQ_LEN=null CPU_OFFLOAD=False ACTIVATION_CHECKPOINTING=True MBS=64 GAS=1 EPOCHS=1 SEED=42 MAX_STEPS=20 bash run_llama_3_3_LoRA_wiki.sh

# Test the llama-3.3-70B qLoRA fineutning with wikitext dataset
# To enable, packed=true, set the sequence length to 512, 1024, etc.
# If you want to run for a complete epoch, remove MAX_STEPS
MODEL_DIR=./models/Llama-3.3-70B-Instruct COMPILE=True PACKED=False SEQ_LEN=null CPU_OFFLOAD=False ACTIVATION_CHECKPOINTING=True MBS=64 GAS=1 EPOCHS=1 SEED=42 MAX_STEPS=20 bash run_llama_3_3_qLoRA_wiki.sh
```


&nbsp;

## License

torchtune is released under the [BSD 3 license](./LICENSE). However you may have other legal obligations that govern your use of other content, such as the terms of service for third-party models.
