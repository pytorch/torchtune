
# Torchtune Finetuning
This section describes finetuning llama models with full-weight/LoRA/qLoRA using alpaca dataset on a single node using [Torchtune](https://pytorch.org/torchtune/stable/index.html) utility.

### Environment Setup and Sample Examples Commands

```bash
docker pull  rocm/pytorch-training:v25.5
docker run -it --device /dev/dri --device /dev/kfd --network host --ipc host --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged -v  $HOME/.ssh:/root/.ssh -v /home/amd:/home/amd --shm-size 128G --name YOUR_NAME_HERE  rocm/pytorch-training:v25.5

# This is the main branch
git clone https://github.com/AMD-AIG-AIMA/torchtune.git

cd torchtune/examples

# If you don't have access to this model, see the section below for an alternative source.
huggingface-cli login
huggingface-cli download meta-llama/Llama-3.3-70B-Instruct --local-dir ./models/Llama-3.3-70B-Instruct --exclude 'original/*.pth'
# Note: Finetune part of the tutorial requires 140GB of disk space for model + dataset.

# Test the llama-3.3-70B full fineutning with wikitext dataset
# To enable, packed=true, set the sequence length to 512, 1024, etc.
# If you want to run for a complete epoch, remove MAX_STEPS
MODEL_DIR=./models/Llama-3.3-70B-Instruct COMPILE=True PACKED=False SEQ_LEN=null CPU_OFFLOAD=False ACTIVATION_CHECKPOINTING=True MBS=64 GAS=1 EPOCHS=1 SEED=42 MAX_STEPS=20 bash run_llama_3_3_full.sh

# Test the llama-3.3-70B LoRA fineutning with wikitext dataset
# To enable, packed=true, set the sequence length to 512, 1024, etc.
# If you want to run for a complete epoch, remove MAX_STEPS
MODEL_DIR=./models/Llama-3.3-70B-Instruct COMPILE=True PACKED=False SEQ_LEN=null CPU_OFFLOAD=False ACTIVATION_CHECKPOINTING=True MBS=64 GAS=1 EPOCHS=1 SEED=42 MAX_STEPS=20 bash run_llama_3_3_LoRA.sh

# Test the llama-3.3-70B qLoRA fineutning with wikitext dataset
# To enable, packed=true, set the sequence length to 512, 1024, etc.
# If you want to run for a complete epoch, remove MAX_STEPS
MODEL_DIR=./models/Llama-3.3-70B-Instruct COMPILE=True PACKED=False SEQ_LEN=null CPU_OFFLOAD=False ACTIVATION_CHECKPOINTING=True MBS=64 GAS=1 EPOCHS=1 SEED=42 MAX_STEPS=20 bash run_llama_3_3_qLoRA.sh

# Similary you can finetune different llama models with full-weight, LoRA, qLoRA optimizations.
```
#### Changes required for finetuning with wikitext dataset:

1. If you want to finetune with **Wikitext** from **EleutherAI/wikitext_document_level**, just change the **_component_** to `torchtune.datasets.wikitext_dataset` in the yaml files.
2. If you want to run with a much bigger version of ***Wikitext-103-v1***, then follow the steps: \
   2.1 Go to the file `torchtune/datasets/_wikitext.py` \
   2.2 We need to just put **wikitext** instead of **EleutherAI/wikitext_document_level** for `source` \
   2.3 Change the column name from **page** to **text** \
   2.4 Change the **_component_** to `torchtune.datasets.wikitext_dataset` in the yaml files.


