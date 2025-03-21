#!/bin/bash

#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --mem=16gb
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --time=05:00:00

module load cuda/12.4.1
conda init bash
conda shell.bash activate torchtune

rm /fs/class-projects/fall2024/cmsc473/c473g002/models/Meta-Llama-3.1-8B-Instruct/checkpoints/*.pt
tune run lora_finetune_single_device --config llama3_1/8B_lora_single_device
# tune run lora_generate --config lora_generation