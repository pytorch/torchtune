#!/bin/bash

#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --mem=32gb
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --time=24:00:00

module load cuda/11.8.0
conda init bash
conda shell.bash activate torchtune

tune run lora_finetune_single_device --config recipes/configs/guardian_models/qwen_1B_lora_2500.yaml