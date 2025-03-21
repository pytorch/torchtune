#!/bin/bash

#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --mem=32gb
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --time=24:00:00

module load cuda/12.4.1
conda init bash
conda shell.bash activate torchtune

tune run lora_finetune_single_device --config recipes/configs/guardian_models/8B_lora_5000.yaml