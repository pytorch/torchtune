# launch with:
# launch launch_main.sh --classical_logfile_names --gpu_type rtxa5000 --mem 30

tune run lora_finetune_single_device --config recipes/configs/guardian_models/1B_lora_2500.yaml
tune run lora_finetune_single_device --config recipes/configs/guardian_models/1B_lora_5000.yaml
tune run lora_finetune_single_device --config recipes/configs/guardian_models/1B_lora_7500.yaml
tune run lora_finetune_single_device --config recipes/configs/guardian_models/3B_lora_2500.yaml
tune run lora_finetune_single_device --config recipes/configs/guardian_models/3B_lora_5000.yaml
tune run lora_finetune_single_device --config recipes/configs/guardian_models/3B_lora_7500.yaml
tune run lora_finetune_single_device --config recipes/configs/guardian_models/8B_lora_2500.yaml
tune run lora_finetune_single_device --config recipes/configs/guardian_models/8B_lora_5000.yaml
tune run lora_finetune_single_device --config recipes/configs/guardian_models/8B_lora_7500.yaml
