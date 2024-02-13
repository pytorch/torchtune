---
language:
- en
pipeline_tag: text-generation
inference: false
tags:
- facebook
- meta
- pytorch
- llama
- llama-2
---

## Goal of this file

TODO: Make it clear that this uploaded model was finetuned using torchtune, how people can finetune and how people can load this model for an inference

## Download

Need a `pth` file which would be a torch native checkpoint

```python
import torch
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
```

## Dependencies

```
sudo apt-get install git-lfs

```
pip install torch torchvision huggingface_hub
```

## Upload

```bash
python script_name.py \
  --repo_name your_repo_name \
  --model_path path/to/model.pt \
  --hf_username your_hf_username \
  --hf_token your_hf_token \
  --private
```

Default `README.md` is the torchtune one
Default `LICENSE` is the Llama2 one

```bash
python upload.py \
  --repo_name tune \
  --model_path <PATH_TO_PTH_FILE> \
  --hf_username <YOUR_HF_USERNAME> \
  --hf_token <YOUR_HF_TOKEN> \
  --private
```

## Eval
