## FSDP2 Recipes

This directory contains distributed training recipes for LoRA and QLoRA using [FSDP2](https://github.com/pytorch/pytorch/issues/114299).
Currently FSDP2 is only available in PyTorch nightly releases.

To set up your environment to run these recipes, you should first install torchtune dependencies,
then install PyTorch nightlies. E.g.

```
pip install torchtune
pip3 install --upgrade --pre torch --index-url https://download.pytorch.org/whl/nightly/cu124
```
