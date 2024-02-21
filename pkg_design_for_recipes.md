# Package Design (Recipes-specific)

Distinction between core recipes and external recipes

vital for the community to be contributing their own recipes in order for this project to take off

### Questions

1. Do we want people to be able to run `import recipes`?

This is a proxy for asking: "Do we want to package `recipes` as an importable package with TorchTune?"

I think the answer to this one is likely no. According to our documentation, recipes are
"the primary entry points for TorchTune users. These can be thought of as end-to-end pipelines for training
and optionally evaluating LLMs." Another way of thinking about this is that the core TorchTune package
has components and models that the recipes **use**.

However, this doesn't end the conversation b/c if we don't package recipes...

2. How do we make sure users can access recipes without being able to import them directly?

We still want to enable the following workflow:
```tune run full_finetune alpaca_llama2_full_finetune.yaml``

Which means users will have to be able to access the `full_finetune.py` recipe and the `alpaca_llama2_full_finetune.yaml`
files.

3. But users still need to be able to quickly know what the built-in recipes are and copy them, how?

Now that the `recipes` folder is not a package, there's no way to do `from recipes import list_recipes`. This function
is necessary for running both `tune ls` and `tune cp`.

**There should be :**

4. What ....?

### Proposal
