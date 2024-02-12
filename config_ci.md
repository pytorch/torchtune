# [RFC] Config CI and Integration Tests
As we start to add more user-facing configs, we should think about how we want
to continuously test them so they can be run OOTB without breaking as the
codebase evolves.

Configs are an entry point into the corresponding recipe. They’re not code, so
what does it mean to test them? Our guarantee to the user should be that they
can run the recipe with that specific combination of parameters and get an
expected result. Thus, a proper test would be that the config runs the recipe
from start to finish without failing. The question is more, to what extent do we
run the recipe - full training run with loss curves and eval? Or just a few
steps as a sanity check?

The other dimension to this is the environment that the recipe is run in, i.e.,
can the recipe be run on CPU, on GPU, on multiple GPUs, with FSDP, etc. Although
these can be specified by the config, testing these environments is more akin to
testing the recipe itself.

Thus, instead of testing every combination of user config x environment, which
can quickly blow up, I propose we add different testing environments for each
recipe and only test user configs once in some canonical “debug” mode to make it
lightweight. Every recipe will then have (# of environments + # of user configs)
tests instead of (# of environments x # of user configs) tests.

## What we already have

Recipes: full_finetune, lora_finetune, alpaca_generate
Recipe Integration Tests: full_finetune (single CPU, non-distributed), alpaca_generate (single CPU, non-distributed)
Configs: alpaca_llama2_full_finetune.yaml, alpaca_llama2_lora_finetune.yaml

We are only testing recipes on single CPU, non-distributed environments. We need
to consider all of the following:
- Single CPU, non-distributed
- Multi CPU, distributed
- Single GPU, non-distributed
- Single GPU, distributed
- Multi GPU, distributed

For configs, we are already testing that they can be instantiated with the
corresponding recipe dataclass, but we are not actively testing that they can
run the recipe. Note that multi-gpu distributed CI is not available right now,
but something we should be setting up to add easily.

## What we need to add
- Recipe integration test for LoRA (in progress) and config for alpaca generate
(needs review)
- Pre-defined environment args that we can loop through for each recipe
integration test that covers the above environments. We can continue checking
loss as we do for the existing integration tests
- Add a “debug” mode that runs any config with lightweight options. These should
be non-behavior altering overrides, such as number of epochs, number of steps,
smaller checkpoint, etc. We don’t want to modify the config in a way that
changes the original intended behavior otherwise the test is not valid
- Set up a cadence for running these tests in GitHub CI (i.e., with every PR, or
once every few days). The config integration tests could be run more frequently
if debug mode is lightweight enough and the recipe integration tests could be
less often since they are resource intensive
