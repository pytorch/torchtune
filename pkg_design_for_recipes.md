# Package Design (Recipes-specific)

As brought up in [#394](https://github.com/pytorch-labs/torchtune/issues/394),
our packaging needed to be fixed and fast.
1. _scripts needs to be a proper entrypoint in the `torchtune` package: resolved
in [this commit](https://github.com/pytorch-labs/torchtune/commit/5ae616964546813f23f8d9b1beaae06d3877bd1e).
2. `recipes` should not be an importable module in the `torchtune` package.
3. `tests` should not be an importable module in the `torchtune` package.

Specifically for 2, we were relying on being able to import things from a package (`recipes`)
that *should not* have been importable and hardcoding values to point to that package.

### The Journey of 10k (or, like, 4) Questions

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

Which means the recipe file and config file will need to be included in the final `torchtune` package.
That's fine, according to Python's "setuptool", we can [include files without allowing users to import directly](https://setuptools.pypa.io/en/latest/userguide/miscellaneous.html#controlling-files-in-the-distribution),
but

3. How will users be able to tell from the CLI what built-in recipes/configs they can use?

Now that the `recipes` folder is not a package, there's no way to do `from recipes import list_recipes`. This function
is necessary for running both `tune ls` and `tune cp`. It's bad UX to force a user to consult the web documentation
to do something that's a core functionality. (See [my previous RFC]())

The list and copy functionality is based on a hardcoded list and dictionary that we maintain in the recipes directory.
We can easily move this functionality to the core `torchtune` directory, thereby making these functions importable.

### Explicit Proposal

1. Remove `__init__.py` file from `recipes/` directory, unregistering it as a [package](https://setuptools.pypa.io/en/latest/userguide/package_discovery.html#finding-simple-packages)
2. Add a [`MANIFEST.in`](https://setuptools.pypa.io/en/latest/userguide/miscellaneous.html#controlling-files-in-the-distribution) file
to the base directory that explicitly adds all files in recipes to the package. (This is the standard way to include files outside of a package.)
3. Move `list_recipes` and `list_configs` functions to `torchtune/__init__.py`
4. Move `interfaces.py` to `torchtune/_recipe_interfaces.py`

**Why are you moving `interfaces.py` under the `torchtune` directory?** Nothing in `recipes/` should be importable.
Recipe interfaces are helpers for setting up different recipes. As such, people should be able to import and use as needed, therefore,
it should be included underneath the `torchtune` directory.

### Alternatives Considered

#### Packaging `recipes` under the `torchtune` directory

**Pros**: This would accomplish our goal to include the recipes so that users can utilize list and copy, and we could set it up so that
it's not a module and therefore would not be importable.

**Cons**: The biggest drawback is in the mental model of recipes vs. torchtune components. [Recipes should serve as examples](https://pytorch.org/tutorials/recipes/recipes_index.html)
and users should be able to take a quick look and extend or modify them. As such, including recipes under the torchtune directory
would, in my mind, only serve to confuse users and make them harder to find.

#### Moving `recipes` to its own repository

**Pros**: This would definitely accomplish the goal of not packaging recipes with our `torchtune` package and would
be easy way to allow a bunch of people to hack onto and add their own recipes.

**Cons**: We would have to maintain a central registry of all recipes/configs we support, additional complexity of
copying these files. In addition, there'd be a slight mental tax from forcing users to go to two different repositories
to look at the code vs. examples of using the code.

#### Deleting all of `torchtune`

**Pros**: No packaging issues, rate of bugs drops to 0%

**Cons**: Doesn't exactly make TorchTune a successful offering
