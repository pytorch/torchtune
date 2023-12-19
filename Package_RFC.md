# [RFC] torchtune & recipes packaging

Creating this issue to discuss recommended approaches for packaging both torchtune core library components and recipes.

**TLDR**: We propose a single package for both the core torchtune library[^1] and recipes, with (a) two different pip install modes and (b) decoupled CI across the core library and recipes.

The primary way that users are interacting with current fine-tuning libraries is though git clone, as shown  [here](https://docs.google.com/document/d/1qP-QW7hDTjRI1AgHNyNTS2i9cZ1g3XmgVCoQW8879cA/edit#heading=h.2qx7miaohxld). This encourages users to edit the library directly but makes long term maintainability of the users code difficult as they have to manage merging changes from the source library. Encouraging users to interact with the library primarily through pip packages, as is standard for PyTorch libraries, encourages users to have cleanly separated code from the library code. The goal of our package and library design should be to allow the kind of free code access that interacting directly with a cloned repo provides while still allowing our code to work as a code versioned library. Below we propose a package setup to meet users where they're currently at while also facilitating simple "pip install" workflows.


### Goals and constraints

To support our [users](https://github.com/pytorch-labs/torchtune/pull/54/files?short_path=dda34b2#diff-dda34b2e50075ce560d9f896ac2834b74c9e851a94f35fdfec031da03efa22c3) different flows we need to satisfy the following constraints:

1) Support CLI access to recipes for User 1 based on just "pip install torchtune" and docs
2) Support copy and pasting recipes for User 2. These must be version matched to the package so they don't get out of date.
3) Support "torchtune" component library that is available for import and used by User 3 and available for import. This should be stable over time and preserve backward compatibility.
4) Recipes must be continuously tested to ensure they're not broken in packages
5) Recipes must be accessible from the command line and for "copy paste" but not importable as they're scripts and not functions. We don't want to standardize args available to all recipes so they each need to manage their own.
6) Ecosystem plugins need to be available for users but treated as optional dependencies to not break the core library.


## Proposal: Single package for both core lib and recipes

Naively bundling core + recipes into a single package will not satisfy constraints #2 or #3. So we propose a couple minor modifications to address these points.

1) Use pip's [optional dependencies](https://setuptools.pypa.io/en/latest/userguide/dependency_management.html#optional-dependencies) to address #2.
2) Construct a separate test directory with corresponding CI job for recipes to address #3.
3) Recipes will be included in the package but only available to the CLI. One option is adding them like a non code file like [Manifest.in](https://python-packaging.readthedocs.io/en/latest/non-code-files.html). For Copy/Paste access the recipes will also be available from the docs.
4) Package will include an [entry point](https://packaging.python.org/en/latest/specifications/entry-points/) which will be a script to wrap torchrun and have access to the package resources for recipe access.

Therefore through the CLI you can access the packaged recipes but from python directly you only have access to torchtune.

## Alternatives considered

### Alternative 1: Package core lib only

**Pros**: Keeping recipes in sync is hard, and it's fairly common for things to go out of sync in other libraries. If we set a lower quality bar on recipes, it will make our lives easier.

**Cons**: This doesn't meet constraint #4: users will need to git clone in order to use the recipes. Also User 1 is probably the most likely to use `pip install` over `git clone`, and also probably most likely to want an out-of-the box training script they can run as a one-liner. This solution will not work for their use case.

### Alternative 2: Package core lib and recipes into separate packages

**Pros**: Satisfies all the constraints laid out.

**Cons**: Needlessly complicated? We would need to have e.g. `torchtune` and `torchtune-recipes` with the latter taking the former as a dependency. This can provide a confusing and convoluted user experience.

[^1]: By core library we generally mean components that are imported in a standalone fashion. This can include nn.Modules, transforms, dataset classes, and general training utilities. By recipes we generally mean a Python script that is executable via the command line (ideally still a relatively generalized one, hence the choice of "recipe" over "example"). To be even more explicit, as of today (commit 505aede) the core library is everything under torchtune/, and recipes are everything under recipes/
