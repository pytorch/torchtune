# Contributing to torchtune
We want to make contributing to this project as easy and transparent as possible.

&nbsp;

## Dev install
You should first [fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo) the torchtune repository
and then clone your forked repository. Make sure to keep your fork in sync with the torchtune repository over time.

```git clone https://github.com/<YOUR_GITHUB_USER>/torchtune.git```

Then navigate into the newly cloned repo and install dependencies needed for development.

**Step 1:** [Install PyTorch](https://pytorch.org/get-started/locally/). torchtune is tested with the latest stable PyTorch release as well as the preview nightly version.


**Step 2:** Install all the additional dependencies and dev dependencies in the local repo:

```
cd torchtune
pip install -e ".[dev]"
```

&nbsp;

## Contributing workflow
We actively welcome your pull requests.

1. Create your new branch from `main` in your forked repo, with a name describing the work you're completing e.g. `add-feature-x`.
2. If you've added code that should be tested, add tests. Ensure all tests pass. See the [testing section](#testing) for more information.
3. If you've changed APIs, [update the documentation](#updating-documentation).
4. Make sure your [code lints](#coding-style).
5. If you haven't already, complete the [Contributor License Agreement ("CLA")](#contributor-license-agreement-cla)

&nbsp;

## Testing
torchtune contains three different types of tests: unit tests, recipe tests, and regression tests. These tests are distinguished by their complexity and the resources they require to run. Recipe tests and regression tests are explicitly marked via pytest.mark decorators and both require S3 access to download the requisite assets.

- **Unit tests**
  - These should be minimal tests runnable without remote access. (No large models, no downloading weights). Unit tests should be under [tests/torchtune](https://github.com/pytorch/torchtune/tree/main/tests/torchtune).
  - All unit tests can be run via ```pytest tests```.
- **Recipe tests**
  - These are relatively small-scale integration tests for running our recipes. These include
  both single-device recipes and distributed recipes. In the latter case, tests should be marked with the `@gpu_test` decorator to indicate how many GPUs they need to run.
  - Recipe tests require remote access as (small) model weights will be downloaded from S3 to run them.
  - Recipe tests are found under [tests/recipes](https://github.com/pytorch/torchtune/tree/main/tests/recipes) and should be marked with the `@pytest.mark.integration_test` decorator.
  - To run only recipe tests, you can run `pytest tests -m integration_test`.
- **Regression tests**
  - These are the most heavyweight tests in the repo. They involve building a full model (i.e. 7B size or larger), then running some finetune and/or evaluation via a combination of tune CLI commands. Whereas an individual recipe test runtime is generally still O(seconds), integration tests should be O(minutes) or greater. Like recipe tests, regression tests also require S3 access.
  - Regression tests are found under [tests/regression_tests](https://github.com/pytorch/torchtune/tree/main/tests/regression_tests) and should be marked with the `@pytest.mark.slow_integration_test` decorator.
  - To run only regression tests, you can use the command `pytest tests -m slow_integration_test`.

Whenever running tests in torchtune, favor using the command line flags as much as possible (e.g. run `pytest tests -m integration_test` over `pytest tests/recipes`). This is because (a) the default behavior is to run unit tests only (so you will miss recipe tests without the flag), and (b) using the flags ensures pytest will automatically download any remote assets needed for your test run.

Note that the above flags can be combined with other pytest flags, so e.g. `pytest tests -m integration_test -k 'test_loss'` will run only recipe tests matching the substring `test_loss`.

&nbsp;

## Updating documentation
Each API and class should be clearly documented. Well-documented code is easier to review and understand/extend. All documentation is contained in the [docs directory](docs/source):

* All files following the pattern `api_ref_*` document top-level APIs.
* All files under the [deep dives directory](docs/source/deep_dives) contain "deep-dive" tutorials
* All files under the [tutorials directory](docs/source/tutorials) contain regular tutorials

Documentation is written in [RST](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html) format.

### Adding a new class/method to the API References
Once you've added an API that is meant to be exposed publically, you should add it to the appropriate rst file. For example, any new API within the [configs/](torchtune/configs)
directory should be added to `api_ref_configs.rst`, [data/](torchtune/data) should be added to `api_ref_data.rst`, [datasets](torchtune/datasets) should be added to
`api_ref_datasets.rst`, and so on. To add, it's as simple as adding the name of the exposed API somewhere in the appropriate RST file.

All code written within the docstring of the class or method will be correctly rendered there.

> Note: Our RST theme expects code to be specified using double backticks instead of single. Eg: ``hidden_dim``. Single backticks will be rendered as italics instead of as "code".

### Adding documentation for a recipe

If you've contributed a new recipe, or you're interesting in adding documentation for an existing recipe, you can add a new page in [the recipes directory](docs/source/recipes). Please refer to existing recipe docpages to understand the format of these documentation pages. Broadly speaking:

- Recipe documentation pages are like beefed up API references for recipes.
- They should have a low noise/information ratio, i.e. information in the recipe documentation page should mostly be relevant for using that recipe.
- Relevant information could include:
  - A cookbook/manual-style description of all the ways in which the recipe can be modified. For instance, does it support different loss functions? If so, describe those loss functions and help a user understand when they might want to use them.
  - Example commands for using and customizing the recipe, particularly w.r.t the specific knobs and levers unique to the recipe.
  - Pre-requisites for the recipe including models and datasets.
  - Reference outputs for a recipe to help a user understand what successful training looks like e.g. loss curves, eval results, generations, etc.
  - References to the appropriate [memory optimization](https://pytorch.org/torchtune/main/tutorials/memory_optimizations.html) features which can be used in the recipe. If you've contributed new memory optimization features which could be used across other recipes, consider adding them to the overview!


Finally, make sure you update the [recipe overview page](docs/source/recipes/recipes_overview.rst), and the [index sidebar](docs/source/index.rst).

### Building docs

All documentation is built for each PR and contains a preview on the PR. However, this takes awhile (~8 minutes) and you should first build docs from your local machine.

From the [docs/](docs) directory:

1. Install dependencies:

```
pip install -r requirements.txt
```

2. Run make command:

```
make html
# Now open build/html/index.html
```

To avoid building the examples (which execute python code and can take time) you
can use `make html-noplot`. To build a subset of specific examples instead of
all of them, you can use a regex like `EXAMPLES_PATTERN="plot_the_best_example*"
make html`.

If the doc build starts failing for a weird reason, try `make clean`.

#### Serving docs locally (if building from a GPU env)

If you're developing locally, you can just open the generated `index.html` file in your browser.

If instead you're using a remote machine, you can use a combination of a simple python HTTP server and port forwarding to serve the docs locally. This allows you to iterate on the documentation much more quickly than relying on PR previews.

To do so, after following the above doc build steps, run the following from the `docs/build/html` folder:

```
python -m http.server 8000 # or any free port
```

This will open up a simple HTTP server serving the files in the build directory. If this is done on a remote machine, you can set up port forwarding from your local machine to access the server, for example:

```
ssh -L 9000:localhost:8000 $REMOTE_DEV_HOST
```

Now, you can navigate to `localhost:9000` on your local machine to view the rendered documentation.

&nbsp;

## Coding Style
`torchtune` uses pre-commit hooks to ensure style consistency and prevent common mistakes. Enable it by:

```
pre-commit install
```

After this pre-commit hooks will be run before every commit.

You can also run this manually on every file using:

```
pre-commit run --all-files
```

&nbsp;

## Best Practices

This section captures some best practices for contributing code to torchtune. Following these will make PR reviews easier.

- **Modular Blocks instead of Monolithic Classes**. Stuffing all of the logic into a single class limits readability and makes it hard to reuse logic. Think about breaking the implementation into self-contained blocks which can be used independently from a given model. For example, attention mechanisms, embedding classes, transformer layers etc.
- **Say no to Implementation Inheritance**. You really don’t need it AND it makes the code much harder to understand or refactor since the logic is spread across many files/classes. Where needed, consider using Protocols.
- **Clean Interfaces**. There’s nothing more challenging than reading through functions/constructors with ~100 parameters. Think carefully about what needs to be exposed to the user and don’t hesitate to hard-code parameters until there is a need to make them configurable.
- **Intrusive Configs**. Config objects should not intrude into the class implementation. Configs should interact with these classes through cleanly defined builder functions which convert the config into flat parameters needed to instantiate an object.
- **Limit Generalization**. Attempting to generalize code before this is needed unnecessarily complicates implementations - you are anticipating use cases you don’t know a lot about. When you actually need to generalize a component, think about whether it’s worth it to complicate a given interface to stuff in more functionality. Don’t be afraid of code duplication if it makes things easier to read.
- **Value Checks and Asserts**. Don’t check values in higher level modules - defer the checks to the modules where the values are actually used. This helps reduce the number of raise statements in code which generally hurts readability, but are critical for correctness.

&nbsp;

## Issues
We use GitHub issues to track public bugs. Please ensure your description is clear and has sufficient instructions to be able to reproduce the issue.

Meta has a [bounty program](https://www.facebook.com/whitehat/) for the safe disclosure of security bugs. In those cases, please go through the process outlined on that page and do not file a public issue.

&nbsp;

## License
By contributing to torchtune, you agree that your contributions will be licensed under the LICENSE file in the root directory of this source tree.

&nbsp;

## Contributor License Agreement ("CLA")
In order to accept your pull request, we need you to submit a CLA. You only need to do this once to work on any of Meta's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

&nbsp;
