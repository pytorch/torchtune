#### Context
What is the purpose of this PR? Is it to
- [ ] add a new feature
- [ ] fix a bug
- [ ] update tests and/or documentation
- [ ] other (please add here)

Please link to any issues this PR addresses.

#### Changelog
What are the changes made in this PR?
*

#### Test plan
Please make sure to do each of the following if applicable to your PR. If you're unsure about any one of these just ask and we will happily help. We also have a [contributing page](https://github.com/pytorch/torchtune/blob/main/CONTRIBUTING.md) for some guidance on contributing.

- [ ] run pre-commit hooks and linters (make sure you've first installed via `pre-commit install`)
- [ ] add [unit tests](https://github.com/pytorch/torchtune/tree/main/tests/torchtune) for any new functionality
- [ ] update [docstrings](https://github.com/pytorch/torchtune/tree/main/docs/source) for any new or updated methods or classes
- [ ] run unit tests via `pytest tests`
- [ ] run recipe tests via `pytest tests -m integration_test`
- [ ] manually run any new or modified recipes with sufficient proof of correctness
- [ ] include relevant commands and any other artifacts in this summary (pastes of loss curves, eval results, etc.)

#### UX
If your function changed a public API, please add a dummy example of what the user experience will look like when calling it.
Here is a [docstring example](https://github.com/pytorch/torchtune/blob/6a7951f1cdd0b56a9746ef5935106989415f50e3/torchtune/modules/vision_transformer.py#L285)
and a [tutorial example](https://pytorch.org/torchtune/main/tutorials/qat_finetune.html#applying-qat-to-llama3-models)

- [ ] I did not change any public API
- [ ] I have added an example to docs or docstrings
