## Verifying correctness
This directory compares the current implementation of `mistral` to the reference implementation at https://github.com/mistralai/mistral-src/blob/main/one_file_ref.py. Additionally, `torchtune.models.mistral._component_builders.mistral_mlp` is compared in `tests.torchtune.models.mistral.scripts.compare_feed_forward.py`

Since `torchtune.models.mistral` shares nearly all components with `torchtune.models.llama2`, please see `tests.torchtune.models.llama2.scripts` for comparison scripts for individual components.

## Running the scripts

You can run the scripts using the following command as an example.
Each script should print out the value being used in the associated unit tests.

```
python3 -m tests.torchtune.models.mistral.scripts.compare_mistral
```
