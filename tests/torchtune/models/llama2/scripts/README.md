# Verifying Correctness against Reference Implementations

This repository puts a high bar on correctness and testing. To make sure our model and
module implementations are correct, we compare our implementation against reference implementations
where possible. This folder contains scripts used for these comparisons.


## Running the scripts

You can run the scripts using the following command as an example.
Each script should print out the value being used in the associated unit tests.

```
python3 -m tests.llm.llama2.scripts.compare_attention
```
