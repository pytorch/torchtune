# Evalz Are Hard

Enjoy the mandatory reading, friends.

***
* Context
    * What are the eval options today?
    * How are other LLM tools handling evaluation?
* Current Eval Solution
    * Overview
    * Limitations
* Proposal
* Alternatives
* Appendix
***

## Context

Evaluations of LLMs are crucial for assessing their performance and validating new approaches.

This is important because after someone finetunes an LLM, they need to determine if it's any good. However, the definition
of "good" can vary depending on the reasons why a person might be finetuning a model. There are several options:
1. A "general" LLM, examples being [Phi 2 Super](https://huggingface.co/abacaj/phi-2-super), [Zephyr](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta), [MobilLlama](https://huggingface.co/MBZUAI/MobiLlama-05B)
2. A "specific" LLM, examples being [ChatMusician](https://huggingface.co/m-a-p/ChatMusician), [StarCoder](https://huggingface.co/bigcode/starcoder2-15b)

**What are the eval options today?**

> The following lists are not exhaustive

OSS Code-Based Options
- [EleutherAI LLM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)
- Stanford's Holistic Evaluation of Language Models [(HELM)](https://github.com/stanford-crfm/helm)
- [Mosaic Eval Gauntlet](https://www.mosaicml.com/llm-evaluation)

OSS Platform-Based Options
- [HuggingFace OpenLLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
- [OpenCompass](https://opencompass.org.cn/home)

Corporate Solutions
- [Arize AI](https://arize.com/blog-course/llm-evaluation-the-definitive-guide/)
- [HoneyHive](https://www.honeyhive.ai/evaluation)


**How are other LLM training tools handling evaluation?**

[Llama Recipes](): Includes a separate eval/ directory that contains a README with instructions on downloading
EleutherAI LLM Evaluation Harness and running evals. It also includes a bash script that will call the eval
script to properly execute all benchmarks in the OpenLLM Leaderboard.

[Lit-GPT](): Includes a separate eval/ directory. They utilize EleutherAI LLM Evaluation Harness, but the dependency
is included in `requirements-all.txt` instead of `requirements.txt`.

[Llama Factory](): Custom built MMLU evaluations. Strange.

[HuggingFace](): Makes you build your own; however, if you upload your model in a safetensor format you can easily
run it on their OpenLLM Leaderboard with the click of a button.

[Axolotl](): No eval story - smart, LLM eval is a confusing, hellish nightmare and I wish I could also ignore it.

### Current Eval Solution

#### Overview

@rvarm1 did some great work to get a initial evaluation script up and running.
The script is accessible through the tune CLI and connects to the EluetherAI LLM
Evaluation Harness. It supports full finetuned Llama2 7B models and LoRA finetuned
Llama2 7B models and any number of individual tasks through the Eval Harness.

```bash
tune eval \
    --model llama2_7b \
    --model-checkpoint /tmp/llama2-finetuned.pt \
    --tokenizer llama2_tokenizer \
    --tokenizer-checkpoint /tmp/tokenizer.model \
    --tasks hellaswag truthfulqa \
```

#### Limitations of Current Eval Solution

1. It is currently broken.
```bash
$ tune eval --help
Traceback (most recent call last):
  File "/home/jrcummings/.conda/envs/tt-test-2/bin/tune", line 8, in <module>
    sys.exit(main())
             ^^^^^^
  File "/home/jrcummings/projects/torchtune/torchtune/_cli/tune.py", line 108, in main
    runpy.run_path(str(cmd), run_name="__main__")
  File "<frozen runpy>", line 291, in run_path
  File "<frozen runpy>", line 98, in _run_module_code
  File "<frozen runpy>", line 88, in _run_code
  File "/home/jrcummings/projects/torchtune/torchtune/_cli/eval.py", line 239, in <module>
    choices=models.list_models(),
            ^^^^^^^^^^^^^^^^^^
AttributeError: module 'torchtune.models' has no attribute 'list_models'
```

2. The EleutherAI LLM Evaluation package (lm_eval) includes dependencies on both HuggingFace's `transformers`
and `accelerate` libraries.

In addition to being (respectfully) a mess of a library, `transformers`
comes in at 82MB - rather large. And if we were goign to rely on `transformers`
anyways, then what is the difference between us and Axolotl?

3. No custom dataset support

Since we just directly integrate with EleutherAI Eval Harness, we have no control over people testing
on individual datasets they may want. If you want to clone EleutherAI's harness, you can add a new task,
but requests for random new tasks to be added are addressed at the whim of the developers.

4. Slow

Currently only runs on CPU or single GPU and is somehow also slower than vanilla EleutherAI Eval.

### Proposal

We need our evals to be comprehensive enough to accurately judge "general" models, flexibile enough to
judge "specific" models, not introduce cumbersome dependencies, and fast.

```
$ tune eval --help
Evaluate your LLM.

USAGE:
  tune eval [OPTIONS]

COMMANDS:
  ls                      List all possible tasks to evaluate on.

OPTIONS:
  --base-model            Base model to use for eval
  --model-checkpoint      Path to model checkpoint
  --tokenizer             Tokenizer to use for eval
  --tokenizer-checkpoint  Path to tokenizer checkpoint
  --tasks                 Comma-separated list of tasks, or suite of tasks
  --num-gpus              Number of GPUs to use in order to run eval, defaults to 0
  --batch-size            Batch size of samples
  --num-fewshot           Number of examples in few-shot context
  -h, --help              Show help message and exit

EXAMPLES:
  # Run task with EleutherAI LLM Evaluation Harness
  $ tune eval \
    --base-model llama2_7b \
    --model-checkpoint <PATH> \
    --tokenizer llama2_tokenizer \
    --tokenizer-checkpoint <PATH> \
    --tasks bigbench
  Error: Task `bigbench` only available with EleutherAI Eval Harness. Install with `pip install lm-eval`.

  # Run eval using multiple GPUs
  $ tune eval \
    --base-model llama2_7b \
    --model-checkpoint <PATH> \
    --tokenizer llama2_tokenizer \
    --tokenizer-checkpoint <PATH> \
    --tasks hellaswag \
    --num-gpus 4

  # Specify a standard suite of tasks
  $ tune eval \
    --base-model llama2_7b \
    --model-checkpoint <PATH> \
    --tokenizer llama2_tokenizer \
    --tokenizer-checkpoint <PATH> \
    --tasks open-llm-leaderboard

  # Evaluate on (most) HuggingFace Hub dataset
  $ tune eval \
    --base-model llama2_7b \
    --model-checkpoint <PATH> \
    --tokenizer llama2_tokenizer \
    --tokenizer-checkpoint <PATH> \
    --tasks m-a-p/MusicTheoryBench

Need to see all built-in tasks you can use? Try running `tune eval ls`.
```

In addition to the eval-specific engineering effort above, we need the following:
1. A guide on which tasks to use for evaluation - In my readings (see Appendix), it's clear
that people have a hard time understanding benchmarks and what they mean, besides the fact that in general,
a higher number is better. We should include a tutorial on how to pick the best benchmark for the model.

2. Better chat capabilities - "Playing around" with a finetuned model can often give "good enough" insights
into how a model is working. Running benchmarks can take a long time and don't always capture the information
users really want to know. (Cite source from twitter, LocalLlama, etc.)

**Considerations**
- Much of the work to support arbitrary HuggingFace Hub datasets will have to be in conjunction with the current
work Rafi is doing to enable a similar thing in our training pipeline
- We will start by only supporting an accuracy metric mirroring EleutherAI Eval Harness

### Alternatives

**Integration with HELM**
* Pros:
  * Standard format of dataset for each interop
  * Collection of metrics beyond accuracy
* Cons:
  * HELM *also* depends on `transformers` and `accelerate` + a bunch more
  * Anecdotally, @msaroufim had a bad experience trying to integrate with them while working on [Neurips LLM Efficiency Challenge](https://github.com/llm-efficiency-challenge/neurips_llm_efficiency_challenge)

**Build our own evaluation framework**
* Pros:
  * Gives us total control over dependencies
  * Better experience for TorchTune - we won't have to adapt to other evaluation frameworks quirks
* Cons:
  * Minimum 6+ months worth of work

**Delete TorchTune**
* Pros:
  * No code = no bugs
* Cons:
  * Doesn't solve the problem

### Appendix

- [Current state of benchmarking](https://www.confident-ai.com/blog/the-current-state-of-benchmarking-llms#:~:text=A%20benchmark%20dataset%20is%20a,natural%20language%20understanding%20and%20Q%26A)
- [How to evaluate LLMs](https://www.analyticsvidhya.com/blog/2023/05/how-to-evaluate-a-large-language-model-llm/)
- [Mosaic on evaluating LLMs](https://www.mosaicml.com/llm-evaluation)
