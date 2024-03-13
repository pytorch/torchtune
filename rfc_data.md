# Datasets in TorchTune

## Motivation
When fine-tuning LLMs, there are three main areas where users can influence the
end result and final performance: the model architecture, fine-tuning hyperparameters,
and the dataset. For open source users, many who are hobbyists and hackers, modifying
the architecture itself or experimenting with hyperparameters is unfeasible either
due to lack of expertise or lack of resources. Most often they will just use the
best pre-trained model for their use case given the plethora of options with some
default hyperparameters. The most common user journey for fine-tuning an LLM is
to quickly bootstrap a pre-trained model with their custom dataset(s). This means
that data is OSS users’ primary means of controlling the model. It then becomes
imperative that we curate the smoothest user experience with world-class API design
for plugging in custom datasets to fine-tune with TorchTune. This document will
overview the abstractions needed to support custom datasets in TorchTune, a high
level API design, and what the user journey with these components looks like in
TorchTune.

“However, the real challenge lies in preparing the data. A massive wiki of product
documentation, a thousand PDFs of your processes, or even a bustling support forum
with countless topics - they all amount to nothing if you don't have your data in
the right format. Projects like Dolly and Orca have shown us how enriching data
with context or system prompts can significantly improve the final model's quality
[...] Personally, I mostly utilize the #instruction, #input, #output format for
most of my fine-tuning tasks. So, shaping your data in the correct format is, without
a doubt, the most difficult and time-consuming step when creating a Language Learning
Model (LLM) for your company's documentation, processes, support, sales, and so
forth.” - user on r/LocalLLaMA

## Existing OSS Solutions

### HuggingFace

Links:
- Load_dataset: https://huggingface.co/docs/datasets/v1.12.0/loading.html
- Cloud storage: https://huggingface.co/docs/datasets/en/filesystems
- Streaming: https://huggingface.co/docs/datasets/en/stream
- Preprocessing: https://huggingface.co/docs/datasets/en/process

HuggingFace does a great job provide an incredible breadth of utilities that allow
users to load in any local file, remote huggingface dataset, or dataset on cloud
storage and provides basic preprocessing functionality (including shard, map, multiprocess,
interleave, concatenate, filter, split, etc). These datasets can also be streamed
so it is not all downloaded at once. If a user wants to use a dataset present on
the huggingface hub, they should be able to leverage the full functionality of this
ecosystem.

### Axolotl
Link: https://github.com/OpenAccess-AI-Collective/axolotl?tab=readme-ov-file#dataset

One aspect Axolotl does really well is maintaining a suite of various prompt templates
that will automatically tokenize a data source into the template, with some configurability
from the YAML config. This covers everything from instruction to conversations to
raw text corpus. The awesome part is that you can combine multiple datasets all
from the config (I think) and specify their template.

## TuneDataset
The general flow of loading a dataset from data file to tokenized prompt and label consists of:
- (optional) Packing the dataset offline and caching it / saving it for use in current and future runs
- Get a single sample
- Apply user transform to any of the columns - this could also be converting from one template to another, as is the case for SlimOrca: sharegpt -> llama2 chat
- Format into provided template using PromptTemplate’s methods
- Tokenize with provided tokenizer
- Collate tokenized output - padding, modify masks for packing, etc
Since each step uses a user-provided component, every step is fully customizable
using standard components in the library or custom components from the user, provided
that it follows a particular interface.
```
from torch.utils.data import Dataset

class TuneDataset(Dataset):
    def __init__(
        self,
        source: str,
        column_map: Optional[Dict[str, str]],
        transform: Optional[Callable],
        template: Union[PromptTemplate, str],
        tokenizer: Tokenizer,
        collator: Callable,
        packing: bool = False,
    ) -> None:
        # Set all attributes
        ...
        self._data = get_data(source)
        if packing:
            self._data = sample_packing(self._data)

    def __getitem__(self, index: int) -> Tuple[List[int], List[int]]:
        # Get a row
        sample = self._data[index]
        # Apply any transforms
        sample = self._transform(sample)
        # Format into template
        prompt = self._template.format(sample, self._column_map)
        # Tokenize
        encoded_prompt = self._tokenizer.encode(prompt)
        labels = self._tokenizer.encode(sample["output"])
        # Collate
        collated_prompt, collated_labels = self._collator(encoded_prompt, labels)

        return collated_prompt, collated_labels
```

### Source
Datasets we should support are:
- HuggingFace Hub, specified with the standard username/dataset_name. A common specification is the data_files parameter, which we could support by letting users continue the HF path: username/dataset_name/files/user/wants. Ex: allenai/c4/en/c4-train.0000*-of-01024.json.gz.
- Local files, specified with a simple path string
- Remote files via HTTPS, specified with a URL
All should be readily supported by using HF’s load_dataset() API, including support
for JSON, JSONL, text, arrow, parquet, CSV, etc. We can provide a convenient utility
get_data or similar that handles loading the data from the right location given
a simple path string.

### Transform
This is an optional parameter that users can provide to do any preprocessing on
their data before templating. The most immediate use case would be to convert from
one prompt template to another. For example, in TorchTune’s SlimOrcaDataset implementation,
we convert from ShareGPT Conversation template to llama2 chat. We can pass in a
method that does this as the transform here. Another example is llama recipes’ grammar
dataset which requires splitting and processing the strings directly before templating.

### Template
This class should handle massaging the input data columns/fields into a predefined
prompt template. We will support the most common ones: instruct, chat, sharegpt,
etc. This is akin to Axolotl’s Prompters.
```
class PromptTemplate:
    system = "This is an example system prompt: {input} {output}"
    def format(self, sample: Dict[str, str], column_map: Dict[str, str]) -> str:
        return self.system.format(input=sample[column_map["input"]], output=sample[column_map["output"]])
```

One challenge is mapping the input data columns into the correct fields for the
prompt template, given that the input dataset could have different columns, formats,
etc. We can allow users to provide the column to field mapping for the prompt template.

We could also just make the prompt templates plain strings and use the native string
format method. However, for cases like Alpaca where we want to handle with / without
input, we need a bit more functionality. Using a class is also more extensible.

### Collator
While transforms process the sample BEFORE templating and tokenization, collators
include any data utils that process the sample AFTER tokenization. The primary util
is padding to a max length, which we can repurpose utils.padded_collate for. We
can also make the train_on_input functionality a collator.

Open question: should we ditch using the collate_fn kwarg in DataLoader in favor
of coupling the collator with TuneDataset? What’s the tradeoff here?

### Sample packing
Packing involves stuffing multiple data samples in the input upto the max sequence
length to make full use of the context window. The algorithms to achieve this are
the same ones used to solve the classic bin packing problem. Unfortunately, these
algorithms require knowledge of the full distribution of sample lengths, meaning
we need to iterate through the entire dataset before we can begin sample packing.
However, packing results in faster training since the model can see more samples
at a time, but just requires some additional processing of the dataset. There are
two approaches to sample packing:
* Offline bin-packing: take a dataset and iterate through the sample lengths. Use
one of the algorithms to pack the dataset and either cache it or upload to HF hub.
Axolotl uses the first-fit-decreasing algorithm.
    - Tradeoff: TTFB is much longer, need to figure out caching which may require
    significant storage space. But overall fine-tuning is faster.
* Online greedy: instead of packing offline, do it as we iterate through the dataset.
We don’t see all the sample lengths at once so we cannot use a bin-packing algorithm;
instead greedily pack the context window as we iterate. This is done by llama recipes.
    - Tradeoff: Faster TTFB, no caching of entire dataset. Slower fine-tuning.

The approach we take could dictate the design of sample packing API. Options are:
a separate offline script, a boolean flag in the TuneDataset class, an entirely
different dataset class.

On masking for packed samples: Because we have multiple samples in the same input,
we need to tell the model to not attend to other irrelevant samples. Generally,
the guidance in OSS has been that the EOS token between samples is sufficient because
samples are uncorrelated. A more prudent approach would be to use an integer mask
that labels which sample is which to prevent cross-attending. This is something
we should support, and may involve a custom collator to handle this. More discussion
can be found here: https://github.com/facebookresearch/llama-recipes/issues/341

## Configuring datasets in configs
As proposed, TuneDataset requires other higher level components as parameters, such
as PromptTemplate, Tokenizer, Callables, each with their own keyword arguments.
This is problematic because we purposely restricted nested components in the config
system to avoid meta-programming via yaml file. In other words, you cannot configure
a dataset directly with TuneDataset from the config. There is an alternative approach
that still enable configurable datasets without compromising on the nested components
principle.

### Builders
Common datasets will have builder functions with flat params that can be easily
specified with one component in the config file. We can also provide a builder function
for custom datasets with limited functionality. This may require some registry object
that contains the mapping from string to common prompt templates or collators and
an associated getter function. While this is not ideal and something that the current
config.instantiate API was originally trying to bypass, we can keep it fairly contained
to just prompt templates and basic collators/transforms, for example.
```
def build_dataset(
    source: str,
    column_map: Dict[str, str],
    tokenizer: Tokenizer,  # do we need a mapping for all tokenizers or rely on partial instantiation?
    template: str,  # Choose from common templates in library
    pad: bool = True,
    packing: bool = False,
) -> TuneDataset

# In the yaml config - we cannot do nested components
dataset:
  _component_: torchtune.datasets.TuneDataset
  template:
    _component_: torchtune.datasets.prompt_templates.AlpacaInstructTemplate
    ...

# Instead, specify a builder
dataset:
  _component_: torchtune.datasets.build_dataset
  source: tatsu-lab/alpaca
  tokenizer: # TBD how to handle tokenizers
  template: instruct
  pad: True
  packing: True
```

Options that require more involved customization, such as custom transforms, will
require a user to create custom builder functions that they can then specify via
the config. Since transforms will typically require the user to write code anyway,
adding the responsibility of creating a builder is not too burdensome. I contend
that this extra user burden is worth intentionally restricting nested components
to prevent intrusive configs.
```
def my_custom_dataset(
    my_param: str,
) -> TuneDataset:
    # Logic to create custom dataset here with exact components
    ...

dataset:
  _component_: torchtune.datasets.my_custom_dataset
  my_param: hello
```

## Supported Datasets
Here, we use our flagship supported datasets as examples to exhibit the versatility
and extensibility of the TuneDataset API design.

### Alpaca
Instruct tasks. Good example of straightforward dataset that doesn’t require transforms,
column mapping, template conversion.
- Source: tatsu-lab/alpaca
- Source Template: instruction, input, output columns
- Target Template: https://github.com/tatsu-lab/stanford_alpaca#data-release - is this the standard instruct template?
- Collator: pad, train_on_input
```
def alpaca_dataset(tokenizer: Tokenizer, use_clean: bool = False) -> TuneDataset:
    return TuneDataset(
        source="yahma/alpaca-cleaned" if use_clean else "tatsu-lab/alpaca",
        template=AlpacaInstructTemplate(),
        tokenizer=tokenizer,
        collator=pad_and_train_on_input,
    )
```

### SlimOrca
Conversational/chat tasks. Good example of utilizing the transform for template
conversion.
- Source: Open-Orca/SlimOrca-Dedup
- Source Template: sharegpt conversation
- Target Template: In TorchTune, we convert to llama2 chat using _generate_prompt_label
- Collator: pad
```
def slim_orca_dataset(tokenizer: Tokenizer) -> TuneDataset:
    return TuneDataset(
        source="Open-Orca/SlimOrca-Dedup",
        transform=convert_sharegpt_to_llama2_chat,
        template=Llama2ChatTemplate(),
        tokenizer=tokenizer,
        collator=pad,
    )
```

### Samsum
Summarization tasks.
- Source: samsum
- Source Template: dialogue and summary columns
- Target Template: Summary - similar to SummarizeTLDRPrompter in Axolotl. Also see the version in llama recipes
- Collator: pad
```
def samsum_dataset(tokenizer: Tokenizer) -> TuneDataset:
    return TuneDataset(
        source="samsum",
        template=SummarizeTemplate(),
        tokenizer=tokenizer,
        collator=pad,
    )
```

### Grammar
Primarily for grammatical error correction tasks. In llama recipes, two datasets
are used. For now, we will use one dataset until the multi-dataset UX is more thought
out.
- Source: jhu-clsp/jfleg + liweili/c4_200m
- Source Template: jfleg requires preprocessing, c4 just needs to map the columns
- Target Template: A grammar correction template? Doesn’t seem to be a standard. https://github.com/facebookresearch/llama-recipes/blob/main/src/llama_recipes/datasets/grammar_dataset/grammar_dataset.py#L50
- Collator: pad
```
def grammar_dataset(tokenizer: Tokenizer) -> TuneDataset:
    return TuneDataset(
        source="liweili/c4_200m",
        column_map={"sentence": "input", "correction": "output"},
        template="Correct this to standard English: {sentence}\n---\nCorrected: {correction}",
        tokenizer=tokenizer,
        collator=pad,
    )
```

### Open questions
- TuneDataset has high level components as parameters. How can users specify this
in a config if nested configs / recursive instantiation are not enabled? How can
we make builders general enough that users can easily specify multiple datasets
in a config without using nested components?
- Should we use the collate_fn keyword in DataLoader for collators, or keep it coupled with TuneDataset?
- Should we perform sample packing offline or do an online ad-hoc approach?
- Are there other abstractions we need to consider, or abstractions we should simplify? (i.e., just using str instead of a PromptTemplate class)
