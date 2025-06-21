### Core Issues

1. No support for iterative dataset:
   - Dataset has to be fully loaded in memory
   - With map-style, no control over multi-sample operations (e.g. packing or skipping)
   - Map-style is slower
   - No support for streaming

2. No support for weighted dataset:
   - We have it in a single newly added dev recipe/config, but API needs polishing
   - We also support ConcatDataset, but it's map style and there is no weighting

3. No support for on-the-fly data packing:
   - It's done before training, taking a long time for large datasets

### UX Issues

4. Unclear boundaries between HF and torchtune args:

```python
def alpaca_dataset(
    # --- message specific args ---
    train_on_input: bool = True,

    # --- torchtune specific args ---
    tokenizer: ModelTokenizer,
    packed: bool = False,

    # --- HF loading args ---
    source: str = "tatsu-lab/alpaca",
    column_map: Optional[Dict[str, str]] = None,
    split: str = "train",
    **load_dataset_kwargs: Dict[str, Any],

    # --- HF dataset method ---
    filter_fn: Optional[Callable] = None,
) -> Union[SFTDataset, PackedDataset]:
```

5. Lack of dataloader args:
   - Args are scattered in the config
   - Important args are not exposed (e.g. num_workers, pin_memory)

```yaml
dataset:
  _component_: torchtune.datasets.multimodal.the_cauldron_dataset
seed: null
batch_size: 8
shuffle: True
collate_fn: torchtune.data.padded_collate_tiled_images_and_mask
```

6. Different datasets have different arguments due to different message transforms

### Principles

- Common API signatures for all datasets
- Offload what we can to HF datasets methods directly
- Less protagonism from our functions (e.g. config manipulations, instantiation). Not the focus of this diff.

### Proposal

#### Config Example (config.yaml)

```yaml
###########
# Tokenizer
###########
tokenizer:
  _component_: torchtune.models.llama3_2_vision.llama3_2_vision_transform
  path: /tmp/Llama-3.2-11B-Vision-Instruct/original/tokenizer.model
  image_size: 560
  max_seq_len: 8192

##########
# Dataloader
# Consolidate all dataloader args here (currently scattered)
##########
dataloader:
  _component_: torchdata.stateful_dataloader.StatefulDataLoader
  batch_size: 4
  num_workers: 4
  pin_memory: true
  collate_fn: torchtune.data.padded_collate

#########
# Dataset Options
#########

# Option 1: Direct Class Usage (current SFTDataset approach)
dataset:
  - _component_: torchtune.datasets.HfIterableDataset
    load_args:
        path: "tatsu-lab/alpaca"
        split: "train"
    message_transform:
        _component_: torchtune.datasets.alpaca_message_transform
        masking_strategy: "output_only"
        column_map:
            input: "prompt"
            output: "response"
            system_prompt: "foo"
    filter_args:
        function: torchtune.datasets.filter_fn_even_indices
        with_indices: True
    weight: 0.8
  - _component_: torchtune.datasets.HfIterableDataset
    load_args:
        path: "tatsu-lab/gsm8k"
        split: "train"
    message_transform:
        _component_: torchtune.datasets.gsm8k_message_transform
        masking_strategy: "output_only"
        column_map:
            input: "prompt"
            output: "response"
            system_prompt: "bar"
    weight: 0.2

# Option 2: Using Builders
# TODO: test indexing "tune run config – dataset[0].load_arg.split=train"
dataset:
  - _component_: torchtune.datasets.build_alpaca_dataset
    load_args:
        split: "valid"
    weight: 0.8
  - _component_: torchtune.datasets.build_gsm8k_dataset
    message_transform:
      system_prompt: "bar"
    weight: 0.2

# Option 3: Single Dataset
dataset:
  _component_: torchtune.datasets.build_alpaca_dataset

#########
# Common Dataset Arguments
# Used as cfg = dataset_defaults.update(dataset_cfg)
#########
dataset_defaults:
    shuffle_buffer_size: 1000
    num_shards_per_worker: 16
    seed: ${seed}
    tokenizer: ${tokenizer}
    recipe_transform:
      _component_: torchtune.datasets.SFTTransform

#########
# Dataset Setup Arguments (not dataset specific)
#########
dataset_setup:
    packing:
        _component_: torchtune.datasets.packing.SFTPacking
        max_seq_len: ${tokenizer.max_seq_len}
    multidataset_stopping_strategy: "first_exhausted"  # or "all_exhausted"
```

#### Builder Example
Location: torchtune/datasets/alpaca_dataset.py

```python
def alpaca_dataset(
    *,
    load_args: Optional[Dict],
    message_transform: Optional[Union[Callable, Dict]],
    tokenizer: ModelTokenizer,
    recipe_transform: Callable,
    *args, **kwargs
):
    _load_args = {
        "source": "tatsu-lab/alpaca",
        "split": "train"
    }
    _message_transform_args = {
        "train_on_input": False,
        "column_map": {"input": "prompt", "output": "response"}
    }

    # Unify args
    if load_args:
        _load_args.update(**load_args)

    # Unify args
    if not message_transform and isinstance(message_transform, dict):
        # Remove component key since we're using alpaca_message_transform as default
        message_transform.pop("_component_", None)

        # Instantiate the message transform
        _message_transform_args.update(message_transform)
        message_transform = alpaca_message_transform(**_message_transform_args)

    return HfIterableDataset(
        load_args, message_transform, tokenizer, recipe_transform, *args, **kwargs
    )
```

#### Iterable Dataset Implementation
This is shared for all datasets and recipes (SFT, DPO, etc). Differences are in the transforms.
Location: torchtune/datasets/hf_iterable_dataset.py

```python
class HfIterableDataset(IterableDataset, Stateful):
    def __init__(
        self,
        *,
        load_args: Dict,
        message_transform: Callable,
        tokenizer: Callable,
        recipe_transform: Callable,
        shuffle_buffer_size: Optional[int] = 1000,
        seed: Optional[int] = 42,
        num_shards_per_worker: int = 16,
        weight: float = 1.0,
        filter_args: Optional[Dict] = None,
        *args, **kwargs
    ):
        """Initialize a single dataset with its specific transformations."""
        self.weight = weight

        world_size = 1
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()

        # TODO: Maybe num shards should be based on dataset size, if we know it
        num_shards = world_size * num_shards_per_worker
        ds = load_dataset(**load_args)
        ds = ds.to_iterable_dataset(num_shards)

        if filter_args:
            function = filter_args.get("function", None)
            if function and not isinstance(function, Callable):
                raise ValueError(
                    f"filter_args['function'] must be a callable. Found {type(function)}"
                )
            # https://huggingface.co/docs/datasets/v3.6.0/en/stream#filter
            ds = ds.filter(**filter_args)

        def _apply_transforms(sample):
            sample = message_transform(sample)
            sample = tokenizer(sample)
            return recipe_transform(sample)

        ds = ds.map(_apply_transforms)  # lazy

        if shuffle_buffer_size and shuffle_buffer_size > 0:
            ds = ds.shuffle(shuffle_buffer_size, seed)

        # Distribute
        if world_size > 1:
            ds = split_dataset_by_node(
                ds,
                rank=torch.distributed.get_rank(),
                world_size=world_size,
            )

        self.ds = ds

    def __iter__(self):
        # Expose the for loop so extra logic can be added here, e.g. drop if no trainable tokens
        # TODO: should we add try/except to handle/log errors?
        for sample in self.ds:
            yield sample

    def state_dict(self):
        state_dict = self.ds.state_dict()
        state_dict["weight"] = self.weight
        return state_dict

    def load_state_dict(self, state_dict):
        self.weight = state_dict.pop("weight")
        self.ds.load_state_dict(state_dict)
```

#### Setup Data
Method in recipes/full_distributed.py or *utility* used in the recipe

```python
from datasets import interleave_datasets, split_dataset_by_node
from torchtune.models.tokenizers import ModelTokenizer
import torch

# NOTE: Mixed feelings about passing multiple ConfigDict to setup_data.
# Hard for users to know what they should contain.
# On the other hand:
# i) setup_data doesn't need to make assumptions about the configs
# ii) we already do it currently
# Alternative: use dataclasses?

def setup_data(
    dataset_cfg: ConfigDict,
    dataset_defaults: ConfigDict,
    data_setup_cfg: ConfigDict,
    dataloader_cfg: ConfigDict,
    seed: int,
    pad_idx: int,
    ignore_idx: int,
    pad_to_multiple_of: int,
) -> "IterableDataset":
    """Equivalent to setup_data in the recipe."""
    iterable_datasets = []
    weights = []
    dataset_defaults = {} if dataset_defaults is None else dataset_defaults

    # Add dataset to a list just for processing
    if not isinstance(dataset_cfg, list):
        dataset_cfg = [dataset_cfg]

    # instantiate
    for base_cfg in dataset_cfg:
        weight = base_cfg.get("weight", 1.0)
        weights.append(weight)

        base_cfg = OmegaConf.merge(dataset_defaults, base_cfg)
        ds = instantiate(base_cfg)
        iterable_datasets.append(ds)

    # Interleave for multidataset
    if len(iterable_datasets) > 1:
        weights = normalize_weights(weights)  # sum to 1
        ds = interleave_datasets(
            iterable_datasets,
            probabilities=weights,
            seed=seed,
            # strategies: https://huggingface.co/docs/datasets/v3.3.2/en/package_reference/main_classes#datasets.interleave_datasets.stopping_strategy
            stopping_strategy=data_setup_cfg.multidataset_stopping_strategy,
        )
    else:
        ds = iterable_datasets[0]

    if setup_cfg.packing:
        # Subclass of IterableDataset, takes any iterator as input
        ds = instantiate(
            data_setup_cfg.packing,
            dataset=ds,
            padding_idx=pad_id,  # TODO: in the future, move padding to collate_fn
        )

    # Instantiate collate_fn
    collate_fn = dataloader_cfg.pop("collate_fn", None)
    # TODO: in the future, unify those two
    if collate_fn is None:
        collate_fn = (
            "torchtune.data.padded_collate_packed"
            if packing else
            "torchtune.data.padded_collate_sft"
        )

    collate_fn = _get_component_from_path(collate_fn)
    collate_fn = partial(
        collate_fn,
        padding_idx=pad_idx,
        ignore_idx=ignore_id,
        pad_to_multiple_of=pad_to_multiple_of
    )

    # Dropping last avoids shape issues with compile + flex attention
    if "drop_last" not in dataloader_cfg:
        dataloader_cfg["drop_last"] = True

    dataloader = instantiate(dataloader_cfg, dataset=ds, collate_fn=collate_fn)

    return dataloader
```

#### Recipe Train Loop

```python
for epoch in range(n_epochs):
    # TODO: review the proper way to reshuffle after each epoch
    # https://huggingface.co/docs/datasets/v3.6.0/en/about_mapstyle_vs_iterable#speed-differences
    self._dataloader.set_epoch(epoch)
    for example in dataloader:
        pass
```

### Backward Compatibility

Options:

1. Make setup_data an utility, and have two utilities supporting old and new config formats.
   After deprecation period, old utility is removed.

   Pros:
   - Use it across recipes. Updates need to be done in one place.
   - Step towards our modularization goal.

   Cons:
   - Big change in how we handle recipe utilities

2. Create an adapter migrate_old_to_new_config:

   Pros:
   - Recipes still have method _setup_data exposing the logic

   Cons:
   - Hard to debug the migrated configs
   - Edge cases not covered by the adapter
   - ConcatDataset is handled differently

3. No migration. Old config with old recipe will break:
   - Users need to update their configs
   - Unknown impact on llamastack / startups / others

#### Implementation of Option 1 (Make setup_data an utility)

Location: torchtune/training/data_utils.py or similar

```python
@deprecated
def is_legacy_data_config(cfg: DictConfig) -> bool:
    """Detect if config follows legacy format vs new iterable dataset format."""
    # Check for new format indicators first
    has_dataloader_section = "dataloader" in cfg
    has_dataset_defaults = "dataset_defaults" in cfg
    has_dataset_setup = "dataset_setup" in cfg

    return not (has_dataloader_section or has_dataset_defaults or has_dataset_setup)

@deprecated
def setup_data_legacy(
    ...
) -> StatefulDataLoader:
    """
    Legacy data setup function to maintain backward compatibility.
    This replicates the current behavior in full_finetune_distributed.py
    """
    # same as current setup_data in the recipe....
    return dataloader
```

In the recipe:
```python
def _setup(...):
    ...
    if is_legacy_data_config(cfg):
        dataloader = setup_data_legacy(...)
    else:
        dataloader = setup_data(...)
```
