# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import ast
import difflib
import inspect
import json
import os
import textwrap
from pathlib import Path
from typing import Any, Optional, Union

import torch
from torch.utils.data import Dataset

CKPT_COMPONENT_MAP = {
    "tune": "torchtune.training.FullModelTorchTuneCheckpointer",
    "meta": "torchtune.training.FullModelMetaCheckpointer",
    "hf": "torchtune.training.FullModelHFCheckpointer",
}


class DummyDataset(Dataset):
    def __init__(self, *args, **kwargs):
        self._data = torch.LongTensor(
            [
                [0, 2, 4, 2, 5, 6, 7, 8, 9, 1, 2, 4, 3, 3, 5, 6, 8, 2, 1, 1],
                [1, 2, 5, 6, 7, 8, 2, 3, 1, 9, 9, 9, 5, 6, 7, 0, 0, 0, 1, 2],
                [5, 6, 8, 2, 1, 0, 3, 4, 0, 0, 0, 2, 4, 7, 8, 8, 2, 2, 1, 0],
                [4, 6, 7, 1, 0, 2, 0, 2, 0, 2, 3, 9, 9, 9, 7, 5, 1, 8, 4, 1],
            ]
        )
        self._labels = torch.LongTensor(
            [
                [2, 6, 7, 8, 2, 2, 1, 0, 0, 1],
                [1, 2, 5, 6, 7, 8, 2, 3, 1, 9],
                [6, 1, 1, 2, 5, 0, 9, 0, 2, 1],
                [5, 8, 6, 0, 2, 0, 0, 3, 2, 1],
            ]
        )

    def __getitem__(self, index):
        return {"tokens": self._data[index], "labels": self._labels[index]}

    def __len__(self):
        return len(self._data)


def get_assets_path():
    return Path(__file__).parent.parent / "assets"


def dummy_stack_exchange_dataset_config():
    data_files = os.path.join(get_assets_path(), "stack_exchange_paired_tiny.json")
    out = [
        "dataset._component_=torchtune.datasets.stack_exchange_paired_dataset",
        "dataset.source='json'",
        f"dataset.data_files={data_files}",
        "dataset.split='train'",
    ]
    return out


def dummy_alpaca_dataset_config():
    data_files = os.path.join(get_assets_path(), "alpaca_tiny.json")
    out = [
        "dataset._component_=torchtune.datasets.alpaca_dataset",
        "dataset.source='json'",
        f"dataset.data_files={data_files}",
        "dataset.split='train'",
    ]
    return out


def dummy_text_completion_alpaca_dataset_config():
    """
    Constructs a minimal text-completion-style dataset from ``alpaca_tiny.json``.
    This is used for testing PPO fine-tuning.
    """
    data_files = os.path.join(get_assets_path(), "alpaca_tiny.json")
    out = [
        "dataset._component_=torchtune.datasets.text_completion_dataset",
        "dataset.source='json'",
        f"dataset.data_files={data_files}",
        "dataset.column='instruction'",
        "dataset.split='train[:10%]'",  # 10% of the dataset gets us 8 batches
        "dataset.add_eos=False",
    ]
    return out


def llama2_test_config() -> list[str]:
    return [
        "model._component_=torchtune.models.llama2.llama2",
        "model.vocab_size=32_000",
        "model.num_layers=4",
        "model.num_heads=16",
        "model.embed_dim=256",
        "model.max_seq_len=2048",
        "model.norm_eps=1e-5",
        "model.num_kv_heads=8",
    ]


def llama2_classifier_test_config() -> list[str]:
    return [
        "model._component_=torchtune.modules.classifier_model",
        "model.base_model_path=torchtune.models.llama2.llama2",
        "model.num_classes=1",
        "model.vocab_size=32_000",
        "model.num_layers=4",
        "model.num_heads=16",
        "model.embed_dim=256",
        "model.max_seq_len=2048",
        "model.norm_eps=1e-5",
        "model.num_kv_heads=8",
    ]


def llama3_test_config() -> list[str]:
    return [
        "model._component_=torchtune.models.llama3.llama3",
        "model.vocab_size=128_256",
        "model.num_layers=2",
        "model.num_heads=8",
        "model.embed_dim=64",
        "model.max_seq_len=1024",
        "model.norm_eps=1e-5",
        "model.num_kv_heads=4",
    ]


def llama3_test_config_137m() -> list[str]:
    """
    Test config with slightly larger embed dim to be paged and flex attention friendly
    """
    return [
        "model._component_=torchtune.models.llama3.llama3",
        "model.vocab_size=128_256",
        "model.num_layers=2",
        "model.num_heads=4",
        "model.embed_dim=512",
        "model.max_seq_len=1024",
        "model.norm_eps=1e-5",
        "model.num_kv_heads=2",
    ]


def llama3_2_vision_test_config() -> list[str]:
    return [
        "model=tests.recipes.utils.dummy_vision_model",
        "tokenizer._component_=torchtune.models.llama3_2_vision._transform.Llama3VisionTransform",
        "tokenizer.patch_size=9",
        "tokenizer.max_num_tiles=2",
        "tokenizer.tile_size=18",
        "tokenizer.max_seq_len=4096",
    ]


def dummy_vision_model():
    from torchtune.models.llama3_2_vision._component_builders import (
        llama3_2_vision_decoder,
        llama3_2_vision_encoder,
    )
    from torchtune.modules.model_fusion import DeepFusionModel

    vision_encoder = llama3_2_vision_encoder(
        clip_embed_dim=128,
        clip_num_layers=4,
        num_heads=4,
        tile_size=18,
        patch_size=9,
        max_num_tiles=2,
        in_channels=3,
        clip_hidden_states=[0, 1],
        num_layers_projection=2,
        decoder_embed_dim=128,
    )
    vision_decoder = llama3_2_vision_decoder(
        vocab_size=128256,
        num_layers=4,
        fusion_interval=2,
        num_special_tokens=2,
        num_heads=8,
        num_kv_heads=4,
        embed_dim=128,
        max_seq_len=4096,
        encoder_max_seq_len=4096,
    )

    model = DeepFusionModel(
        encoder=vision_encoder,
        decoder=vision_decoder,
        encoder_trainable=False,
        decoder_trainable=False,
        fusion_trainable=False,
    )
    return model


def lora_llama2_test_config(
    lora_attn_modules,
    apply_lora_to_mlp: bool = False,
    apply_lora_to_output: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 16,
    quantize_base: bool = False,
    use_dora: bool = False,
) -> list[str]:
    return [
        # Note: we explicitly use _component_ so that we can also call
        # config.instantiate directly for easier comparison
        "model._component_=torchtune.models.llama2.lora_llama2",
        f"model.lora_attn_modules={lora_attn_modules}",
        f"model.apply_lora_to_mlp={apply_lora_to_mlp}",
        f"model.apply_lora_to_output={apply_lora_to_output}",
        "model.vocab_size=32000",
        "model.num_layers=4",
        "model.num_heads=16",
        "model.embed_dim=256",
        "model.max_seq_len=2048",
        "model.norm_eps=1e-5",
        "model.num_kv_heads=8",
        f"model.lora_rank={lora_rank}",
        f"model.lora_alpha={lora_alpha}",
        "model.lora_dropout=0.0",
        f"model.quantize_base={quantize_base}",
        f"model.use_dora={use_dora}",
    ]


def lora_llama3_test_config(
    lora_attn_modules,
    apply_lora_to_mlp: bool = False,
    apply_lora_to_output: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 16,
    quantize_base: bool = False,
) -> list[str]:
    return [
        # Note: we explicitly use _component_ so that we can also call
        # config.instantiate directly for easier comparison
        "model._component_=torchtune.models.llama3.lora_llama3",
        f"model.lora_attn_modules={lora_attn_modules}",
        f"model.apply_lora_to_mlp={apply_lora_to_mlp}",
        f"model.apply_lora_to_output={apply_lora_to_output}",
        "model.vocab_size=128_256",
        "model.num_layers=2",
        "model.num_heads=8",
        "model.embed_dim=64",
        "model.max_seq_len=1024",
        "model.norm_eps=1e-5",
        "model.num_kv_heads=4",
        f"model.lora_rank={lora_rank}",
        f"model.lora_alpha={lora_alpha}",
        "model.lora_dropout=0.0",
        f"model.quantize_base={quantize_base}",
    ]


def write_hf_ckpt_config(ckpt_dir: Union[str, Path]):
    config = {
        "hidden_size": 256,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
    }
    config_file = Path.joinpath(Path(ckpt_dir), "config.json")
    with config_file.open("w") as f:
        json.dump(config, f)


def write_hf_vision_ckpt_config(ckpt_dir: str):
    config = {
        "text_config": {
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "hidden_size": 128,
            "vocab_size": 128256,
            "cross_attention_layers": [1, 4],
        },
        "vision_config": {
            "hidden_size": 128,
            "image_size": 18,
            "max_num_tiles": 2,
            "supported_aspect_ratios": [[1, 1], [1, 2], [2, 1]],
        },
    }
    config_file = Path.joinpath(Path(ckpt_dir), "config.json")
    with config_file.open("w") as f:
        json.dump(config, f)


MODEL_TEST_CONFIGS = {
    "llama2": llama2_test_config(),
    "llama3": llama3_test_config(),
    "llama3_137M": llama3_test_config_137m(),
    "llama2_lora": lora_llama2_test_config(
        lora_attn_modules=["q_proj", "k_proj", "v_proj", "output_proj"],
        apply_lora_to_mlp=False,
        apply_lora_to_output=False,
        lora_rank=8,
        lora_alpha=16,
    ),
    "llama2_dora": lora_llama2_test_config(
        lora_attn_modules=["q_proj", "k_proj", "v_proj", "output_proj"],
        apply_lora_to_mlp=False,
        apply_lora_to_output=False,
        lora_rank=8,
        lora_alpha=16,
        use_dora=True,
    ),
    "llama2_qlora": lora_llama2_test_config(
        lora_attn_modules=["q_proj", "k_proj", "v_proj", "output_proj"],
        apply_lora_to_mlp=True,
        apply_lora_to_output=False,
        lora_rank=8,
        lora_alpha=16,
        quantize_base=True,
    ),
    "llama3_lora": lora_llama3_test_config(
        lora_attn_modules=["q_proj", "k_proj", "v_proj", "output_proj"],
        apply_lora_to_mlp=False,
        apply_lora_to_output=False,
        lora_rank=8,
        lora_alpha=16,
    ),
}

# --- Compare content of methods in recipes ---


class StripIgnoredNodes(ast.NodeTransformer):
    """
    AST transformer to remove specified keyword arguments from function calls,
    and assignments to specified attributes or names. Optionally removes docstrings from expressions.
    """

    def __init__(
        self,
        ignored_kwargs: Optional[list[str]] = None,
        ignored_attrs: Optional[list[str]] = None,
        ignore_docstrings: bool = False,
    ) -> None:
        self.ignored_kwargs = set(ignored_kwargs or [])
        self.ignored_attrs = set(ignored_attrs or [])
        self.ignore_docstrings = ignore_docstrings

    def visit_Call(self, node: ast.Call) -> ast.Call:  # noqa: N802
        """Remove specified keyword arguments from function calls
        For example, we might have a call like:
        >>> self._checkpoint_client.save_checkpoint(
        >>>     model=self._model,
        >>>     optimizer=self.optimizer,
        >>>     training_progress=TrainingProgress(
        >>>         ...
        >>>     ),
        >>>     epoch=epoch,
        >>>     adapter_config=self._adapter_config.copy(),
        >>>     adapter_only=self._save_adapter_weights_only,
        >>> )
        and we want to skip the `adapter_config` and `adapter_only` kwargs, but check for equality of everything else.
        """

        node.keywords = [
            kw for kw in node.keywords if kw.arg not in self.ignored_kwargs
        ]
        return self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> Optional[ast.Assign]:  # noqa: N802
        """Remove assignments to specified attribute or variable names
        For example, we might have a method like:
        >>> def __init__(self, cfg: DictConfig) -> None:
        >>>     ...
        >>>     self._output_dir = cfg.output_dir
        >>>     self._log_every_n_steps = cfg.get("log_every_n_steps", 1)
        >>>     self._log_peak_memory_stats = cfg.get("log_peak_memory_stats", False)
        >>>     self._logger = utils.get_logger(cfg.log_level)
        >>>     ...
        and we want to ignore the `_save_adapter_weights_only` attribute, but check for equality of everything else.
        """

        for target in node.targets:
            if isinstance(target, ast.Attribute) and target.attr in self.ignored_attrs:
                return None
            if isinstance(target, ast.Name) and target.id in self.ignored_attrs:
                return None
        return self.generic_visit(node)

    def visit_Expr(self, node: ast.Expr) -> Optional[ast.Expr]:  # noqa: N802
        """Remove docstrings from expressions"""

        if (
            self.ignore_docstrings
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            return None
        return self.generic_visit(node)


def get_method_ast(cls: type[Any], method_name: str) -> ast.FunctionDef:
    """
    Returns the AST (Abstract Syntax Tree) node for the specified method in the given class: a tree-like structure that
    represents the code in a way that Python can analyze.

    Note: Comments are not included in Python's AST.
    """

    method = getattr(cls, method_name)
    source = inspect.getsource(method)
    source = textwrap.dedent(source).strip()  # Fix indentation for class methods
    tree = ast.parse(source)

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            return node
    raise ValueError(f"No method definition for {method_name} found in {cls.__name__}")


def filter_ast(
    node: ast.FunctionDef,
    ignored_kwargs: Optional[list[str]] = None,
    ignored_attrs: Optional[list[str]] = None,
    ignore_docstrings: bool = False,
) -> str:
    """Process a function's AST by filtering out specified keyword arguments and attributes,
    then convert the modified AST back to source code."""

    stripper = StripIgnoredNodes(ignored_kwargs, ignored_attrs, ignore_docstrings)
    stripped = stripper.visit(ast.fix_missing_locations(node))
    return ast.unparse(stripped).strip()


def diff_strings(a: str, b: str, from_file: str = "", to_file: str = "") -> str:
    """Generate a unified diff string between two input strings.

    This function compares two strings line by line and returns a unified diff,
    which highlights the differences between them in a format similar to the output
    of the Unix `diff -u` command. This is useful for displaying changes between
    two code snippets, text files, or any multi-line strings.

    Args:
        a (str): The original string to compare.
        b (str): The modified string to compare against.
        from_file (str, optional): The label for the original string, typically a filename.
            Defaults to an empty string.
        to_file (str, optional): The label for the modified string, typically a filename.
            Defaults to an empty string.

    Returns:
        str: A unified diff string showing the differences between `a` and `b`.
             If there are no differences, returns an empty string.

    Example:
        >>> diff_strings("foo\\nbar", "foo\\nbaz", from_file="old.py", to_file="new.py")
        --- old.py
        +++ new.py
        @@ -1,2 +1,2 @@
         foo
        -bar
        +baz
    """

    return "\n".join(
        difflib.unified_diff(
            a.strip().splitlines(),
            b.strip().splitlines(),
            fromfile=from_file,
            tofile=to_file,
            lineterm="",
        )
    )
