# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import inspect
import sys
from pathlib import Path

import pytest

from tests.recipes.utils import diff_strings, filter_ast, get_method_ast


# Hack to bypass the import error for the recipes module.
def import_class(file_path: str, class_name: str):
    module_name = Path(file_path).stem
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # Register the module
    spec.loader.exec_module(module)  # Execute the module
    return getattr(module, class_name)


FullFinetuneRecipeSingleDevice = import_class(
    "recipes/full_finetune_single_device.py", "FullFinetuneRecipeSingleDevice"
)
LoRAFinetuneRecipeSingleDevice = import_class(
    "recipes/lora_finetune_single_device.py", "LoRAFinetuneRecipeSingleDevice"
)


@pytest.mark.parametrize(
    "recipe_class_a, recipe_class_b",
    [
        (FullFinetuneRecipeSingleDevice, LoRAFinetuneRecipeSingleDevice),
    ],
)
@pytest.mark.parametrize(
    "method_name",
    [
        "_update_recipe_state",
        "_setup_profiler",
        "_setup_optimizer",
        "_setup_lr_scheduler",
        "_setup_data",
        "_loss_step",
        "train",
    ],
)
def test_shared_method_are_identical(
    recipe_class_a: str, recipe_class_b: str, method_name: str
) -> None:
    ast_a = get_method_ast(recipe_class_a, method_name)
    ast_b = get_method_ast(recipe_class_b, method_name)

    code_a = filter_ast(ast_a)
    code_b = filter_ast(ast_b)

    if code_a != code_b:
        diff = diff_strings(
            code_a,
            code_b,
            from_file=Path(inspect.getfile(recipe_class_a)).name,
            to_file=Path(inspect.getfile(recipe_class_b)).name,
        )
        pytest.fail(f"Method '{method_name}' differs:\n\n{diff}")


@pytest.mark.parametrize(
    "recipe_class_a, recipe_class_b",
    [
        (FullFinetuneRecipeSingleDevice, LoRAFinetuneRecipeSingleDevice),
    ],
)
@pytest.mark.parametrize(
    "method_name,ignored_kwargs,ignored_attrs",
    [
        ("__init__", [], ["_save_adapter_weights_only"]),
        ("save_checkpoint", ["adapter_config", "adapter_only"], []),
    ],
)
def test_methods_are_close(
    recipe_class_a: str,
    recipe_class_b: str,
    method_name: str,
    ignored_kwargs: list[str],
    ignored_attrs: list[str],
) -> None:
    ast_a = get_method_ast(recipe_class_a, method_name)
    ast_b = get_method_ast(recipe_class_b, method_name)

    code_a = filter_ast(ast_a, ignored_kwargs, ignored_attrs)
    code_b = filter_ast(ast_b, ignored_kwargs, ignored_attrs)

    if code_a != code_b:
        diff = diff_strings(
            code_a,
            code_b,
            from_file=Path(inspect.getfile(recipe_class_a)).name,
            to_file=Path(inspect.getfile(recipe_class_b)).name,
        )
        pytest.fail(f"Method '{method_name}' differs:\n\n{diff}")
