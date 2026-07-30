# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import inspect
from argparse import Namespace
from importlib import import_module
from typing import Any, Optional, Union

from omegaconf import DictConfig, OmegaConf

from torchtune.config._errors import InstantiationError
from torchtune.utils._logging import get_logger, log_rank_zero


def log_config(recipe_name: str, cfg: DictConfig) -> None:
    """
    Logs the resolved config (merged YAML file and CLI overrides) to rank zero.

    Args:
        recipe_name (str): name of the recipe to display
        cfg (DictConfig): parsed config object
    """
    logger = get_logger("DEBUG")
    cfg_str = OmegaConf.to_yaml(cfg, resolve=True, sort_keys=True)
    log_rank_zero(
        logger=logger, msg=f"Running {recipe_name} with resolved config:\n\n{cfg_str}"
    )


def _has_component(node: Union[dict[str, Any], DictConfig]) -> bool:
    return (OmegaConf.is_dict(node) or isinstance(node, dict)) and "_component_" in node


def _get_component_from_path(
    path: str, caller_globals: Optional[dict[str, Any]] = None
) -> Any:
    """
    Resolve a Python object from a dotted path or simple name.

    Retrieves a module, class, or function from a string like `"os.path.join"` or `"os"`. For dotted paths,
    it imports the module and gets the final attribute. For simple names, it imports the module or checks
    `caller_globals` (defaults to `__main__` globals if not provided).

    Args:
        path (str): Dotted path (e.g., "os.path.join") or simple name (e.g., "os").
        caller_globals (Optional[dict[str, Any]]): The caller's global namespace. Defaults to __main__ if None.

    Returns:
        Any: The resolved object (module, class, function, etc.).

    Raises:
        InstantiationError: If the path is empty, not a string, or if the module/attribute cannot be resolved.
        ValueError: If the path contains invalid dotstrings (e.g., relative imports like ".test" or "test..path").

    Examples:
        >>> _get_component_from_path("torch.nn.Linear")
        <class 'torch.nn.modules.linear.Linear'>
        >>> _get_component_from_path("torch")
        <module 'torch' from '...'>
        >>> # Assuming FooBar is in caller's globals
        >>> _get_component_from_path("FooBar")
        <class 'FooBar'>
    """
    if not path or not isinstance(path, str):
        raise InstantiationError(f"Invalid path: '{path}'")

    # Check for ".test", "test..path", "test..", etc.
    parts = path.split(".")
    if any(not part for part in parts):
        raise ValueError(
            f"Invalid dotstring. Relative imports are not supported. Got {path=}."
        )

    # single part, e.g. "torch" or "my_local_fn"
    if len(parts) == 1:
        name = parts[0]
        try:
            # try to import as a module, e.g. "torch"
            return import_module(name)
        except ImportError:
            # if caller_globals is None, collect __main__ globals of the caller
            search_globals = caller_globals if caller_globals is not None else {}
            if caller_globals is None:
                current_frame = inspect.currentframe()
                if current_frame and current_frame.f_back:
                    search_globals = current_frame.f_back.f_globals

            # check if local_fn is in caller_globals, e.g. "my_local_fn"
            if name in search_globals:
                return search_globals[name]
            else:
                # scope to differentiate between provided globals and caller's globals in error message
                scope = (
                    "the provided globals"
                    if caller_globals is not None
                    else "the caller's globals"
                )
                raise InstantiationError(
                    f"Could not resolve '{name}': not a module and not found in {scope}."
                ) from None

    # multiple parts, e.g. "torch.nn.Linear"
    module_path = ".".join(parts[:-1])
    try:
        module = import_module(module_path)
        component = getattr(module, parts[-1])
        return component
    except ImportError as e:
        raise InstantiationError(
            f"Could not import module '{module_path}': {str(e)}."
        ) from e
    except AttributeError as e:
        raise InstantiationError(
            f"Module '{module_path}' has no attribute '{parts[-1]}'."
        ) from e


def _merge_yaml_and_cli_args(yaml_args: Namespace, cli_args: list[str]) -> DictConfig:
    """
    Takes the direct output of argparse's parse_known_args which returns known
    args as a Namespace and unknown args as a dotlist (in our case, yaml args and
    cli args, respectively) and merges them into a single OmegaConf DictConfig.

    If a cli arg overrides a yaml arg with a _component_ field, the cli arg can
    be specified with the parent field directly, e.g., model=torchtune.models.lora_llama2_7b
    instead of model._component_=torchtune.models.lora_llama2_7b. Nested fields within the
    component should be specified with dot notation, e.g., model.lora_rank=16.

    Example:
        >>> config.yaml:
        >>>     a: 1
        >>>     b:
        >>>       _component_: torchtune.models.my_model
        >>>       c: 3

        >>> tune full_finetune --config config.yaml b=torchtune.models.other_model b.c=4
        >>> yaml_args, cli_args = parser.parse_known_args()
        >>> conf = _merge_yaml_and_cli_args(yaml_args, cli_args)
        >>> print(conf)
        >>> {"a": 1, "b": {"_component_": "torchtune.models.other_model", "c": 4}}

    Args:
        yaml_args (Namespace): Namespace containing args from yaml file, components
            should have _component_ fields
        cli_args (list[str]): list of key=value strings

    Returns:
        DictConfig: OmegaConf DictConfig containing merged args

    Raises:
        ValueError: If a cli override is not in the form of key=value
    """
    # Convert Namespace to simple dict
    yaml_kwargs = vars(yaml_args)
    yaml_top_level_keys = set(yaml_kwargs.keys())
    unused_keys: list[str] = []
    cli_dotlist = []
    for arg in cli_args:
        # If CLI override uses the remove flag (~), remove the key from the yaml config
        if arg.startswith("~"):
            dotpath = arg[1:].split("=")[0]
            if "_component_" in dotpath:
                raise ValueError(
                    f"Removing components from CLI is not supported: ~{dotpath}"
                )
            try:
                _remove_key_by_dotpath(yaml_kwargs, dotpath)
            except (KeyError, ValueError):
                raise ValueError(
                    f"Could not find key {dotpath} in yaml config to remove"
                ) from None
            continue
        # Get other overrides that should be specified as key=value
        try:
            k, v = arg.split("=")
        except ValueError:
            raise ValueError(
                f"Command-line overrides must be in the form of key=value, got {arg}"
            ) from None
        # If a cli arg overrides a yaml arg with a _component_ field, update the
        # key string to reflect this
        if k in yaml_kwargs and _has_component(yaml_kwargs[k]):
            k += "._component_"

        # Track CLI overrides that introduce a new top-level key not defined in the
        # YAML config. These are likely typos or recipe-unsupported kwargs and should
        # be surfaced to the user as a warning (see issue #1646).
        top_level_key = k.split(".")[0]
        if top_level_key not in yaml_top_level_keys and top_level_key not in unused_keys:
            unused_keys.append(top_level_key)

        # None passed via CLI will be parsed as string, but we really want OmegaConf null
        if v == "None":
            v = "!!null"

        # TODO: this is a hack but otherwise we can't pass strings with leading zeroes
        # to define the checkpoint file format. We manually override OmegaConf behavior
        # by prepending the value with !!str to force a string type
        if "max_filename" in k:
            v = "!!str " + v
        cli_dotlist.append(f"{k}={v}")

    if unused_keys:
        logger = get_logger("WARNING")
        suggestions = ", ".join(
            f"'{key}' (did you mean one of: "
            f"{', '.join(_closest_yaml_keys(key, yaml_top_level_keys)) or 'no matches'}?)"
            for key in unused_keys
        )
        log_rank_zero(
            logger=logger,
            msg=(
                "The following CLI override key(s) are not present in the YAML config "
                f"and may be unused by the recipe: {suggestions}. If this was "
                "intentional (e.g. adding a brand new field), you can ignore this "
                "warning. Otherwise, please check the key name against the config and "
                "recipe for a typo or unsupported option."
            ),
        )

    # Merge the args
    cli_conf = OmegaConf.from_dotlist(cli_dotlist)
    yaml_conf = OmegaConf.create(yaml_kwargs)

    # CLI takes precedence over yaml args
    return OmegaConf.merge(yaml_conf, cli_conf)


def _closest_yaml_keys(key: str, yaml_keys: set[str], max_suggestions: int = 3) -> list[str]:
    """Return up to ``max_suggestions`` keys from ``yaml_keys`` that look similar to
    ``key``. Uses a simple substring + difflib heuristic to surface likely typos."""
    import difflib

    candidates: list[tuple[float, str]] = []
    for yk in yaml_keys:
        score = difflib.SequenceMatcher(None, key, yk).ratio()
        # also reward substring containment so prefix typos rank high
        if key in yk or yk in key:
            score = max(score, 0.75)
        if score >= 0.6:
            candidates.append((score, yk))
    candidates.sort(reverse=True)
    return [yk for _, yk in candidates[:max_suggestions]]


def _remove_key_by_dotpath(nested_dict: dict[str, Any], dotpath: str) -> None:
    """
    Removes a key specified by dotpath from a nested dict. Errors should handled by
    the calling function.

    Args:
        nested_dict (dict[str, Any]): dict to remove key from
        dotpath (str): dotpath of key to remove, e.g., "a.b.c"
    """
    path = dotpath.split(".")

    def delete_non_component(d: dict[str, Any], key: str) -> None:
        if _has_component(d[key]):
            raise ValueError(
                f"Removing components from CLI is not supported: ~{dotpath}"
            )
        del d[key]

    def recurse_and_delete(d: dict[str, Any], path: list[str]) -> None:
        if len(path) == 1:
            delete_non_component(d, path[0])
        else:
            recurse_and_delete(d[path[0]], path[1:])
            if not d[path[0]]:
                delete_non_component(d, path[0])

    recurse_and_delete(nested_dict, path)
