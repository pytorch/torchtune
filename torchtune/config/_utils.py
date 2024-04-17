# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace
from importlib import import_module
from types import ModuleType
from typing import Any, Dict, List, Union

from omegaconf import DictConfig, OmegaConf

from torchtune.config._errors import InstantiationError
from torchtune.data import ChatFormat, InstructTemplate
from torchtune.utils import get_logger, get_world_size_and_rank


def log_config(recipe_name: str, cfg: DictConfig) -> None:
    """
    Logs the parsed config to rank zero.

    Args:
        recipe_name (str): name of the recipe to display
        cfg (DictConfig): parsed config object
    """
    # Log the config only on rank 0
    _, rank = get_world_size_and_rank()
    if rank != 0:
        return

    logger = get_logger("DEBUG")
    cfg_str = OmegaConf.to_yaml(cfg, resolve=True, sort_keys=True)
    logger.info(msg=f"Running {recipe_name} with resolved config:\n\n{cfg_str}")


def _has_component(node: Union[Dict[str, Any], DictConfig]) -> bool:
    return (OmegaConf.is_dict(node) or isinstance(node, dict)) and "_component_" in node


def _get_component_from_path(path: str) -> Any:
    """
    Return an object by name or dotted path, importing as necessary.
    The base functionality relies on ``getattr()`` and handles all
    possible exceptions accordingly.

    Based on Hydra's `_locate` from Facebook Research:
    https://github.com/facebookresearch/hydra/blob/main/hydra/_internal/utils.py#L614

    Args:
        path (str): Dotted path of the object

    Returns:
        Any: The object

    Raises:
        InstantiationError: If there is an exception loading the
            object from the provided path
        ValueError: If a relative or invalid dotpath is passed in
    """
    if path == "":
        raise ValueError("Empty path")

    parts = [part for part in path.split(".")]
    for part in parts:
        # If a relative path is passed in, the first part will be empty
        if not len(part):
            raise ValueError(
                f"Error loading '{path}': invalid dotstring."
                + "\nRelative imports are not supported."
            )
    # First module requires trying to import to validate
    part0 = parts[0]
    try:
        obj = import_module(part0)
    except ImportError as exc_import:
        raise InstantiationError(
            f"Error loading '{path}':\n{repr(exc_import)}"
            + f"\nAre you sure that module '{part0}' is installed?"
        ) from exc_import
    # Subsequent components can be checked via getattr() on first module
    # It can either be an attribute that we can return or a submodule that we
    # can import and continue searching
    for m in range(1, len(parts)):
        part = parts[m]
        try:
            obj = getattr(obj, part)
        # If getattr fails, check to see if it's a module we can import and
        # continue down the path
        except AttributeError as exc_attr:
            parent_dotpath = ".".join(parts[:m])
            if isinstance(obj, ModuleType):
                mod = ".".join(parts[: m + 1])
                try:
                    obj = import_module(mod)
                    continue
                except ModuleNotFoundError as exc_import:
                    raise InstantiationError(
                        f"Error loading '{path}':\n{repr(exc_import)}"
                        + f"\nAre you sure that '{part}' is importable from module '{parent_dotpath}'?"
                    ) from exc_import
                # Any other error trying to import module can be raised as
                # InstantiationError
                except Exception as exc_import:
                    raise InstantiationError(
                        f"Error loading '{path}':\n{repr(exc_import)}"
                    ) from exc_import
            # If the component is not an attribute nor a module, it doesn't exist
            raise InstantiationError(
                f"Error loading '{path}':\n{repr(exc_attr)}"
                + f"\nAre you sure that '{part}' is an attribute of '{parent_dotpath}'?"
            ) from exc_attr
    return obj


def _merge_yaml_and_cli_args(yaml_args: Namespace, cli_args: List[str]) -> DictConfig:
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
        cli_args (List[str]): List of key=value strings

    Returns:
        DictConfig: OmegaConf DictConfig containing merged args

    Raises:
        ValueError: If a cli override is not in the form of key=value
    """
    # Convert Namespace to simple dict
    yaml_kwargs = vars(yaml_args)
    cli_dotlist = []
    for arg in cli_args:
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
        cli_dotlist.append(f"{k}={v}")

    # Merge the args
    cli_conf = OmegaConf.from_dotlist(cli_dotlist)
    yaml_conf = OmegaConf.create(yaml_kwargs)

    # CLI takes precedence over yaml args
    return OmegaConf.merge(yaml_conf, cli_conf)


def _try_get_component(module_path: str, component_name: str, class_type: str) -> Any:
    """
    Try-except wrapper around `_get_component_from_path`, used to quickly retrieve
    a class from a name string with better error handling.

    Args:
        module_path (str): path string of the file the class resides in
        component_name (str): name of the class
        class_type (str): type of the class, only used for more descriptive error message


    Returns:
        Any: the class

    Raises:
        ValueError: if the string is not a valid class
    """
    try:
        return _get_component_from_path(module_path + "." + component_name)
    except InstantiationError:
        raise ValueError(f"Invalid {class_type} class: '{component_name}'") from None


def _get_instruct_template(template: str) -> InstructTemplate:
    """
    Get the instruct template class from the template string.

    Args:
        template (str): class name of template, or string with placeholders

    Returns:
        InstructTemplate: the prompt template class or the same verified string
    """
    return _try_get_component(
        "torchtune.data._instruct_templates", template, "InstructTemplate"
    )


def _get_chat_format(chat_format: str) -> ChatFormat:
    """
    Get the chat format class from a string.

    Args:
        chat_format (str): class name of the ChatFormat

    Returns:
        ChatFormat: the chat format class
    """
    return _try_get_component("torchtune.data._chat_formats", chat_format, "ChatFormat")
