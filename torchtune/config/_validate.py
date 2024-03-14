# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from omegaconf import DictConfig
from torchtune.config._errors import ConfigError
from torchtune.config._instantiate import _instantiate_node
from torchtune.config._utils import _has_component


def validate(cfg: DictConfig) -> None:
    """
    Ensure that all components in the config can be instantiated correctly

    Args:
        cfg (DictConfig): The config to validate

    Raises:
        Exception: If any component cannot be instantiated
    """

    errors = []
    for k, v in cfg.items():
        if _has_component(v):
            try:
                obj = _instantiate_node(v)
            # Some objects require other objects as arguments, like optimizers,
            # lr_schedulers, datasets, etc. Try doing partial instantiation
            except TypeError as e:
                if "required positional argument" in str(e):
                    obj = _instantiate_node(v, partial_instantiate=True)
                else:
                    errors.append(e)

    if errors:
        raise ConfigError(errors)
