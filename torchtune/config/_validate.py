# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import inspect

from omegaconf import DictConfig
from torchtune.config._errors import ConfigError
from torchtune.config._utils import _get_component_from_path, _has_component


def validate(cfg: DictConfig) -> None:
    """
    Ensure that all components in the config can be instantiated correctly

    Args:
        cfg (DictConfig): The config to validate

    Raises:
        ConfigError: If any component cannot be instantiated
    """

    errors = []
    for node, nodedict in cfg.items():
        if _has_component(nodedict):
            try:
                _component_ = _get_component_from_path(nodedict.get("_component_"))
                kwargs = {k: v for k, v in nodedict.items() if k != "_component_"}
                sig = inspect.signature(_component_)
                sig.bind(**kwargs)
            # Some objects require other objects as arguments, like optimizers,
            # lr_schedulers, datasets, etc. Try doing partial instantiation
            except TypeError as e:
                if "missing a required argument" in str(e):
                    sig.bind_partial(**kwargs)
                else:
                    # inspect.signature does not retain the function name in the
                    # exception, so we manually add it back in
                    e = TypeError(f"{_component_.__name__} {str(e)}")
                    errors.append(e)

    if errors:
        raise ConfigError(errors)
