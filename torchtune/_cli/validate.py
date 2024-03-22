# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from torchtune import config
from torchtune.config._utils import _merge_yaml_and_cli_args


def validate_cmd(args):
    yaml_args = args.yaml_args
    cli_args = args.cli_args
    conf = _merge_yaml_and_cli_args(yaml_args, cli_args)
    config.validate(cfg)
    print("Config is well-formed!")
