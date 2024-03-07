# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from omegaconf import DictConfig
from torchtune import config


@config.parse
def main(cfg: DictConfig):
    config.validate(cfg)
    print("Config is well-formed!")


if __name__ == "__main__":
    main()
