# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""
Keys used during checkpoint load and checkpoint save.
"""

# adapter config containing info about LoRA modules, rank, alpha
ADAPTER_CONFIG = "adapter_config"
# key used for adapter weights such as LoRA weights
ADAPTER_KEY = "adapter"
# number of epochs completed thus far
EPOCHS_KEY = "epochs_run"
MAX_STEPS_KEY = "max_steps_per_epoch"
MODEL_KEY = "model"
OPT_KEY = "optimizer"
SEED_KEY = "seed"
# total number of epochs for training; resumed training runs for
# (total_epochs - epochs_run) number of epochs
TOTAL_EPOCHS_KEY = "total_epochs"
