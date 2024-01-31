# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""
Keys used during checkpoint load and checkpoint save.
"""
# number of epochs completed thus far
EPOCHS_RUN_CKPT_DICT_KEY = "epochs_run"

# model weights
MODEL_WEIGHTS_CKPT_DICT_KEY = "model"

# optimizer state
OPTIMIZER_STATE_CKPT_DICT_KEY = "optimizer"

# seed used during training
SEED_CKPT_DICT_KEY = "seed"
