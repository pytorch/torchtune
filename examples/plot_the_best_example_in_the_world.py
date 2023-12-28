# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
The Best Example in the World
=============================

This is the best example in the world.
"""

# %%
# This is a new cell
# ------------------
#
# With rst syntax in it.
# Let's use :class:`~torchtune.modules import CausalSelfAttention`. <-- This
# should be a link to the class, and you should see a "Examples using CausalSelfAttetnion" backlink on that docstring.

# %%
# Another cell, with code this time:

from torchtune.modules import CausalSelfAttention

print(CausalSelfAttention)
