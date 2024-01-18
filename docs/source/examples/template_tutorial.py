# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Template Tutorial
=================

.. grid:: 2

    .. grid-item-card:: :octicon:`mortar-board;1em;` What you will learn

      * Item 1
      * Item 2
      * Item 3

    .. grid-item-card:: :octicon:`list-unordered;1em;` Prerequisites

      * Do not feel obliged to fill those boxes if it's not relevant.
      * This is just an example

      
For instructions on how to build the docs and write that kind of examples,
please follow the instructions on the torchtune README:
https://github.com/pytorch-labs/torchtune. Reach out to Nicolas if anything is
confusing. If you're developing on a remote machine, look at
https://github.com/pytorch-labs/torchtune/pull/149/files

"""


# %%
# This is a new cell
# ------------------
#
# With rst syntax in it.
# Let's use :class:`~torchtune.modules.CausalSelfAttention`. <-- This
# should be a link to the class, and you should see a "Examples using
# CausalSelfAttention" backlink on that docstring. Go check it out by clicking
# on it!

from torchtune.modules import CausalSelfAttention

print(CausalSelfAttention)

# %%
# Note that you can open this file with vscode and execute it as a notebook
# within vscode. It should work with other IDEs like PyCharm.

print("OK cool.")

# %%
# If you have a video, add it here like this:
# 
# .. raw:: html
# 
#    <div style="margin-top:10px; margin-bottom:10px;">
#      <iframe
#         width="560"
#         height="315"
#         src="https://www.youtube.com/embed/IC0_FRiX-sw"
#         frameborder="0"
#         allow="accelerometer; encrypted-media; gyroscope; picture-in-picture"
#         allowfullscreen>
#       </iframe>
#    </div>