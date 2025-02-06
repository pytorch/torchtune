# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file mainly exists because we want to ensure that `recipes` aren't
# importable *from the tests*.
# We're using the `prepend` pytest import mode which adds the root dir (i.e. the
# parent of torchtune/, tests/, recipes/) to the pythonpath during pytest
# sessions
# (https://docs.pytest.org/en/7.1.x/explanation/pythonpath.html#import-modes).
# This has the positive effect that the `tests` folder becomes importable when
# testing (we need that, considering how tests are currently set up) but ALSO
# has the negative effect of making the `recipes/` importable when testing.
# Since we don't want the tests to to incorrectly assume that recipes are
# importable, we have to explicitly raise an error here.

raise ModuleNotFoundError(
    "The torchtune recipes directory isn't a package and you should not import anything from here. "
    "Refer to our docs for detailed instructions on how to use recipes: "
    "https://pytorch.org/torchtune/main/deep_dives/recipe_deepdive.html"
)
