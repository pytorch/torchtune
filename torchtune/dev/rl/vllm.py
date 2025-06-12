# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

VllmCompletionOutput = from_dataclass(vllm.outputs.CompletionOutput)

# move this into KVStore
def get_state_dict_version(state_dict, store):
    """
    Return KVStore parameter version. If multiple
    versions are found, return None as the store is
    being updated.
    """
    versions = {}
    for k in state_dict.keys():
        versions |= {store.get_version(k)}
    if len(versions) > 1:
        return None
    else:
        return versions.pop()
