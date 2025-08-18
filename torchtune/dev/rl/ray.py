# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional

import ray

from torch.distributed.tensor import DTensor
from torchtune import utils


@ray.remote
class KVStore:
    """
    - One actor per host
        - Ray cluster placement group
    - process group (stateless?) between KVStore workers
        - created based on ray.state.actors with class_name==KVStore
    """

    def get_version(self, key):
        """Get current version of key"""

    def read(self, key, offset=None, version=None):
        """
        - name and offset requested
        - if version is provided, rasie error if version mismatch
            - immediately lock key at source, async copy to destination
            - locks don't block writing, but force new writes to copy to new location until read is done
        - location of shards determined in local actor
        - local actor gloo.recv and gloo.send called on remote actors
            - gloo picks fastest protocol available (RDMA or TCP)
            - same host creates one time copy
        - return address for tensor Plasma (shared memory) location
        """

    def write(self, key, tensor, version):
        """
        - metadata (name/offset/location) shared in object store
        - object offloaded via Plasma in shared memory with KVStore
        - version tracked, all copies invalidated
        - not available for use until all shards added
        """


def get_kvstore():
    """Return local kvstore Actor"""
    my_ip = str(ray.util.get_node_ip_address())
    local_actor = ray.util.state.list_actors(
        filters=[("class_name", "=", "KVStore"), ("node_id", "=", my_ip)]
    )
    if len(local_actor) != 1:
        raise Exception(
            f"{len(local_actor)} local KVStore actors found, there should be 1"
        )
    return local_actor[0]


def get_actors(name, singleton=True):
    """ """
    actors = ray.util.state.list_actors(
        filters=[
            ("class_name", "=", name),
        ]
    )
    if singleton:
        if len(actors) != 1:
            raise Exception(f"{len(actors)} {name} actors found, there should be 1")
        return actors[0]
    else:
        return actors


def update_state_dict_from_store(kvstore, state_dict, version):
    """
    - Read state_dict from kvstore (async read and then get)
    - raise error if version doesn't match request
    """
    state_handles = {}
    for k, v in state_dict.items():
        offset = v.offset if isinstance(v, DTensor) else None
        state_handles[k] = kvstore.read.remote(k, offset, version)
    for k, v in state_handles.items():
        state_dict[k] = ray.get(v)
    return state_dict


def update_store_from_state_dict(kvstore, state_dict, version):
    """
    - Read state_dict from kvstore (async read and then get)
    - raise
    """
    handles = []
    for k, v in state_dict.items():
        handle = kvstore.write(k, v, version)
        handles.append(handle)
    return handles


def init_ray():
    """
    - init with resources needed for all actors (or take all resources?)
    - setup KVStore on each host
    """


def launch_actors(
    actor: ray.Actor,
    kwargs: Dict,
    num_actors: int = 1,
    resources: Optional[Dict] = None,
    distributed=True,
):
    """Launch a group of Ray actors"""
    name = type(actor).__name__
    if resources == None:
        resources = [{"CPU": 1}]
    pg = ray.util.placement_group(
        name=name, bundles=[resources] * num_actors, strategy="STRICT_PACK"
    )
    ray.get(pg.ready())  # Block until ready

    actors = []
    for i in range(num_actors):
        env_vars = {}
        if distributed:
            env_vars = distributed_vars(i, num_actors)
        a = actor.options(
            name=f"{name}_{i}",
            placement_group=pg,
            placement_group_bundle_index=i,
            runtime_env=env_vars,
        ).remote(**kwargs)
        actors.append(a)
    return actors


def distributed_vars(rank, world_size):
    return {
        "RANK": str(rank),
        "WORLD_SIZE": world_size,
        "MASTER_ADDR": utils.get_ip,
        "MASTER_PORT": utils.get_open_port,
    }
