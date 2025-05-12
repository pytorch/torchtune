# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# TODO: add read_set and write_set to KVStore
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

	def read(self, key, offset=None):
		"""
		- name and offset requested
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
		"""


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
		distributed=True
	):
	"""Launch a group of Ray actors"""
	name = type(actor).__name__
	if resources == None:
		resouces = [{"CPU": 1}]
	pg = ray.util.placement_group(
	    name=name,
	    bundles=[resources]*num_actors,
	    strategy="STRICT_PACK"
	)
	ray.get(pg.ready()) # Block until ready

	actors = []
	for i in range(num_gpus):
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
