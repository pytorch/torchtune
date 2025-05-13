# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import ray
import functools
from OmegaConf import DictConfig

from torchtune import config
from torchtune.dev.rl.ray import (
    launch_actors,
    update_state_dict_from_store,
    update_store_from_state_dict,
    get_kvstore,
    init_ray,
    get_actors,
)

# TODO: step by step walkthrough of vllm
@ray.remote
class GRPOGeneratorDistributed():
    # Data collector
    # envs from configs (include model transform)
    # policy
    # reward workers?

    def prefetch(self, state_dict):
    # check version of all keys, if len(version) == 1 and version != self.version then fetch
    # new_weights = {k: self.store(k) for k,v in state_dict} # this owned by generator, can't be invalidated
    # self.model.sync_weights



# TODO: step by step walkthrough of trainer worker
@ray.remote
class GRPOTrainerDistributed:
    # match SFTDistributed except:
    # init -> ParameterStore
    # setup_data -> ReplayBuffer
    # train -> offload weights
    # logging

    def setup(self)
        # add store actor
        # find metric logger line
        # find setup data
        # add logging method

    def sync_weights(self, model, step):
        state_dict = {}
        for k,v in model.state_dict():
            # async offload to cpu
            cpu_weights = torch.cuda.Stream()  # Create a stream
            with torch.no_grad():
                async_copy = v.to('cpu', non_blocking=True, stream=cpu_weights)
                state_dict[k] = async_copy
        # figure out how to have update_store_from_state_dict work with the streams
        return update_store_from_state_dict(self.kvstore, state_dict, step)


    def opt_step(self, step):
        # lost of half broken code here
        if self.offload_handle is not None:
            await self.sync_handles
        self.opt.step()
        self.sync_handles = async sync_weights(self.model, step)

    def train(...):

        for ... in self.replay_buffer:
            ...
            loss = self.loss_step(...)
            loss.backward()
            ...


            self.opt.step(step)

# TODO
# show train offload -> save to store
#   - only save to store if vllm updated
#   - don't call opt_step while still offloading
#   - start offload -> move to store -> store marks new version (per offset) (does store propagate copies to past copies or invalidate?) -> block update unitl offloaded and no new offload until vllm pulls
#       - do you want two copies to be able to exist if vllm and trainer are on the same machine (i.e. read always creates a copy (write location and read location, reads get invalidated with new write))
#
# actors get other actors from ray.state


def launch_replay_buffer(cfg):
    ray_buffer = ray.remote(**cfg.replay_buffer.orchestration)(ReplayBuffer)

    storage = functools.partial(LazyStackStorage, max_size=cfg.replay_buffer.size)
    batch_size = cfg.training.batch_size
    # bs and collate_fn should come from inside trainer
    # ideal setting: ReplayBuffer exists in trainer, storage on object store for other actors to update
    return ray_buffer.options(name="ReplayBuffer").remote(storage, batch_size)


def launch_metric_logger(cfg):
    logger_cls = cfg.metric_logger.pop("_component_")
    ray_logger = ray.remote(**cfg.metric_logger.orchestration)(logger_cls)
    return ray_logger.options(name="MetricLogger").remote(**cfg.metric_logger)


@config.parse
def recipe_main(cfg: DictConfig) -> None:
    """
    Entry point for the recipe.

    Configurable parameters are read in the following order:
        - Parameters specified in config (see available configs through ``tune ls``)
        - Overwritten by arguments from the command-line
    """
    config.log_config(recipe_name="FullAsyncGRPODistributed", cfg=cfg)

    init_ray()

    base_cfg = cfg - cfg.orchestration - cfg.collectors - cfg.post_processing - cfg.training - cfg.metric_logger
    collectors = launch_actors(GRPOGeneratorDistributed, base_cfg + cfg.collector, **cfg.collector.resources)
    postprocessing = []
    for proccess_cfg in cfg.postprocessing:
        processes = launch_actors(RewardActor, base_cfg + process_cfg, **proccess_cfg.resources)
        postprocessing.append(processes)
    trainers = launch_actors(GRPOTrainerDistributed, base_cfg + cfg.training, **cfg.training.resources)

    launch_replay_buffer(cfg)
    launch_metric_logger(cfg)
    ray.util.queue.Queue(actor_options={"name": "PostprocessQueue"})

    collector_handles = [collector.run.remote() for collector in collectors]
    process_handles = [process.run.remote() for process in postprocessing]
    trainer_handles = [trainer.train.remote() for trainer in trainers]
    ray.get(trainer_handles)
    [ray.kill(w) for w in collector_handles + process_handles]
    ray.get(trainers[0].cleanup.remote())

    ray.shutdown()


    if __name__ == "__main__":
    sys.exit(recipe_main())
