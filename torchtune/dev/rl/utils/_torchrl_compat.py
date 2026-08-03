# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# torchrl >= 0.13 removed the collector classes the async RL data path
# subclasses. Import the real classes when available and fall back to
# import-compatible stand-ins so the data path can be imported and unit-tested
# against released torchrl.
try:
    from torchrl.collectors import (
        SyncDataCollector,
        WeightUpdateReceiverBase,
        WeightUpdateSenderBase,
    )
except ImportError:
    from torchrl.collectors import BaseCollector


    class SyncDataCollector(BaseCollector):
        """Import-compatible stand-in for ``torchrl.collectors.SyncDataCollector``.

        The stand-in accepts the constructor arguments of the original class
        and sets up the attributes the subclass relies on, but real data
        collection is not supported.
        """

        def __init__(self, *args, **kwargs):
            super().__init__()
            env = kwargs.get("create_env_fn")
            self.env = env() if callable(env) else env
            self.policy = kwargs.get("policy")
            self.frames_per_batch = kwargs.get("frames_per_batch", -1)
            self.total_frames = kwargs.get("total_frames", -1)
            self.weight_update_receiver = kwargs.get("weight_update_receiver")
            self.weight_update_sender = kwargs.get("weight_update_sender")
            self.reset_at_each_iter = kwargs.get("reset_at_each_iter", False)
            self.replay_buffer = None
            self._shuttle = None

        def _setup_data(self, *args, **kwargs):
            raise NotImplementedError(
                "SyncDataCollector is only available with torchrl < 0.13."
            )

        def _update_traj_ids(self, data):
            pass


    class WeightUpdateReceiverBase:
        pass


    class WeightUpdateSenderBase:
        pass
