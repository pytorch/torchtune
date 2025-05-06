# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import ray
from torchtune import config


@ray.remote(num_cpus=1, num_gpus=0)
class MetricLoggerWorker:
    def __init__(self, cfg):
        self.logger = config.instantiate(cfg.metric_logger)
        self.logger.log_config(cfg)

    def log_dict(self, log_dict, step=None):
        # allowing actors to use their own step counters
        self.logger.log_dict(log_dict, step=step)

    def log_table(self, table_data, columns, table_name, step=None):
        """Log a table to WandB."""
        import wandb

        table = wandb.Table(columns=columns, data=table_data)
        self.logger.log_dict({table_name: table}, step=step)

    def close(self):
        if hasattr(self.logger, "close"):
            self.logger.close()
