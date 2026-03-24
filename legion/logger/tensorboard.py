# SPDX-FileCopyrightText: Copyright (c) 2026 Louis Gevers
# SPDX-License-Identifier: BSD-3-Clause
#
# See LICENSE file for full license information.
#
import logging
import tensorboardX


class TensorboardLogger:
    def __init__(self, log_dir: str):
        # Mute warnings about NaNs and Infs coming from unstable simulations
        logging.getLogger("tensorboardX.x2num").setLevel(logging.ERROR)
        self._writer = tensorboardX.SummaryWriter(log_dir)

    def log_metrics(self, step: int, metrics: dict[str, float]):
        for key, value in metrics.items():
            # Brax computes per step statistics when there's a _per_step suffix
            if key.endswith("_per_step"):
                key = key.removesuffix("_per_step").removeprefix("episode/")
            self._writer.add_scalar(key, value, step)

    def close(self):
        self._writer.close()
