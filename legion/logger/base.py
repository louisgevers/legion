# SPDX-FileCopyrightText: Copyright (c) 2026 Louis Gevers
# SPDX-License-Identifier: BSD-3-Clause
#
# See LICENSE file for full license information.
#
from typing import Protocol


class Logger(Protocol):

    def log_metrics(self, step: int, metrics: dict[str, float]): ...

    def close(self): ...
