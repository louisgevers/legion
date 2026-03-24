# SPDX-FileCopyrightText: Copyright (c) 2026 Louis Gevers
# SPDX-License-Identifier: BSD-3-Clause
#
# See LICENSE file for full license information.
#
from typing import Protocol

from legion.physics import PhysicsState


class Viewer(Protocol):
    def render(self, state: PhysicsState): ...
    def is_running(self) -> bool: ...
    def close(self): ...
