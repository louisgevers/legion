# SPDX-FileCopyrightText: Copyright (c) 2026 Louis Gevers
# SPDX-License-Identifier: BSD-3-Clause
#
# See LICENSE file for full license information.
#
import numpy as np
from typing import Protocol


from legion.physics import PhysicsState


class Renderer(Protocol):
    def render(self, state: PhysicsState) -> np.ndarray: ...
    def close(self): ...
