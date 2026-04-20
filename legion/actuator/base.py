# SPDX-FileCopyrightText: Copyright (c) 2026 Louis Gevers
# SPDX-License-Identifier: BSD-3-Clause
#
# See LICENSE file for full license information.
#
from numpy.typing import ArrayLike
from typing import Protocol, NamedTuple

from legion.backend import Backend, RNGKey
from legion.embodiment import Embodiment
from legion.physics import SensorData


class ActuatorState(NamedTuple):
    pass


class Actuator(Protocol):
    n_u: int

    def __init__(self, backend: Backend, embodiment: Embodiment, **kwargs): ...

    def reset(self, rng: RNGKey) -> ActuatorState: ...
    def step(self, state: ActuatorState, dt: float) -> ActuatorState: ...

    def tau(
        self,
        u: ArrayLike,
        sensor_data: SensorData,
        state: ActuatorState,
    ) -> ArrayLike: ...
