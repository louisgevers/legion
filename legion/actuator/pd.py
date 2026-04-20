# SPDX-FileCopyrightText: Copyright (c) 2026 Louis Gevers
# SPDX-License-Identifier: BSD-3-Clause
#
# See LICENSE file for full license information.
#
from numpy.typing import ArrayLike

from legion.registry import ACTUATORS, register
from legion.backend import Backend, RNGKey
from legion.embodiment import Embodiment
from legion.physics import SensorData

from .base import ActuatorState


@register(ACTUATORS, "pd")
class PDActuator:
    def __init__(
        self,
        backend: Backend,
        embodiment: Embodiment,
        kp: float | list[float],
        kd: float | list[float],
        gain: float | list[float],
    ):
        self.backend = backend
        self.n_u = embodiment.n_actuators
        self.q_nominal = backend.array(embodiment.q_nominal)
        self.kp = backend.array(kp)
        self.kd = backend.array(kd)
        self.gain = backend.array(gain)

    def reset(self, rng: RNGKey) -> ActuatorState:
        # No state
        return ActuatorState()

    def step(
        self, u: ArrayLike, sensor_data: SensorData, state: ActuatorState, dt: float
    ) -> tuple[ArrayLike, ActuatorState]:
        qdes = self.q_nominal + self.gain * u
        return self.kp * (qdes - sensor_data.q) - self.kd * sensor_data.dq, state
