# SPDX-FileCopyrightText: Copyright (c) 2026 Louis Gevers
# SPDX-License-Identifier: BSD-3-Clause
#
# See LICENSE file for full license information.
#
from typing import Protocol, Literal
from numpy.typing import ArrayLike

from legion.registry import METRICS, register
from legion.backend import Backend
from legion.embodiment import Embodiment
from legion.actuator import Actuator
from legion.physics import SensorData


# Same signature as a reward term but without weight
class MetricsTerm(Protocol):
    name: str
    required_signals: tuple[str, ...]

    def __init__(
        self,
        backend: Backend,
        embodiment: Embodiment,
        actuator: Actuator,
        **kwargs,
    ): ...

    def __call__(
        self,
        signals: ArrayLike,
        sensor_data: SensorData,
        action: ArrayLike,
    ) -> float: ...


@register(METRICS, "energy")
class EnergyMetric:
    name = "energy"
    required_signals = ()

    def __init__(
        self,
        backend: Backend,
        embodiment: Embodiment,
        actuator: Actuator,
    ):
        self.backend = backend

    def __call__(
        self,
        signals: ArrayLike,
        sensor_data: SensorData,
        action: ArrayLike,
    ):
        return self.backend.sum(self.backend.abs(sensor_data.tau * sensor_data.dq))


@register(METRICS, "distance_x")
class DistanceMetric:
    name = "distance_x"
    required_signals = ()

    def __init__(
        self,
        backend: Backend,
        embodiment: Embodiment,
        actuator: Actuator,
    ):
        self.backend = backend

    def __call__(
        self,
        signals: ArrayLike,
        sensor_data: SensorData,
        action: ArrayLike,
    ):
        return sensor_data.base_linear_vel[0]  # Integrated at each step


@register(METRICS, "distance_y")
class DistanceMetric:
    name = "distance_y"
    required_signals = ()

    def __init__(
        self,
        backend: Backend,
        embodiment: Embodiment,
        actuator: Actuator,
    ):
        self.backend = backend

    def __call__(
        self,
        signals: ArrayLike,
        sensor_data: SensorData,
        action: ArrayLike,
    ):
        return sensor_data.base_linear_vel[1]  # Integrated at each step
