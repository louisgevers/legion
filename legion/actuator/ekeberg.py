# SPDX-FileCopyrightText: Copyright (c) 2026 Louis Gevers
# SPDX-License-Identifier: BSD-3-Clause
#
# See LICENSE file for full license information.
#
from typing import Literal
from numpy.typing import ArrayLike

from legion.registry import ACTUATORS, register
from legion.backend import Backend, RNGKey
from legion.embodiment import Embodiment
from legion.physics import SensorData

from .base import ActuatorState


@register(ACTUATORS, "ekeberg")
class EkebergActuator:
    def __init__(
        self,
        backend: Backend,
        embodiment: Embodiment,
        alpha: float,
        beta: float,
        gamma: float,
        delta: float,
        control_mode: Literal["individual", "antagonist"] = "individual",
        theta_ref: tuple[float] | None = None,
    ):
        self.backend = backend
        self.n_u = embodiment.n_actuators * 2
        self.theta_ref = (
            backend.array(embodiment.q_nominal)
            if theta_ref is None
            else self.backend.array(theta_ref)
        )
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

        self._u_diff_sum_fn = {
            "individual": u_diff_sum_individual,
            "antagonist": u_diff_sum_antagonist,
        }[control_mode]

    def reset(self, rng: RNGKey, dt: float) -> ActuatorState:
        # No state
        return ActuatorState()

    def step(self, state: ActuatorState) -> ActuatorState:
        # No state
        return state

    def tau(
        self, u: ArrayLike, sensor_data: SensorData, state: ActuatorState
    ) -> ArrayLike:
        # Split action into flexor/extensor difference and summation
        u_diff, u_sum = self._u_diff_sum_fn(u)

        # Deviation from reference angle
        delta_theta = self.theta_ref - sensor_data.q

        # Active and passive torque contributions
        tau_active = self.alpha * u_diff + self.beta * u_sum * delta_theta
        tau_passive = self.beta * self.gamma * delta_theta - self.delta * sensor_data.dq

        return tau_active + tau_passive


def u_diff_sum_individual(u: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
    # Map to [0, 1] range
    u_scaled = (u + 1) / 2

    # Split action into flexors/extensors
    n_u = len(u)
    u1 = u_scaled[: n_u // 2]
    u2 = u_scaled[n_u // 2 :]

    # Return difference and sum
    return (u1 - u2, u1 + u2)


def u_diff_sum_antagonist(u: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
    # Split action directly into difference and sum
    n_u = len(u)
    u_diff = u[: n_u // 2]
    u_sum = u[n_u // 2 :]

    # Sum is additive
    u_sum += 1

    return (u_diff, u_sum)
