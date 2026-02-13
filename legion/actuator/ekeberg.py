from numpy.typing import ArrayLike

from legion.registry import ACTUATORS, register
from legion.backend import Backend
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
    ):
        self.backend = backend
        self.n_u = embodiment.n_actuators * 2
        self.theta_ref = backend.array(embodiment.q_nominal)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def reset(self) -> ActuatorState:
        # No state
        return ActuatorState()

    def step(self, state: ActuatorState) -> ActuatorState:
        # No state
        return state

    def tau(
        self, u: ArrayLike, sensor_data: SensorData, state: ActuatorState
    ) -> ArrayLike:
        # Map to [0, 1] range
        u_scaled = (u + 1) / 2

        # Split action into flexors/extensors
        u1 = u_scaled[: self.n_u // 2]
        u2 = u_scaled[self.n_u // 2 :]

        # Deviation from reference angle
        delta_theta = self.theta_ref - sensor_data.q

        # Active and passive torque contributions
        tau_active = self.alpha * (u1 - u2) + self.beta * (u1 + u2) * delta_theta
        tau_passive = self.beta * self.gamma * delta_theta - self.delta * sensor_data.dq

        return tau_active + tau_passive
