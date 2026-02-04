from typing import Protocol
from numpy.typing import ArrayLike

from legion.backend import Backend
from legion.embodiment import Embodiment
from legion.actuator import Actuator
from legion.physics import SensorData


class RewardTerm(Protocol):
    name: str
    required_signals: tuple[str, ...]
    weight: float

    def __init__(
        self,
        backend: Backend,
        embodiment: Embodiment,
        actuator: Actuator,
        weight: float,
        **kwargs,
    ): ...

    def __call__(
        self,
        signals: ArrayLike,
        sensor_data: SensorData,
        action: ArrayLike,
    ) -> float: ...


class VelocityTrackingReward:
    name = "velocity_tracking"
    required_signals = ("velocity_command",)

    def __init__(
        self,
        backend: Backend,
        embodiment: Embodiment,
        actuator: Actuator,
        weight: float,
        sensitivity: float,
    ):
        self.backend = backend
        self.weight = weight
        self.sensitivity = sensitivity

    def __call__(
        self,
        signals: ArrayLike,
        sensor_data: SensorData,
        action: ArrayLike,
    ):
        cmd = signals[0]
        actual = sensor_data.base_xyz[:2]  # vx, vy
        diff = self.backend.norm(actual - cmd)
        return self.backend.exp(-(diff**2) / self.sensitivity)


class ActionRegularizationReward:
    name = "action_regularization"
    required_signals = ("prev_action",)

    def __init__(
        self,
        backend: Backend,
        embodiment: Embodiment,
        actuator: Actuator,
        weight: float,
    ):
        self.backend = backend
        self.weight = weight

    def __call__(
        self,
        signals: ArrayLike,
        sensor_data: SensorData,
        action: ArrayLike,
    ):
        prev_action = signals[0]
        return self.backend.norm(action - prev_action)
