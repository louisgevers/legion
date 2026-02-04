from typing import Protocol
from numpy.typing import ArrayLike

from legion.backend import Backend
from legion.embodiment import Embodiment
from legion.actuator import Actuator
from legion.physics import SensorData


class ObsTerm(Protocol):
    name: str
    required_signals: tuple[str, ...]
    size: int

    def __init__(
        self,
        backend: Backend,
        embodiment: Embodiment,
        actuator: Actuator,
        **kwargs,
    ): ...

    def __call__(
        self,
        signals: tuple[ArrayLike, ...],
        sensor_data: SensorData,
    ) -> ArrayLike: ...


class PrevActionObs:
    name = "prev_action"
    required_signals = ("prev_action",)

    def __init__(self, backend: Backend, embodiment: Embodiment, actuator: Actuator):
        self.size = actuator.n_u

    def __call__(
        self, signals: tuple[ArrayLike, ...], sensor_data: SensorData
    ) -> ArrayLike:
        prev_action = signals[0]
        return prev_action


class VelocityCommandObs:
    name = "velocity_command"
    required_signals = ("velocity_command",)
    size = 2

    def __init__(self, backend: Backend, embodiment: Embodiment, actuator: Actuator):
        pass

    def __call__(
        self, signals: tuple[ArrayLike, ...], sensor_data: SensorData
    ) -> ArrayLike:
        vel_cmd = signals[0]
        return vel_cmd
