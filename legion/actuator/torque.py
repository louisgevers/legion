from numpy.typing import ArrayLike

from legion.backend import Backend
from legion.embodiment import Embodiment
from legion.physics import SensorData

from .base import ActuatorState


class TorqueActuator:
    def __init__(
        self, backend: Backend, embodiment: Embodiment, gain: float | list[float]
    ):
        self.backend = backend
        self.n_u = embodiment.n_actuators
        self.gain = backend.array(gain)

    def reset(self) -> ActuatorState:
        # No state
        return ActuatorState()

    def step(self, state: ActuatorState) -> ActuatorState:
        # No state
        return state

    def tau(
        self, u: ArrayLike, sensor_data: SensorData, state: ActuatorState
    ) -> ArrayLike:
        return u * self.gain
