from numpy.typing import ArrayLike
from typing import Protocol, NamedTuple

from legion.backend import Backend
from legion.embodiment import Embodiment
from legion.physics import SensorData


class ActuatorState(NamedTuple):
    pass


class ActuatorOutput(NamedTuple):
    tau: ArrayLike
    state: ActuatorState


class Actuator(Protocol):

    def __init__(self, backend: Backend, embodiment: Embodiment, **kwargs): ...

    def reset(self) -> ActuatorState: ...
    def tau(
        self,
        u: ArrayLike,
        sensor_data: SensorData,
        state: ActuatorState,
    ) -> ActuatorOutput: ...
