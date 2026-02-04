from typing import Protocol
from numpy.typing import ArrayLike

from legion.backend import Backend
from legion.embodiment import Embodiment
from legion.actuator import Actuator
from legion.physics import SensorData


class TerminationTerm(Protocol):
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
    ) -> bool: ...


class FallTermination:
    name = "fall"
    required_signals = ()

    def __init__(
        self,
        backend: Backend,
        embodiment: Embodiment,
        actuator: Actuator,
        min_height: float,
    ):
        self.height = min_height

    def __call__(self, signals: ArrayLike, sensor_data: SensorData):
        return sensor_data.base_xyz[2] < self.height
