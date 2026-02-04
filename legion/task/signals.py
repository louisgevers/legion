from typing import Protocol
from numpy.typing import ArrayLike

from legion.registry import SIGNALS, register
from legion.backend import Backend
from legion.embodiment import Embodiment
from legion.actuator import Actuator
from legion.physics import SensorData


class Signal(Protocol):
    name: str
    shape: tuple[int, ...]

    def __init__(
        self,
        name: str,
        backend: Backend,
        embodiment: Embodiment,
        actuator: Actuator,
        **kwargs,
    ): ...

    def reset(self) -> ArrayLike: ...
    def step(
        self,
        signal: ArrayLike,
        sensor_data: SensorData,
        action: ArrayLike,
    ) -> ArrayLike: ...


@register(SIGNALS, "prev_action")
class PreviousActionSignal:

    def __init__(
        self,
        name: str,
        backend: Backend,
        embodiment: Embodiment,
        actuator: Actuator,
    ):
        self.name = name
        self.shape = (actuator.n_u,)
        self.backend = backend

    def reset(self) -> ArrayLike:
        return self.backend.zeros(self.shape)

    def step(
        self,
        signal: ArrayLike,
        sensor_data: SensorData,
        action: ArrayLike,
    ) -> ArrayLike:
        return self.backend.array(action)


@register(SIGNALS, "fixed_value")
class FixedValuesSignal:

    def __init__(
        self,
        name: str,
        backend: Backend,
        embodiment: Embodiment,
        actuator: Actuator,
        values: list[float],
    ):
        self.name = name
        self.shape = (len(values),)
        self.values = tuple(values)
        self.backend = backend

    def reset(self) -> ArrayLike:
        return self.backend.array(self.values)

    def step(
        self,
        signal: ArrayLike,
        sensor_data: SensorData,
        action: ArrayLike,
    ):
        return signal
