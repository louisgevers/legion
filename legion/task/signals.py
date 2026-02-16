from typing import Protocol
from numpy.typing import ArrayLike

from legion.registry import SIGNALS, register
from legion.backend import Backend, RNGKey
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

    def reset(self, rng: RNGKey) -> ArrayLike: ...
    def step(
        self,
        signal: ArrayLike,
        sensor_data: SensorData,
        action: ArrayLike,
        dt: float,
        rng: RNGKey,
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

    def reset(self, rng: RNGKey) -> ArrayLike:
        return self.backend.zeros(self.shape)

    def step(
        self,
        signal: ArrayLike,
        sensor_data: SensorData,
        action: ArrayLike,
        dt: float,
        rng: RNGKey,
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

    def reset(self, rng: RNGKey) -> ArrayLike:
        return self.backend.array(self.values)

    def step(
        self,
        signal: ArrayLike,
        sensor_data: SensorData,
        action: ArrayLike,
        dt: float,
        rng: RNGKey,
    ):
        return signal


@register(SIGNALS, "episode_uniform_value")
class EpisodeUniformValueSignal:

    def __init__(
        self,
        name: str,
        backend: Backend,
        embodiment: Embodiment,
        actuator: Actuator,
        min_values: list[float],
        max_values: list[float],
    ):
        assert len(min_values) == len(
            max_values
        ), "Min and max values length does not match"

        self.name = name
        self.shape = (len(min_values),)
        self.backend = backend

        self.min_values = self.backend.array(min_values)
        self.max_values = self.backend.array(max_values)

    def reset(self, rng: RNGKey) -> ArrayLike:
        values = self.backend.rng_uniform(rng, self.shape)
        return self.min_values + (self.max_values - self.min_values) * values

    def step(
        self,
        signal: ArrayLike,
        sensor_data: SensorData,
        action: ArrayLike,
        dt: float,
        rng: RNGKey,
    ):
        return signal


@register(SIGNALS, "feet_contact_signals")
class FeetContactSignals:
    """Stateful signal tracking of
    - air time per foot (n_feet,)
    - previous contact per foot (n_feet,)
    """

    def __init__(
        self,
        name: str,
        backend: Backend,
        embodiment: Embodiment,
        actuator: Actuator,
    ):
        self.name = name
        self.backend = backend
        self.n_feet = embodiment.n_feet

        self.shape = (2 * self.n_feet,)

    def reset(self, rng: RNGKey) -> ArrayLike:
        return self.backend.zeros(self.shape)

    def step(
        self,
        signal: ArrayLike,
        sensor_data: SensorData,
        action: ArrayLike,
        dt: float,
        rng: RNGKey,
    ):
        air_time = signal[: self.n_feet]
        prev_contact = signal[self.n_feet :]
        contact = sensor_data.foot_contacts.astype(air_time.dtype)

        # Increment air time if no contact
        new_air_time = self.backend.where(contact > 0.5, 0.0, air_time + dt)

        return self.backend.concatenate([new_air_time, contact], axis=0)
