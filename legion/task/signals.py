# SPDX-FileCopyrightText: Copyright (c) 2026 Louis Gevers
# SPDX-License-Identifier: BSD-3-Clause
#
# See LICENSE file for full license information.
#
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


@register(SIGNALS, "resampled_uniform_value")
class ResampledUniformValueSignal:
    """Stateful resampled values from uniform distribution
    - random uniform values
    - remaining time before next resampling
    """

    def __init__(
        self,
        name: str,
        backend: Backend,
        embodiment: Embodiment,
        actuator: Actuator,
        max_resample_duration: float,
        min_values: float | list[float],
        max_values: float | list[float],
        p_nonzero: float | list[float] = 0.0,
    ):
        if type(min_values) == float:
            min_values = [min_values]
            max_values = [max_values]
        assert len(min_values) == len(
            max_values
        ), "Min and max values length does not match"

        self.name = name
        self.shape = (len(min_values),)
        self.backend = backend

        self.max_resample_duration = max_resample_duration
        self.min_values = self.backend.array(min_values)
        self.max_values = self.backend.array(max_values)
        self.p_nonzero = self.backend.array(p_nonzero)

    def reset(self, rng: RNGKey) -> ArrayLike:
        key1, key2, key3 = self.backend.rng_split(rng, 3)
        values = self._sample_values(key1, key2)
        sample_time = self._sample_resample_time(key3)
        return self.backend.concatenate([values, sample_time], axis=0)

    def step(
        self,
        signal: ArrayLike,
        sensor_data: SensorData,
        action: ArrayLike,
        dt: float,
        rng: RNGKey,
    ):
        values = signal[:-1]
        sample_time = signal[-1] - dt

        key1, key2, key3 = self.backend.rng_split(rng, 3)
        next_values = self.backend.where(
            sample_time <= 0, self._sample_values(key1, key2), values
        )
        next_times = self.backend.where(
            sample_time <= 0, self._sample_resample_time(key3), sample_time
        )

        return self.backend.concatenate([next_values, next_times])

    def _sample_values(self, key1: RNGKey, key2: RNGKey) -> ArrayLike:
        values = self.backend.rng_uniform(key1, self.shape)
        scaled_values = self.min_values + (self.max_values - self.min_values) * values
        # Set the values to 0 with certain probability
        zero_out = self.backend.rng_bernoulli(key2, self.p_nonzero, self.shape)

        return scaled_values * zero_out

    def _sample_resample_time(self, rng: RNGKey) -> ArrayLike:
        sample_time = (
            self.backend.rng_exponential(rng, (1,)) * self.max_resample_duration
        )
        return sample_time


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
        # Increment air time if no contact
        air_time = signal[: self.n_feet]
        prev_contact = signal[self.n_feet :]
        contact = sensor_data.foot_contacts.astype(air_time.dtype)
        new_air_time = self.backend.where(contact > 0.5, 0.0, air_time + dt)

        return self.backend.concatenate([new_air_time, contact], axis=0)
