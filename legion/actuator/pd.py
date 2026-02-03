from numpy.typing import ArrayLike

from legion.backend import Backend
from legion.embodiment import Embodiment
from legion.physics import SensorData

from .base import ActuatorState, ActuatorOutput


class PDActuator:
    def __init__(
        self,
        backend: Backend,
        embodiment: Embodiment,
        kp: float | list[float],
        kd: float | list[float],
        gain: float | list[float],
    ):
        self.backend = backend
        self.q_nominal = backend.array(embodiment.q_nominal)
        self.kp = backend.array(kp)
        self.kd = backend.array(kd)
        self.gain = backend.array(gain)

    def reset(self) -> ActuatorState:
        # No state
        return ActuatorState()

    def tau(
        self, u: ArrayLike, sensor_data: SensorData, state: ActuatorState
    ) -> ActuatorOutput:
        qdes = self.q_nominal + self.gain * u
        return ActuatorOutput(
            tau=self.kp * (qdes - sensor_data.q) - self.kd * sensor_data.dq,
            state=state,
        )
