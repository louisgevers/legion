from typing import Protocol
from numpy.typing import ArrayLike

from legion.registry import OBSERVATIONS, register
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


@register(OBSERVATIONS, "prev_action")
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


@register(OBSERVATIONS, "velocity_command")
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


@register(OBSERVATIONS, "q")
class JointPositionsObs:
    name = "q"
    required_signals = ()

    def __init__(self, backend: Backend, embodiment: Embodiment, actuator: Actuator):
        self.size = embodiment.n_joints
        self.q_nominal = backend.array(embodiment.q_nominal)

    def __call__(
        self, signals: tuple[ArrayLike, ...], sensor_data: SensorData
    ) -> ArrayLike:
        return sensor_data.q - self.q_nominal


@register(OBSERVATIONS, "dq")
class JointVelocitiesObs:
    name = "dq"
    required_signals = ()

    def __init__(self, backend: Backend, embodiment: Embodiment, actuator: Actuator):
        self.size = embodiment.n_joints

    def __call__(
        self, signals: tuple[ArrayLike, ...], sensor_data: SensorData
    ) -> ArrayLike:
        return sensor_data.dq


@register(OBSERVATIONS, "gravity_vector")
class GravityVectorObs:
    name = "gravity_vector"
    required_signals = ()

    def __init__(self, backend: Backend, embodiment: Embodiment, actuator: Actuator):
        self.backend = backend
        self.size = 3
        self.gravity_vector = self.backend.array([0, 0, -1])

    def __call__(
        self, signals: tuple[ArrayLike, ...], sensor_data: SensorData
    ) -> ArrayLike:
        vector = self.backend.quat_rotate(
            sensor_data.base_quat, self.gravity_vector, inverse=True
        )
        norm = self.backend.norm(vector)
        return vector / norm


@register(OBSERVATIONS, "base_linear_vel")
class BaseLinearVelocityObs:
    name = "base_linear_vel"
    required_signals = ()

    def __init__(self, backend: Backend, embodiment: Embodiment, actuator: Actuator):
        self.backend = backend
        self.size = 3

    def __call__(
        self, signals: tuple[ArrayLike, ...], sensor_data: SensorData
    ) -> ArrayLike:
        return sensor_data.local_base_linear_vel(self.backend)


@register(OBSERVATIONS, "base_angular_vel")
class BaseAngularVelocityObs:
    name = "base_angular_vel"
    required_signals = ()

    def __init__(self, backend: Backend, embodiment: Embodiment, actuator: Actuator):
        self.backend = backend
        self.size = 3

    def __call__(
        self, signals: tuple[ArrayLike, ...], sensor_data: SensorData
    ) -> ArrayLike:
        return sensor_data.local_base_angular_vel(self.backend)


@register(OBSERVATIONS, "foot_contacts")
class FootContactsObs:
    name = "foot_contacts"
    required_signals = ()

    def __init__(self, backend: Backend, embodiment: Embodiment, actuator: Actuator):
        self.size = embodiment.n_feet

    def __call__(
        self, signals: tuple[ArrayLike, ...], sensor_data: SensorData
    ) -> ArrayLike:
        return sensor_data.foot_contacts
