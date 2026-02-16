from typing import Protocol
from numpy.typing import ArrayLike

from legion.registry import REWARDS, register
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


@register(REWARDS, "linear_velocity_tracking")
class LinearVelocityTrackingReward:
    name = "linear_velocity_tracking"
    required_signals = ("linear_velocity_command",)

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
        actual = sensor_data.local_base_linear_vel(self.backend)[:2]  # vx, vy
        error = self.backend.sum(self.backend.square(actual - cmd))
        return self.backend.exp(-error / self.sensitivity)


@register(REWARDS, "angular_velocity_tracking")
class AngularVelocityTrackingReward:
    name = "angular_velocity_tracking"
    required_signals = ("angular_velocity_command",)

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
        actual = sensor_data.local_base_angular_vel(self.backend)[2]  # wz
        error = self.backend.sum(self.backend.square(actual - cmd))
        return self.backend.exp(-error / self.sensitivity)


@register(REWARDS, "action_regularization")
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
        return self.backend.sum(self.backend.square(action - prev_action))


@register(REWARDS, "linear_velocity_penalty")
class LinearVelocityPenalty:
    name = "linear_velocity_penalty"
    required_signals = ()

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
        lin_vel_z = sensor_data.local_base_linear_vel(self.backend)[2]
        return self.backend.square(lin_vel_z)


@register(REWARDS, "angular_velocity_penalty")
class AngularVelocityPenalty:
    name = "angular_velocity_penalty"
    required_signals = ()

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
        ang_vel_xy = sensor_data.local_base_angular_vel(self.backend)[:2]  # wx, wy
        return self.backend.sum(self.backend.square(ang_vel_xy))


@register(REWARDS, "joint_velocity_penalty")
class JointVelocityPenalty:
    name = "joint_velocity_penalty"
    required_signals = ()

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
        return self.backend.sum(self.backend.square(sensor_data.dq))


@register(REWARDS, "joint_acceleration_penalty")
class JointAccelerationPenalty:
    name = "joint_acceleration_penalty"
    required_signals = ()

    def __init__(
        self,
        backend: Backend,
        embodiment: Embodiment,
        actuator: Actuator,
        weight: float,
        clip_acc: float = 100,  # Clip large accelerations to account for unstable simulations
    ):
        self.backend = backend
        self.weight = weight
        self.clip_acc = clip_acc

    def __call__(
        self,
        signals: ArrayLike,
        sensor_data: SensorData,
        action: ArrayLike,
    ):
        clipped_acc = self.backend.clip(
            sensor_data.ddq, min=-self.clip_acc, max=self.clip_acc
        )
        return self.backend.sum(self.backend.square(clipped_acc))


@register(REWARDS, "joint_torque_penalty")
class JointTorquePenalty:
    name = "joint_torque_penalty"
    required_signals = ()

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
        return self.backend.sum(self.backend.square(sensor_data.tau))


@register(REWARDS, "contacts_penalty")
class ContactsPenalty:
    name = "contacts_penalty"
    required_signals = ()

    def __init__(
        self,
        backend: Backend,
        embodiment: Embodiment,
        actuator: Actuator,
        weight: float,
    ):
        self.weight = weight

    def __call__(
        self,
        signals: ArrayLike,
        sensor_data: SensorData,
        action: ArrayLike,
    ):
        return sensor_data.n_contacts


@register(REWARDS, "feet_air_time")
class FeetAirTimeReward:
    name = "feet_air_time"
    required_signals = ("feet_contact_signals",)

    def __init__(
        self,
        backend: Backend,
        embodiment: Embodiment,
        actuator: Actuator,
        weight: float,
        min_time: float,
    ):
        self.backend = backend
        self.weight = weight
        self.n_feet = embodiment.n_feet
        self.min_time = min_time

    def __call__(
        self,
        signals: ArrayLike,
        sensor_data: SensorData,
        action: ArrayLike,
    ):
        air_time = signals[0][: self.n_feet]
        prev_contact = signals[0][self.n_feet :]

        # Detect first contact
        contact = sensor_data.foot_contacts.astype(air_time.dtype)
        first_contact = (contact > 0.5) & (prev_contact < 0.5)

        return self.backend.sum((air_time - self.min_time) * first_contact)
