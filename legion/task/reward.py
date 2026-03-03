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
        cmd = signals[0][:2]
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
        cmd = signals[0][0]
        actual = sensor_data.local_base_angular_vel(self.backend)[2]  # wz
        error = self.backend.sum(self.backend.square(actual - cmd))
        return self.backend.exp(-error / self.sensitivity)


@register(REWARDS, "nominal_pose_tracking")
class NominalPoseTrackingReward:
    name = "nominal_pose_tracking"
    required_signals = ()

    def __init__(
        self,
        backend: Backend,
        embodiment: Embodiment,
        actuator: Actuator,
        weight: float,
        sensitivity: float,
        q_weights: list[float],
    ):
        self.backend = backend
        self.weight = weight
        self.sensitivity = sensitivity
        self.q_nominal = self.backend.array(embodiment.q_nominal)

        # Repeat q_weights if necessary to match q_nominal
        q_weights = self.backend.array(q_weights)
        self.q_weights = self.backend.tile(
            q_weights, len(self.q_nominal) // len(q_weights)
        )

    def __call__(
        self,
        signals: ArrayLike,
        sensor_data: SensorData,
        action: ArrayLike,
    ):
        err = self.backend.sum(
            self.backend.square(self.q_nominal - sensor_data.q) * self.q_weights
        )
        return self.backend.exp(-err / self.sensitivity)


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


@register(REWARDS, "orientation_penalty")
class OrientationPenalty:
    name = "orientation_penalty"
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
        self.gravity_vector = self.backend.array([0, 0, -1])

    def __call__(
        self,
        signals: ArrayLike,
        sensor_data: SensorData,
        action: ArrayLike,
    ):
        # Compute gravity vector
        gravity = self.backend.quat_rotate(
            sensor_data.base_quat, self.gravity_vector, inverse=True
        )
        # Penalize any offsets in xy orientations
        return self.backend.sum(self.backend.square(gravity[:2]))


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


@register(REWARDS, "energy_penalty")
class EnergyPenalty:
    name = "energy_penalty"
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
        return self.backend.sum(self.backend.abs(sensor_data.tau * sensor_data.dq))


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
        self.backend = backend

    def __call__(
        self,
        signals: ArrayLike,
        sensor_data: SensorData,
        action: ArrayLike,
    ):
        return sensor_data.n_contacts - self.backend.sum(sensor_data.foot_contacts)


@register(REWARDS, "feet_air_time")
class FeetAirTimeReward:
    name = "feet_air_time"
    required_signals = (
        "feet_contact_signals",
        "linear_velocity_command",
        "angular_velocity_command",
    )

    def __init__(
        self,
        backend: Backend,
        embodiment: Embodiment,
        actuator: Actuator,
        weight: float,
        min_time: float,
        min_cmd_amplitude: float = 0.1,
    ):
        self.backend = backend
        self.weight = weight
        self.n_feet = embodiment.n_feet
        self.min_time = min_time
        self.min_cmd_amplitude = min_cmd_amplitude

    def __call__(
        self,
        signals: ArrayLike,
        sensor_data: SensorData,
        action: ArrayLike,
    ):
        air_time = signals[0][: self.n_feet]
        prev_contact = signals[0][self.n_feet :]

        cmd = self.backend.concatenate([signals[1][:2], signals[2][:1]], axis=0)
        cmd_large_enough = self.backend.norm(cmd) > self.min_cmd_amplitude

        # Detect first contact
        contact = sensor_data.foot_contacts
        contact_filt = (contact > 0.5) & (
            prev_contact < 0.5
        )  # Note prev contact is a float
        first_contact = (air_time > 0.0) * contact_filt

        rew_air_time = self.backend.sum((air_time - self.min_time) * first_contact)
        return rew_air_time * cmd_large_enough  # No air time required for zero commands
