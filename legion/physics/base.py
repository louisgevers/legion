from typing import Protocol, NamedTuple
from numpy.typing import ArrayLike

from legion.backend import Backend


class PhysicsState(NamedTuple):
    data: object  # internal simulator state


class SensorData(NamedTuple):
    t: ArrayLike  # current simulation time (1,)

    q: ArrayLike  # joint positions (n_joints,)
    dq: ArrayLike  # joint velocities (n_joints,)

    tau: ArrayLike  # applied torques (n_actuators,)

    base_xyz: ArrayLike  # base position in world frame (3,)
    base_quat: ArrayLike  # base orientation quaternion (4,)
    base_linear_vel: ArrayLike  # base linear velocity (3,)
    base_angular_vel: ArrayLike  # base angular velocity (3,)


class PhysicsEngine(Protocol):
    name: str
    backend: Backend

    # Model metadata (extracted from the physics)
    @property
    def joint_names(self) -> tuple[str, ...]: ...
    @property
    def actuator_names(self) -> tuple[str, ...]: ...
    @property
    def n_joints(self) -> int: ...
    @property
    def n_actuators(self) -> int: ...

    # Simulation state functions
    def reset(self) -> PhysicsState: ...
    def step(self, state: PhysicsState) -> PhysicsState: ...
    def apply_torques(self, state: PhysicsState, tau: ArrayLike) -> PhysicsState: ...
    def get_sensor_data(self, state: PhysicsState) -> SensorData: ...
