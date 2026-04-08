# SPDX-FileCopyrightText: Copyright (c) 2026 Louis Gevers
# SPDX-License-Identifier: BSD-3-Clause
#
# See LICENSE file for full license information.
#
from typing import Protocol, NamedTuple
from numpy.typing import ArrayLike

from legion.backend import Backend
from legion.embodiment import Embodiment


class PhysicsState(NamedTuple):
    data: object  # internal simulator state


class SensorData(NamedTuple):
    t: ArrayLike  # current simulation time (1,)

    q: ArrayLike  # joint positions (n_joints,)
    dq: ArrayLike  # joint velocities (n_joints,)
    ddq: ArrayLike  # joint accelerations (n_joints,)

    tau: ArrayLike  # applied torques (n_actuators,)

    base_xyz: ArrayLike  # base position in world frame (3,)
    base_quat: ArrayLike  # base orientation quaternion xyzw (4,)
    base_linear_vel: ArrayLike  # base linear velocity (3,)
    base_angular_vel: ArrayLike  # base angular velocity (3,)

    n_contacts: ArrayLike  # number of contacts (1,)
    foot_contacts: ArrayLike  # boolean foot contacts (n_feet,)
    foot_normal_forces: ArrayLike  # normal contact force magnitude per foot (n_feet,)

    def local_base_linear_vel(self, backend: Backend) -> ArrayLike:
        return backend.quat_rotate(self.base_quat, self.base_linear_vel, inverse=True)

    def local_base_angular_vel(self, backend: Backend) -> ArrayLike:
        return backend.quat_rotate(self.base_quat, self.base_angular_vel, inverse=True)


class PhysicsEngine(Protocol):
    name: str
    backend: Backend
    dt: float

    def __init__(self, embodiment: Embodiment, dt: float, **kwargs): ...

    # Simulation state functions
    def reset(self, q: ArrayLike, base_xyz: ArrayLike) -> PhysicsState: ...
    def step(self, state: PhysicsState) -> PhysicsState: ...
    def apply_torques(self, state: PhysicsState, tau: ArrayLike) -> PhysicsState: ...
    def get_sensor_data(self, state: PhysicsState) -> SensorData: ...

    # Domain randomization handles
    def set_ground_friction(
        self, state: PhysicsState, friction: float
    ) -> PhysicsState: ...
    def add_base_mass(self, state: PhysicsState, mass: float) -> PhysicsState: ...
    def scale_masses(self, state: PhysicsState, scales: ArrayLike) -> PhysicsState: ...
    def offset_joints(
        self, state: PhysicsState, offsets: ArrayLike
    ) -> PhysicsState: ...
    def apply_base_perturbation(
        self, state: PhysicsState, force: ArrayLike
    ) -> PhysicsState: ...
