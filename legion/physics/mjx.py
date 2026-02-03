import mujoco
from mujoco import mjx
from numpy.typing import ArrayLike

from legion.backend import get_backend

from .base import PhysicsState, SensorData
from .mujoco import mj_base_indices, mj_joint_indices, mj_actuator_names


class MJXPhysics:
    name = "mjx"

    def __init__(self, mjcf: str):
        # Lazy load backend
        self.backend = get_backend("jax")

        # Load MJCF
        self._mj_model = mujoco.MjModel.from_xml_path(mjcf)

        # Create mjx model
        self._mjx_model = mjx.put_model(self._mj_model)

        # Precompute joint indices
        self._base_qpos_idx, self._base_qvel_idx = mj_base_indices(
            self._mj_model, self.backend
        )
        self._joint_names, self._joint_qpos_idx, self._joint_qvel_idx = (
            mj_joint_indices(self._mj_model, self.backend)
        )
        self._actuator_names = mj_actuator_names(self._mj_model)

    @property
    def joint_names(self):
        return self._joint_names

    @property
    def actuator_names(self):
        return self._actuator_names

    @property
    def n_joints(self):
        return len(self._joint_names)

    @property
    def n_actuators(self):
        return len(self._actuator_names)

    def reset(self) -> PhysicsState:
        # Create blank data
        data = mjx.make_data(self._mj_model)
        data = mjx.forward(self._mjx_model, data)
        return PhysicsState(data=data)

    def step(self, state: PhysicsState) -> PhysicsState:
        data = mjx.step(self._mjx_model, state.data)
        return PhysicsState(data=data)

    def apply_torques(self, state: PhysicsState, tau: ArrayLike) -> PhysicsState:
        return PhysicsState(data=state.data.replace(ctrl=tau))

    def get_sensor_data(self, state: PhysicsState) -> SensorData:
        return SensorData(
            t=self.backend.array(state.data.time),
            q=self.backend.array(state.data.qpos[self._joint_qpos_idx]),
            dq=self.backend.array(state.data.qvel[self._joint_qvel_idx]),
            tau=self.backend.array(state.data.ctrl),
            base_xyz=self.backend.array(state.data.qpos[self._base_qpos_idx][:3]),
            base_quat=self.backend.array(state.data.qpos[self._base_qpos_idx][3:]),
            base_linear_vel=self.backend.array(
                state.data.qvel[self._base_qvel_idx][:3]
            ),
            base_angular_vel=self.backend.array(
                state.data.qvel[self._base_qvel_idx][3:]
            ),
        )
