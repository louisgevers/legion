import mujoco
from mujoco import mjx
from numpy.typing import ArrayLike

from legion.backend import get_backend

from .base import PhysicsState, SensorData


class MJXPhysics:
    name = "mjx"

    def __init__(self, mjcf: str):
        # Lazy load backend
        self.backend = get_backend("jax")

        # Load MJCF
        self._mj_model = mujoco.MjModel.from_xml_path(mjcf)

        # Create mjx model
        self._mjx_model = mjx.put_model(self._mj_model)

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
        base_pos_idx = self._mj_model.joint("base").qposadr[0]
        base_vel_idx = self._mj_model.joint("base").dofadr[0]
        return SensorData(
            t=self.backend.array(state.data.time),
            q=self.backend.array(state.data.qpos),
            dq=self.backend.array(state.data.qvel),
            tau=self.backend.array(state.data.ctrl),
            base_xyz=self.backend.array(
                state.data.qpos[base_pos_idx : base_pos_idx + 3]
            ),
            base_quat=self.backend.array(
                state.data.qpos[base_pos_idx + 3 : base_pos_idx + 7]
            ),
            base_linear_vel=self.backend.array(
                state.data.qvel[base_vel_idx : base_vel_idx + 3]
            ),
            base_angular_vel=self.backend.array(
                state.data.qvel[base_vel_idx + 3 : base_vel_idx + 6]
            ),
        )
