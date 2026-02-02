import mujoco
from numpy.typing import ArrayLike

from legion.backend import get_backend

from .base import PhysicsState, SensorData


class MujocoPhysics:
    name = "mujoco"

    def __init__(self, mjcf: str):
        # Lazy load backend
        self.backend = get_backend("numpy")

        # Load MJCF
        self._mj_model = mujoco.MjModel.from_xml_path(mjcf)

    def reset(self) -> PhysicsState:
        # Create blank data
        data = mujoco.MjData(self._mj_model)
        mujoco.mj_resetData(self._mj_model, data)
        return PhysicsState(data=data)

    def step(self, state: PhysicsState) -> PhysicsState:
        # MuJoCo state is mutable
        mujoco.mj_step(self._mj_model, state.data)
        return state

    def apply_torques(self, state: PhysicsState, tau: ArrayLike) -> PhysicsState:
        # MuJoCo state is mutable
        state.data.ctrl[:] = tau
        return state

    def get_sensor_data(self, state: PhysicsState) -> SensorData:
        return SensorData(
            t=self.backend.array(state.data.time),
            q=self.backend.array(state.data.qpos),
            dq=self.backend.array(state.data.qvel),
            tau=self.backend.array(state.data.ctrl),
            base_xyz=self.backend.array(state.data.joint("base").qpos[0:3]),
            base_quat=self.backend.array(state.data.joint("base").qpos[3:]),
            base_linear_vel=self.backend.array(state.data.joint("base").qvel[0:3]),
            base_angular_vel=self.backend.array(state.data.joint("base").qvel[3:]),
        )
