import mujoco
from numpy.typing import ArrayLike

from legion.backend import get_backend, Backend
from legion.embodiment import Embodiment

from .base import PhysicsState, SensorData


class MujocoPhysics:
    name = "mujoco"

    def __init__(self, embodiment: Embodiment, mjcf: str):
        # Lazy load backend
        self.backend = get_backend("numpy")

        # Load MJCF
        self._mj_model = mujoco.MjModel.from_xml_path(mjcf)

        # Precompute joint indices
        self._base_qpos_idx, self._base_qvel_idx = mj_base_indices(
            self._mj_model, self.backend
        )
        self._joint_qpos_idx, self._joint_qvel_idx = mj_joint_indices(
            self._mj_model, embodiment, self.backend
        )
        self._actuator_idx = mj_actuator_indices(
            self._mj_model, embodiment, self.backend
        )

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
        state.data.ctrl[self._actuator_idx] = tau
        return state

    def get_sensor_data(self, state: PhysicsState) -> SensorData:
        return SensorData(
            t=self.backend.array(state.data.time),
            q=self.backend.array(state.data.qpos[self._joint_qpos_idx]),
            dq=self.backend.array(state.data.qvel[self._joint_qvel_idx]),
            tau=self.backend.array(state.data.ctrl[self._actuator_idx]),
            base_xyz=self.backend.array(state.data.qpos[self._base_qpos_idx][:3]),
            base_quat=self.backend.array(state.data.qpos[self._base_qpos_idx][3:]),
            base_linear_vel=self.backend.array(
                state.data.qvel[self._base_qvel_idx][:3]
            ),
            base_angular_vel=self.backend.array(
                state.data.qvel[self._base_qvel_idx][3:]
            ),
        )


def mj_base_indices(
    mj_model: mujoco._MjBindModel, backend: Backend
) -> tuple[ArrayLike, ArrayLike]:
    """Extract floating base qpos and qvel indices given the model"""
    qpos_adr = mj_model.joint("base").qposadr[0]
    base_qpos_idx = backend.array(range(qpos_adr, qpos_adr + 7))  # xyz + quaternion

    qvel_adr = mj_model.joint("base").dofadr[0]
    base_qvel_idx = backend.array(
        range(qvel_adr, qvel_adr + 6)
    )  # linear + angular velocity

    return base_qpos_idx, base_qvel_idx


def mj_joint_indices(
    mj_model: mujoco._MjBindModel, embodiment: Embodiment, backend: Backend
) -> tuple[ArrayLike, ArrayLike]:
    """Extract joint qpos and qvel indices given the model and embodiment"""

    joint_qpos_idx = []
    joint_qvel_idx = []
    for name in embodiment.joint_names:
        try:
            j = mj_model.joint(name)
        except Exception:
            raise ValueError(f"Embodiment joint '{name}' not found in MJCF")

        joint_qpos_idx.append(j.qposadr[0])
        joint_qvel_idx.append(j.dofadr[0])

    return backend.array(joint_qpos_idx), backend.array(joint_qvel_idx)


def mj_actuator_indices(
    mj_model: mujoco._MjBindModel, embodiment: Embodiment, backend: Backend
) -> ArrayLike:
    actuator_idx = []
    """Extract actuator indices given the model an embodiment"""
    for name in embodiment.actuator_names:
        try:
            a = mj_model.actuator(name)
        except Exception:
            raise ValueError(f"Embodiment actuator '{name}' not found in MJCF")

        actuator_idx.append(a.id)

    return backend.array(actuator_idx)
