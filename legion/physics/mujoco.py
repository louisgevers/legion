import mujoco
from numpy.typing import ArrayLike

from legion.backend import get_backend, Backend

from .base import PhysicsState, SensorData


class MujocoPhysics:
    name = "mujoco"

    def __init__(self, mjcf: str):
        # Lazy load backend
        self.backend = get_backend("numpy")

        # Load MJCF
        self._mj_model = mujoco.MjModel.from_xml_path(mjcf)

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
    mj_model: mujoco._MjBindModel, backend: Backend
) -> tuple[tuple[str, ...], ArrayLike, ArrayLike]:
    """Extract joint names + qpos and qvel indices given the model"""
    joint_names = []
    joint_qpos_idx = []
    joint_qvel_idx = []
    for j in range(mj_model.njnt):
        # Ignore free joints
        if mj_model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            continue

        # Ignore base
        name = mj_model.joint(j).name
        if name == "base":
            continue

        joint_names.append(name)
        joint_qpos_idx.append(mj_model.jnt_qposadr[j])
        joint_qvel_idx.append(mj_model.jnt_dofadr[j])

    return (
        tuple(joint_names),
        backend.array(joint_qpos_idx),
        backend.array(joint_qvel_idx),
    )


def mj_actuator_names(mj_model: mujoco._MjBindModel) -> tuple[str, ...]:
    """Extract actuator names given the model"""
    return tuple([mj_model.actuator(i).name for i in range(mj_model.nu)])
