import mujoco
from numpy.typing import ArrayLike

from legion.backend import get_backend, Backend
from legion.registry import PHYSICS, register
from legion.embodiment import Embodiment
from legion.utils.assets import get_asset_path

from .base import PhysicsState, SensorData


@register(PHYSICS, "mujoco")
class MujocoPhysics:
    name = "mujoco"

    def __init__(self, embodiment: Embodiment, dt: float, mjcf: str):
        # Lazy load backend
        self.backend = get_backend("numpy")

        # Load MJCF
        self._mj_model = mujoco.MjModel.from_xml_path(get_asset_path(mjcf))

        # Set timestep
        self._mj_model.opt.timestep = dt

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

        # Load foot geom ids for contact detection
        self._foot_geom_ids = mj_foot_geom_ids(self._mj_model, embodiment, self.backend)

    @property
    def dt(self) -> float:
        return self._mj_model.opt.timestep

    def reset(
        self,
        q: ArrayLike,
        base_xyz: ArrayLike,
    ) -> PhysicsState:
        # Create blank data
        data = mujoco.MjData(self._mj_model)
        mujoco.mj_resetData(self._mj_model, data)

        # Apply initial q positions
        data.qpos[self._joint_qpos_idx] = q

        # Apply initial base positions
        data.qpos[self._base_qpos_idx[:3]] = base_xyz

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
            ddq=self.backend.array(state.data.qacc[self._joint_qvel_idx]),
            tau=self.backend.array(state.data.actuator_force[self._actuator_idx]),
            base_xyz=self.backend.array(state.data.qpos[self._base_qpos_idx][:3]),
            base_quat=self.backend.roll(
                state.data.qpos[self._base_qpos_idx][3:], -1
            ),  # Mujoco uses wxyz
            base_linear_vel=self.backend.array(
                state.data.qvel[self._base_qvel_idx][:3]
            ),
            base_angular_vel=self.backend.array(
                state.data.qvel[self._base_qvel_idx][3:]
            ),
            n_contacts=state.data.ncon,
            foot_contacts=self._compute_foot_contacts(state),
        )

    def _compute_foot_contacts(self, state: PhysicsState) -> ArrayLike:
        FLOOR_GEOM_ID = 0
        contact_bools = self.backend.zeros(4)
        for i_con in range(state.data.ncon):
            contact = state.data.contact[i_con]
            for foot_i, foot_id in enumerate(self._foot_geom_ids):
                # If contact with the floor
                if (contact.geom1 == foot_id and contact.geom2 == FLOOR_GEOM_ID) or (
                    contact.geom2 == foot_id and contact.geom1 == FLOOR_GEOM_ID
                ):
                    contact_bools[foot_i] = 1
                    break
        return contact_bools


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


def mj_foot_geom_ids(
    mj_model: mujoco._MjBindModel, embodiment: Embodiment, backend: Backend
) -> ArrayLike:
    """Extract geom foot ids given the model and embodiment"""
    return backend.array([mj_model.geom(foot).id for foot in embodiment.feet_names])
