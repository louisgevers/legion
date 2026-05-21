# SPDX-FileCopyrightText: Copyright (c) 2026 Louis Gevers
# SPDX-License-Identifier: BSD-3-Clause
#
# See LICENSE file for full license information.
#
import copy
import mujoco
from typing import NamedTuple
from numpy.typing import ArrayLike

from legion.backend import get_backend, Backend
from legion.registry import PHYSICS, register
from legion.embodiment import Embodiment
from legion.utils.assets import get_asset_path

from .base import SensorData


FLOOR_GEOM_ID = 0


class MujocoState(NamedTuple):
    data: mujoco._MjBindData
    model: mujoco._MjBindModel


@register(PHYSICS, "mujoco")
class MujocoPhysics:
    name = "mujoco"

    def __init__(self, embodiment: Embodiment, dt: float, mjcf: str):
        # Lazy load backend
        self.backend = get_backend("numpy")

        # Store embodiment
        self.embodiment = embodiment

        # Load MJCF
        self._mj_model = mujoco.MjModel.from_xml_path(get_asset_path(mjcf))

        # Set timestep
        self._mj_model.opt.timestep = dt

        # Precompute joint indices
        self._base_qpos_idx, self._base_qvel_idx = mj_base_indices(
            self._mj_model, self.backend, embodiment.base_name
        )
        self._joint_qpos_idx, self._joint_qvel_idx = mj_joint_indices(
            self._mj_model, embodiment, self.backend
        )
        self._actuator_idx = mj_actuator_indices(
            self._mj_model, embodiment, self.backend
        )

        # Load foot geom ids for contact detection
        self._foot_geom_ids = mj_foot_geom_ids(self._mj_model, embodiment, self.backend)

        # Load base body index
        self._base_body_idx = self._mj_model.body(embodiment.base_name).id

    @property
    def dt(self) -> float:
        return self._mj_model.opt.timestep

    def reset(
        self,
        q: ArrayLike,
        base_xyz: ArrayLike,
    ) -> MujocoState:
        # Copy base model
        model = copy.deepcopy(self._mj_model)

        # Create blank data
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)

        # Apply initial q positions
        data.qpos[self._joint_qpos_idx] = q * self.embodiment.q_directions

        # Apply initial base positions
        data.qpos[self._base_qpos_idx[:3]] = base_xyz

        return MujocoState(data=data, model=model)

    def step(self, state: MujocoState) -> MujocoState:
        # MuJoCo state is mutable
        mujoco.mj_step(state.model, state.data)
        return state

    def apply_torques(self, state: MujocoState, tau: ArrayLike) -> MujocoState:
        # MuJoCo state is mutable
        state.data.ctrl[self._actuator_idx] = tau * self.embodiment.q_directions
        return state

    def get_sensor_data(self, state: MujocoState) -> SensorData:
        return SensorData(
            t=self.backend.array(state.data.time),
            q=self.backend.array(state.data.qpos[self._joint_qpos_idx])
            * self.embodiment.q_directions,
            dq=self.backend.array(state.data.qvel[self._joint_qvel_idx])
            * self.embodiment.q_directions,
            ddq=self.backend.array(state.data.qacc[self._joint_qvel_idx])
            * self.embodiment.q_directions,
            tau=self.backend.array(state.data.actuator_force[self._actuator_idx])
            * self.embodiment.q_directions,
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
            foot_normal_forces=self._compute_foot_normal_forces(state),
        )

    def set_ground_friction(self, state: MujocoState, friction: float) -> MujocoState:
        # MuJoCo model is directly mutable
        state.model.geom(FLOOR_GEOM_ID).friction[0] = friction
        return state

    def add_base_mass(self, state: MujocoState, mass: float) -> MujocoState:
        # MuJoCo model is directly mutable
        state.model.body(self.embodiment.base_name).mass += mass
        return state

    def scale_masses(self, state: MujocoState, scales: ArrayLike) -> MujocoState:
        # MuJoCo model is directly mutable
        for i in range(1, state.model.nbody):  # Skip the worldbody
            state.model.body(i).mass *= scales[i - 1]
        return state

    def offset_joints(self, state: MujocoState, offsets: ArrayLike) -> MujocoState:
        # MuJoCo data is directly mutable
        state.data.qpos[self._joint_qpos_idx] += offsets * self.embodiment.q_directions
        mujoco.mj_forward(state.model, state.data)
        return state

    def apply_base_perturbation(
        self, state: MujocoState, force: ArrayLike
    ) -> MujocoState:
        # MuJoCo data is directly mutable
        state.data.xfrc_applied[self._base_body_idx, :3] = force
        return state

    def _compute_foot_contacts(self, state: MujocoState) -> ArrayLike:
        contact_bools = self.backend.zeros(self.embodiment.n_feet)
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

    def _compute_foot_normal_forces(self, state: MujocoState) -> ArrayLike:
        forces = self.backend.zeros(self.embodiment.n_feet)
        result = self.backend.zeros(6)  # To store contact forces from mujoco
        for i_con in range(state.data.ncon):
            contact = state.data.contact[i_con]
            for foot_i, foot_id in enumerate(self._foot_geom_ids):
                # If contact with the floor
                if (contact.geom1 == foot_id and contact.geom2 == FLOOR_GEOM_ID) or (
                    contact.geom2 == foot_id and contact.geom1 == FLOOR_GEOM_ID
                ):
                    mujoco.mj_contactForce(state.model, state.data, i_con, result)
                    forces[foot_i] += result[
                        0
                    ]  # result[0] is normal force in contact frame
        return forces


def mj_base_indices(
    mj_model: mujoco._MjBindModel, backend: Backend, base_name: str
) -> tuple[ArrayLike, ArrayLike]:
    """Extract floating base qpos and qvel indices given the model"""
    qpos_adr = mj_model.joint(base_name).qposadr[0]
    base_qpos_idx = backend.array(range(qpos_adr, qpos_adr + 7))  # xyz + quaternion

    qvel_adr = mj_model.joint(base_name).dofadr[0]
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
