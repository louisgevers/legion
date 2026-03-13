import mujoco
from mujoco import mjx
from typing import NamedTuple
from numpy.typing import ArrayLike

from legion.backend import get_backend
from legion.registry import PHYSICS, register
from legion.embodiment import Embodiment
from legion.utils.assets import get_asset_path

from .base import SensorData
from .mujoco import (
    mj_base_indices,
    mj_joint_indices,
    mj_actuator_indices,
    mj_foot_geom_ids,
)

FLOOR_GEOM_ID = 0


class MJXState(NamedTuple):
    data: mjx.Data
    model: mjx.Model


@register(PHYSICS, "mjx")
class MJXPhysics:
    name = "mjx"

    def __init__(self, embodiment: Embodiment, dt: float, mjcf: str):
        # Lazy load backend
        self.backend = get_backend("jax")

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

        # Load base body index
        self._base_body_idx = self._mj_model.body("base").id

        # Store q_directions
        self.q_dir = self.backend.array(embodiment.q_directions)

    @property
    def dt(self) -> float:
        return self._mj_model.opt.timestep

    def reset(self, q: ArrayLike, base_xyz: ArrayLike) -> MJXState:
        # Create MJX model
        model = mjx.put_model(self._mj_model)

        # Create blank data
        data = mjx.make_data(model)

        # Apply initial q positions
        qpos = data.qpos.at[self._joint_qpos_idx].set(q * self.q_dir)

        # Apply initial base positions
        qpos = qpos.at[self._base_qpos_idx[:3]].set(base_xyz)

        # Replace qpos in data
        data = data.replace(qpos=qpos)

        # Forward dynamics
        data = mjx.forward(model, data)

        return MJXState(data=data, model=model)

    def step(self, state: MJXState) -> MJXState:
        data = mjx.step(state.model, state.data)
        return state._replace(data=data)

    def apply_torques(self, state: MJXState, tau: ArrayLike) -> MJXState:
        return state._replace(
            data=state.data.replace(ctrl=tau[self._actuator_idx] * self.q_dir)
        )

    def get_sensor_data(self, state: MJXState) -> SensorData:
        return SensorData(
            t=self.backend.array(state.data.time),
            q=self.backend.array(state.data.qpos[self._joint_qpos_idx]) * self.q_dir,
            dq=self.backend.array(state.data.qvel[self._joint_qvel_idx]) * self.q_dir,
            ddq=self.backend.array(state.data.qacc[self._joint_qvel_idx]) * self.q_dir,
            tau=self.backend.array(state.data.actuator_force[self._actuator_idx])
            * self.q_dir,
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
            n_contacts=self.backend.sum(
                state.data.contact.dist < 0
            ),  # Only when distance is negative is there a contact
            foot_contacts=self._compute_foot_contacts(state),
        )

    def set_ground_friction(self, state: MJXState, friction: float) -> MJXState:
        geom_friction = state.model.geom_friction.at[FLOOR_GEOM_ID, 0].set(friction)
        return state._replace(model=state.model.replace(geom_friction=geom_friction))

    def add_base_mass(self, state: MJXState, mass: float) -> MJXState:
        body_mass = state.model.body_mass.at[self._base_body_idx].set(
            state.model.body_mass[self._base_body_idx] + mass
        )
        return state._replace(model=state.model.replace(body_mass=body_mass))

    def scale_masses(self, state: MJXState, scales: ArrayLike) -> MJXState:
        body_mass = state.model.body_mass.at[1:].set(
            state.model.body_mass[1:] * scales
        )  # Skip the worldbody
        return state._replace(model=state.model.replace(body_mass=body_mass))

    def offset_joints(self, state: MJXState, offsets: ArrayLike) -> MJXState:
        qpos = state.data.qpos.at[self._joint_qpos_idx].set(
            state.data.qpos[self._joint_qpos_idx] + offsets * self.q_dir
        )
        data = state.data.replace(qpos=qpos)
        data = mjx.forward(state.model, data)
        return state._replace(data=data)

    def apply_base_perturbation(self, state: MJXState, force: ArrayLike) -> MJXState:
        xfrc_applied = state.data.xfrc_applied.at[self._base_body_idx, :3].set(force)
        return state._replace(data=state.data.replace(xfrc_applied=xfrc_applied))

    def _compute_foot_contacts(self, state: MJXState) -> ArrayLike:
        # Only when distance is negative is there a contact
        active_contacts = state.data.contact.dist < 0  # (233,)

        # Get mask contacts that involve foot geoms
        foot_contact_geoms = (
            state.data.contact.geom[:, None, 0] == self._foot_geom_ids[None, :]
        ) | (
            state.data.contact.geom[:, None, 1] == self._foot_geom_ids[None, :]
        )  # (233, n_feet)

        # Get mask contacts that involve floor
        floor_contact_geoms = (state.data.contact.geom[:, 0] == FLOOR_GEOM_ID) | (
            state.data.contact.geom[:, 1] == FLOOR_GEOM_ID
        )  # (233,)

        # Collapse active contacts
        foot_in_contact = self.backend.any(
            active_contacts[:, None]
            & foot_contact_geoms
            & floor_contact_geoms[:, None],
            axis=0,
        )  # (n_feet,)

        return foot_in_contact
