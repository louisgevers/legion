import mujoco
from mujoco import mjx
from numpy.typing import ArrayLike

from legion.backend import get_backend
from legion.registry import PHYSICS, register
from legion.embodiment import Embodiment

from .base import PhysicsState, SensorData
from .mujoco import (
    mj_base_indices,
    mj_joint_indices,
    mj_actuator_indices,
    mj_foot_geom_ids,
)


@register(PHYSICS, "mjx")
class MJXPhysics:
    name = "mjx"

    def __init__(self, embodiment: Embodiment, dt: float, mjcf: str):
        # Lazy load backend
        self.backend = get_backend("jax")

        # Load MJCF
        self._mj_model = mujoco.MjModel.from_xml_path(mjcf)

        # Set timestep
        self._mj_model.opt.timestep = dt

        # Create mjx model
        self._mjx_model = mjx.put_model(self._mj_model)

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

    def reset(self) -> PhysicsState:
        # Create blank data
        data = mjx.make_data(self._mj_model)
        data = mjx.forward(self._mjx_model, data)
        return PhysicsState(data=data)

    def step(self, state: PhysicsState) -> PhysicsState:
        data = mjx.step(self._mjx_model, state.data)
        return PhysicsState(data=data)

    def apply_torques(self, state: PhysicsState, tau: ArrayLike) -> PhysicsState:
        return PhysicsState(data=state.data.replace(ctrl=tau[self._actuator_idx]))

    def get_sensor_data(self, state: PhysicsState) -> SensorData:
        return SensorData(
            t=self.backend.array(state.data.time),
            q=self.backend.array(state.data.qpos[self._joint_qpos_idx]),
            dq=self.backend.array(state.data.qvel[self._joint_qvel_idx]),
            tau=self.backend.array(state.data.ctrl[self._actuator_idx]),
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
            foot_contacts=self._compute_foot_contacts(state),
        )

    def _compute_foot_contacts(self, state: PhysicsState) -> ArrayLike:
        # Only when distance is negative is there a contact
        active_contacts = state.data.contact.dist < 0  # (233,)

        # Get mask contacts that involve foot geoms
        foot_contact_geoms = (
            state.data.contact.geom[:, None, 0] == self._foot_geom_ids[None, :]
        ) | (
            state.data.contact.geom[:, None, 1] == self._foot_geom_ids[None, :]
        )  # (233, n_feet)

        # Collapse active contacts
        foot_in_contact = self.backend.any(
            active_contacts[:, None] & foot_contact_geoms, axis=0
        )  # (n_feet,)

        return foot_in_contact
