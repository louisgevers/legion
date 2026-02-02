import jax
import mujoco
import mujoco.viewer
from mujoco import mjx

from legion.physics.mjx import MJXPhysics, PhysicsState


class MJXViewer:
    def __init__(self, physics: MJXPhysics, env_idx: int = 0):
        # Keep track of which environment to render
        self.env_idx = env_idx
        # Get Mujoco model reference
        self._mj_model = physics._mj_model
        # Create and keep a mujoco data reference
        self._mj_data = mujoco.MjData(self._mj_model)
        # Create the viewer with Mujoco references
        self._mj_viewer = mujoco.viewer.launch_passive(self._mj_model, self._mj_data)

    def render(self, state: PhysicsState):
        # Only use the state at the right index
        state_to_use = self._select_env(state)

        # Update data from device
        mjx.get_data_into(self._mj_data, self._mj_model, state_to_use.data)

        # Sync viewer
        self._mj_viewer.sync()

    def is_running(self) -> bool:
        return self._mj_viewer.is_running()

    def close(self):
        self._mj_viewer.close()

    def _select_env(self, state: PhysicsState):
        # Check if state is batched, if so pick the desired indexed one
        batched = len(state.data.qpos.shape) > 1
        if not batched:
            return state
        else:
            # Batched
            return jax.tree.map(lambda x: x[self.env_idx], state)
