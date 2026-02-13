import mujoco
import mujoco.viewer

from legion.physics.mujoco import MujocoPhysics, PhysicsState


class MujocoViewer:
    def __init__(self, physics: MujocoPhysics):
        self._mj_model = physics._mj_model
        self._mj_data = mujoco.MjData(self._mj_model)  # Internal data handle
        self._mj_viewer = mujoco.viewer.launch_passive(self._mj_model, self._mj_data)

    def render(self, state: PhysicsState):
        # Copy into internal data
        mujoco.mj_copyState(
            self._mj_model,
            state.data,
            self._mj_data,
            mujoco.mjtState.mjSTATE_INTEGRATION,  # Full state copy
        )
        mujoco.mj_forward(self._mj_model, self._mj_data)
        self._mj_viewer.sync()

    def is_running(self) -> bool:
        return self._mj_viewer.is_running()

    def close(self):
        self._mj_viewer.close()
