import mujoco
import mujoco.viewer

from legion.physics.mujoco import MujocoPhysics, PhysicsState


class MujocoViewer:
    def __init__(self, physics: MujocoPhysics):
        self._mj_model = physics._mj_model
        self._mj_viewer: mujoco.viewer.Handle | None = None
        self._first_run = True

    def render(self, state: PhysicsState):
        # Mujoco state data is just mutated
        if self._mj_viewer is None:
            self._mj_viewer = mujoco.viewer.launch_passive(self._mj_model, state.data)
            self._first_run = False
        self._mj_viewer.sync()

    def is_running(self) -> bool:
        return self._first_run or self._mj_viewer.is_running()

    def close(self):
        self._mj_viewer.close()
