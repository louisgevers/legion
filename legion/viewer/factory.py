from legion.physics import PhysicsEngine

from .base import Viewer


def make_viewer(physics: PhysicsEngine, **kwargs) -> Viewer:
    if physics.name == "mujoco":
        from .mujoco import MujocoViewer

        return MujocoViewer(physics, **kwargs)
    if physics.name == "mjx":
        from .mjx import MJXViewer

        return MJXViewer(physics, **kwargs)
    raise ValueError(f"Viewer not implemented for physics engine: {physics.name}")
