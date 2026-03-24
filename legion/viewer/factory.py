# SPDX-FileCopyrightText: Copyright (c) 2026 Louis Gevers
# SPDX-License-Identifier: BSD-3-Clause
#
# See LICENSE file for full license information.
#
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
