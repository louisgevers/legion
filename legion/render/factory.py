# SPDX-FileCopyrightText: Copyright (c) 2026 Louis Gevers
# SPDX-License-Identifier: BSD-3-Clause
#
# See LICENSE file for full license information.
#
from legion.physics import PhysicsEngine

from .base import Renderer


def make_renderer(physics: PhysicsEngine, **kwargs) -> Renderer:
    if physics.name == "mujoco":
        from .mujoco import MujocoRenderer

        return MujocoRenderer(physics, **kwargs)
    if physics.name == "mjx":
        from .mjx import MJXRenderer

        return MJXRenderer(physics, **kwargs)
    raise ValueError(f"Renderer not implemented for physics engine: {physics.name}")
