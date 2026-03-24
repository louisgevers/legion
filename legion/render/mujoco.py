# SPDX-FileCopyrightText: Copyright (c) 2026 Louis Gevers
# SPDX-License-Identifier: BSD-3-Clause
#
# See LICENSE file for full license information.
#
import numpy as np
import mujoco

from legion.physics.mujoco import MujocoPhysics, MujocoState


class MujocoRenderer:
    def __init__(
        self,
        physics: MujocoPhysics,
        frame_width: int = 480,
        frame_height: int = 480,
        camera: int | str = -1,
    ):
        self._mj_renderer = mujoco.Renderer(
            physics._mj_model, height=frame_height, width=frame_width
        )
        self._camera = camera
        self._options = mujoco.MjvOption()

    def render(self, state: MujocoState) -> np.ndarray:
        self._mj_renderer.update_scene(state.data, self._camera, self._options)
        return self._mj_renderer.render()

    def close(self):
        self._mj_renderer.close()
