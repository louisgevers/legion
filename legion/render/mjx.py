# SPDX-FileCopyrightText: Copyright (c) 2026 Louis Gevers
# SPDX-License-Identifier: BSD-3-Clause
#
# See LICENSE file for full license information.
#
import jax
import numpy as np
import mujoco
from mujoco import mjx

from legion.physics.mjx import MJXPhysics, MJXState


class MJXRenderer:
    def __init__(
        self,
        physics: MJXPhysics,
        frame_width: int = 480,
        frame_height: int = 480,
        camera: int | str = -1,
        max_envs: int = 16,
    ):
        # Get Mujoco model reference
        self._mj_model = physics._mj_model

        # Create renderer
        self._mj_renderer = mujoco.Renderer(
            self._mj_model, height=frame_height, width=frame_width
        )

        # Keep track of options
        self._camera = camera
        self._options = mujoco.MjvOption()
        self._pert = mujoco.MjvPerturb()

        # Mask to only render the bodies of extra batches
        self._batch_catmask = mujoco.mjtCatBit.mjCAT_DYNAMIC
        self._max_envs = max_envs

    def render(self, state: MJXState) -> np.ndarray:
        # Check if batched or not
        batched = len(state.data.qpos.shape) > 1
        if batched:
            # Extract subset of environments
            max_envs = min(self._max_envs, len(state.data.qpos))
            state_subset = jax.tree.map(lambda x: x[:max_envs], state)
            mj_data = mjx.get_data(self._mj_model, state_subset.data)
            # Render the first one completely
            self._render_single(mj_data[0])
            # Render the remaining batch with only the body (up to max envs)
            self._render_batch(mj_data[1:])
        else:
            # Unbatched
            mj_data = mjx.get_data(self._mj_model, state.data)
            self._render_single(mj_data)
        return self._mj_renderer.render()

    def _render_single(self, mj_data):
        mujoco.mj_forward(self._mj_model, mj_data)
        self._mj_renderer.update_scene(mj_data, self._camera, self._options)

    def _render_batch(self, mj_datas):
        for mj_data in mj_datas:
            mujoco.mjv_addGeoms(
                self._mj_model,
                mj_data,
                self._options,
                self._pert,
                self._batch_catmask,
                self._mj_renderer.scene,
            )

    def close(self):
        self._mj_renderer.close()
