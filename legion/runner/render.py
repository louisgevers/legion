# SPDX-FileCopyrightText: Copyright (c) 2026 Louis Gevers
# SPDX-License-Identifier: BSD-3-Clause
#
# See LICENSE file for full license information.
#
import cv2
import time
import datetime
import tqdm

from legion.backend import RNGKey
from legion.policy import Policy
from legion.runtime import Runtime
from legion.render import make_renderer


class VideoRunner:
    def __init__(
        self,
        duration: float,
        fps: float,
        frame_width: int,
        frame_height: int,
        **render_kwargs,
    ):
        self.duration = duration
        self.fps = fps
        self.frame_width = frame_width
        self.frame_height = frame_height
        self._render_kwargs = render_kwargs

    def record(self, output_file: str, runtime: Runtime, policy: Policy, rng: RNGKey):
        # JIT runtime functions for viewer
        jit_start_time = time.perf_counter()
        print("Starting JIT of runtime functions...")
        jit_reset = runtime.backend.jit(lambda rng: runtime.reset(rng))
        jit_step = runtime.backend.jit(
            lambda state, action: runtime.step(state, action)
        )
        jit_observe = runtime.backend.jit(lambda state: runtime.observe(state))
        # Warm up
        warmup_rng = runtime.backend.rng_seed(42)
        warmup_u = runtime.backend.zeros(runtime.actuator.n_u)
        warmup_state = jit_reset(warmup_rng)
        _ = jit_observe(warmup_state)
        _ = jit_step(warmup_state, warmup_u)
        print(
            f"JIT took {datetime.timedelta(seconds=time.perf_counter() - jit_start_time)}"
        )

        # Initialization
        state = jit_reset(rng)
        obs = jit_observe(state)

        # Create renderer
        renderer = make_renderer(
            runtime.physics,
            frame_width=self.frame_width,
            frame_height=self.frame_height,
            **self._render_kwargs,
        )

        # Define codec and create video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            output_file, fourcc, self.fps, (self.frame_width, self.frame_height)
        )

        # Progress bar for recording tracking
        steps = int(self.duration * self.fps)
        render_timestep = 1.0 / self.fps
        next_render_time = 0.0
        for _ in tqdm.trange(steps, unit="frames"):
            # Step physics (FIXME: could be scanned for GPU cases)
            while runtime.physics.get_sensor_data(state.physics).t <= next_render_time:
                # Step
                rng, key = runtime.backend.rng_split(rng)
                u = policy.action(obs, key)
                state, obs, reward, done, metrics = jit_step(state, u)

                # Reset if done (FIXME: could be sped up for GPU cases)
                if runtime.backend.any(done):
                    rng, key = runtime.backend.rng_split(rng)
                    state = jit_reset(key)
                    obs = jit_observe(state)
                    next_render_time = 0.0

            # Render frame
            frame_rgb = renderer.render(state.physics)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
            next_render_time += render_timestep

        writer.release
        renderer.close()
