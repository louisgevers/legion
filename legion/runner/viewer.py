import time
import datetime
import tqdm

from legion.backend import RNGKey
from legion.policy import Policy
from legion.runtime import Runtime
from legion.viewer import make_viewer


class ViewerRunner:
    def run(
        self,
        runtime: Runtime,
        policy: Policy,
        rng: RNGKey,
    ):
        # Create viewer
        viewer = make_viewer(runtime.physics)

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

        # Progress bar for FPS tracking
        pbar = tqdm.tqdm(unit="steps")

        # Simulation loop
        start_time = time.perf_counter()
        ep_reward = 0
        ep_length = 0
        while viewer.is_running():
            # Step
            rng, key = runtime.backend.rng_split(rng)
            u = policy.action(obs, key)
            state, obs, reward, done, metrics = jit_step(state, u)

            # Track stats
            ep_reward += reward
            ep_length += 1

            # Reset if done (FIXME: could be sped up for GPU cases)
            if runtime.backend.any(done):
                print(f"Episode reward: {ep_reward:.2f} (length={ep_length})")
                ep_reward = 0
                ep_length = 0
                start_time = time.perf_counter()

                rng, key = runtime.backend.rng_split(rng)
                state = jit_reset(key)
                obs = jit_observe(state)

            # Sync viewer
            viewer.render(state.physics)

            # Update progress bar
            pbar.update(1)

            # Schedule next frame time
            sim_time = runtime.physics.get_sensor_data(state.physics).t
            while time.perf_counter() - start_time < sim_time:
                continue

        # Close the viewer
        viewer.close()
