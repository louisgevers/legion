# SPDX-FileCopyrightText: Copyright (c) 2026 Louis Gevers
# SPDX-License-Identifier: BSD-3-Clause
#
# See LICENSE file for full license information.
#
import numpy as np
from tqdm import trange
from collections import defaultdict

from legion.backend import RNGKey
from legion.policy import Policy
from legion.runtime import Runtime


def _make_episode_fn(runtime: Runtime, policy: Policy, max_duration: float):
    """Return a pure function ``run_episode(rng)`` whose inner rollout loop is a single
    ``backend.scan`` call (JIT-friendly loop)."""
    backend = runtime.backend
    policy_dt = runtime._policy_dt
    metrics_names = runtime.task.metric_names
    max_steps = int(max_duration // policy_dt)

    def _step(carry, _):
        # Step and accumulators
        state, obs, rng, reward_acc, metrics_acc, done = carry

        # Sample action
        rng, action_key = backend.rng_split(rng)
        u = policy.action(obs, action_key)

        # Step the environment
        transition = runtime.step(state, u)

        # Mask accumulation once episode has ended
        active = ~done
        zero = backend.zeros(())

        # Extract metrics
        data = runtime.physics.get_sensor_data(transition.state.physics)
        new_reward_acc = reward_acc + backend.where(active, transition.reward, zero)
        new_metrics_acc = {
            k: metrics_acc[k] + backend.where(active, transition.metrics[k], zero)
            for k in metrics_names
        }

        # Update done only after (so we still use the final transition)
        new_done = done | transition.done

        # Update new carry
        new_carry = (
            transition.state,
            transition.obs,
            rng,
            new_reward_acc,
            new_metrics_acc,
            new_done,
        )

        # Yield active so the caller can sum it to get episode length
        return new_carry, active

    def run_episode(rng: RNGKey):
        rng, reset_key = backend.rng_split(rng)
        state = runtime.reset(reset_key)
        obs = runtime.observe(state)

        init_carry = (
            state,
            obs,
            rng,
            backend.zeros(()),  # reward_acc
            {k: backend.zeros(()) for k in metrics_names},  # metrics_acc
            backend.array(False),  # done
        )

        final_carry, active_mask = backend.scan(
            _step, init_carry, None, length=max_steps
        )
        _, _, _, reward, metrics, _ = final_carry

        # Active mask is (max_steps,) booleans
        episode_duration = backend.sum(active_mask) * policy_dt

        return reward, metrics, episode_duration

    return run_episode


class EvalRunner:
    def __init__(self, n_episodes: int, max_duration: float):
        self.n_episodes = n_episodes
        self.max_duration = max_duration

    def evaluate(
        self, runtime: Runtime, policy: Policy, rng: RNGKey
    ) -> dict[str, np.ndarray]:
        backend = runtime.backend

        # Build and JIT the episode rollout function
        run_episode = _make_episode_fn(runtime, policy, self.max_duration)
        jit_episode = backend.jit(run_episode)

        # NOTE: We don't do warm-up because that would require running an entirely wasted episode
        print(
            f"Evaluation {self.n_episodes} episodes of max duration {self.max_duration}s each."
        )
        print("Note that the first episode might take a while due to compilation.")

        # Episode loop at python level
        results = defaultdict(list)
        for ep_idx in trange(self.n_episodes, unit="episode"):
            # Run an episode
            rng, ep_key = backend.rng_split(rng)
            reward, metrics, duration = jit_episode(ep_key)

            # Store metrics
            results["episode"].append(ep_idx + 1)
            results["sum_reward"].append(reward)
            results["episode_duration"].append(duration)
            for k, v in metrics.items():
                results[k].append(v)

        # Convert to numpy arrays for processing
        results = {k: np.array(v) for k, v in results.items()}
        return results
