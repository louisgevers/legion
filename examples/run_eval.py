# SPDX-FileCopyrightText: Copyright (c) 2026 Louis Gevers
# SPDX-License-Identifier: BSD-3-Clause
#
# See LICENSE file for full license information.
#
from pathlib import Path
from argparse import ArgumentParser, Namespace

from legion.runtime import make_runtime
from legion.utils import ConfigUtil
from legion.policy import PolicyLoader
from legion.policy.random import RandomUniformPolicy
from legion.runner.eval import EvalRunner


CONFIGS_DIR = (Path(__file__).parent.parent / "configs").resolve(True)


def run_eval(cfg_name: str, policy: Path, n_episodes: int, max_duration: float):
    # Util to load runtime configs
    runtime_configs = ConfigUtil(CONFIGS_DIR / "runtimes")

    # Create runtime from config
    cfg = runtime_configs.load(cfg_name)
    runtime = make_runtime(cfg)

    if policy.name == "random":
        # Create a random policy
        policy = RandomUniformPolicy(
            runtime.backend, runtime.actuator.n_u, action_scaling=1.0
        )
    else:
        # Load the policy
        policy = PolicyLoader.load(policy)

    # Create RNG key
    rng = runtime.backend.rng_seed(42)

    # Run with an evaluation runner, repeated over n episodes
    runner = EvalRunner(n_episodes, max_duration)
    result = runner.evaluate(runtime, policy, rng)

    # Result is a dictionary of arrays with an entry per episode, any postprocessing can happen here
    print(result)


def parse_cli_args() -> Namespace:
    parser = ArgumentParser("Evaluate a policy on a given runtime config.")
    parser.add_argument(
        "config",
        type=str,
        help="Name of the runtime config to evaluate the policy on (in the configs/runtimes directory).",
    )
    parser.add_argument(
        "policy",
        type=Path,
        help="Path to a .policy pickle to load the policy from (or 'random' for a random policy)",
    )
    parser.add_argument(
        "-n",
        "--n-episodes",
        type=int,
        default=10,
        help="Number of episodes to evaluate.",
    )
    parser.add_argument(
        "-d",
        "--max-duration",
        type=float,
        default=10.0,
        help="Max duration of an episode.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_cli_args()
    run_eval(args.config, args.policy, args.n_episodes, args.max_duration)
