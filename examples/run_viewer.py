from pathlib import Path
from argparse import ArgumentParser, Namespace

from legion.runner.viewer import ViewerRunner
from legion.runtime import make_runtime
from legion.utils import ConfigUtil


CONFIGS_DIR = (Path(__file__).parent.parent / "configs").resolve(True)


def run_viewer(cfg_name: str, action_scaling: float):
    # Util to load runtime configs
    runtime_configs = ConfigUtil(CONFIGS_DIR / "runtimes")

    # Create runtime from config
    cfg = runtime_configs.load(cfg_name)
    runtime = make_runtime(cfg)

    # Create a random policy
    random_policy = (
        lambda rng, obs: runtime.backend.rng_uniform(
            rng, runtime.actuator.n_u, minval=-1.0, maxval=1.0
        )
        * action_scaling
    )

    # Create RNG key
    rng = runtime.backend.rng_seed(42)

    # Run with a viewer
    runner = ViewerRunner()
    runner.run(runtime, random_policy, rng)


def parse_cli_args() -> Namespace:
    parser = ArgumentParser(
        "Create an interactive viewer with a random policy for a given runtime config."
    )
    parser.add_argument(
        "config",
        type=str,
        help="Name of the runtime config to create a viewer for (in the configs/runtimes directory).",
    )
    parser.add_argument(
        "-s",
        "--scaling",
        type=float,
        help="Random action scaling",
        default=1.0,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_cli_args()
    run_viewer(args.config, args.scaling)
