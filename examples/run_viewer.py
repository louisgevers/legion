from pathlib import Path
from argparse import ArgumentParser, Namespace

from legion.runtime import make_runtime
from legion.utils import ConfigUtil
from legion.policy import PolicyLoader
from legion.runner.viewer import ViewerRunner


CONFIGS_DIR = (Path(__file__).parent.parent / "configs").resolve(True)


def run_viewer(cfg_name: str, policy: Path):
    # Util to load runtime configs
    runtime_configs = ConfigUtil(CONFIGS_DIR / "runtimes")

    # Create runtime from config
    cfg = runtime_configs.load(cfg_name)
    runtime = make_runtime(cfg)

    # Load the policy
    policy = PolicyLoader.load(policy)

    # Create RNG key
    rng = runtime.backend.rng_seed(42)

    # Run with a viewer
    runner = ViewerRunner()
    runner.run(runtime, policy, rng)


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
        "policy", type=Path, help="Path to a .policy pickle to load the policy from"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_cli_args()
    run_viewer(args.config, args.policy)
