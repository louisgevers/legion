from pathlib import Path
from argparse import ArgumentParser, Namespace

from legion.runtime import make_runtime
from legion.runner.learning import make_learning_runner
from legion.utils import ConfigUtil
from legion.runner.viewer import ViewerRunner
from legion.policy.random import RandomUniformPolicy


CONFIGS_DIR = (Path(__file__).parent.parent / "configs").resolve(True)


def run_learning(cfg_runtime: str, cfg_learning: str):
    # Utils to load configs
    runtime_configs = ConfigUtil(CONFIGS_DIR / "runtimes")
    learning_configs = ConfigUtil(CONFIGS_DIR / "learning")

    # Load configs
    cfg_runtime = runtime_configs.load(cfg_runtime)
    cfg_learning = learning_configs.load(cfg_learning)

    # Create runtime
    runtime = make_runtime(cfg_runtime)

    # Create runner
    runner = make_learning_runner(cfg_learning)

    # Start learning
    policy = runner.learn(runtime, algo_cfg=cfg_learning["algorithm"])

    # Wait for user intervention before visualizing the policy
    input("Press anything to continue!")

    # Create an evaluation runtime with fixed velocity
    cfg_eval = cfg_runtime.copy()
    cfg_eval["signals"][1] = {
        "type": "fixed_value",
        "name": "velocity_command",
        "values": [0.5, 0.0],
    }
    runtime = make_runtime(cfg_eval)

    # Run with a viewer
    rng = runtime.backend.rng_seed(42)
    runner = ViewerRunner()
    runner.run(runtime, policy, rng)


def parse_cli_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "-r",
        "--runtime",
        type=str,
        default="go2_mjx",
        help="Name of the runtime config to perform learning on.",
    )
    parser.add_argument(
        "-l",
        "--learning",
        type=str,
        default="brax",
        help="Name of the learning config",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_cli_args()
    run_learning(args.runtime, args.learning)
