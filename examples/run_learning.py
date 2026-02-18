from pathlib import Path
from argparse import ArgumentParser, Namespace
from datetime import datetime

from legion.runtime import make_runtime
from legion.policy import PolicyLoader
from legion.runner.learning import make_learning_runner
from legion.utils import ConfigUtil
from legion.runner.viewer import ViewerRunner
from legion.logger.tensorboard import TensorboardLogger


EXAMPLES_DIR = (Path(__file__)).parent.resolve(True)
CONFIGS_DIR = (Path(__file__).parent.parent / "configs").resolve(True)


def run_learning(cfg_runtime: str, cfg_learning: str, tag: str | None):
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

    # Create logger (with timestamped log directory)
    log_name = tag if tag is not None else datetime.now().strftime("%y-%m-%d_%H-%M-%S")
    log_dir = EXAMPLES_DIR / "runs" / log_name
    logger = TensorboardLogger(log_dir)

    # Start learning
    policy = runner.learn(runtime, algo_cfg=cfg_learning["algorithm"], logger=logger)

    # Close the logger after learning
    logger.close()

    # Save the policy to file
    policy_name = tag if tag is not None else "learned"
    policy_file = EXAMPLES_DIR / f"{tag}.policy"
    policy.save(policy_file)

    # Reload the policy from file
    policy = PolicyLoader.load(policy_file)

    # Wait for user intervention before visualizing the policy
    input("Press anything to continue!")

    # Create an evaluation runtime (can optionally modify it)
    cfg_eval = cfg_runtime.copy()
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
    parser.add_argument(
        "-t",
        "--tag",
        type=str,
        default=None,
        required=False,
        help="Optional tag to change the log directory name",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_cli_args()
    run_learning(args.runtime, args.learning, args.tag)
