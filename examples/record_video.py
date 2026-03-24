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
from legion.runner.render import VideoRunner


CONFIGS_DIR = (Path(__file__).parent.parent / "configs").resolve(True)


def record_video(
    cfg_name: str,
    policy: Path,
    duration: float,
    fps: int,
    frame_width: float,
    frame_height: float,
):
    # Util to load runtime configs
    runtime_configs = ConfigUtil(CONFIGS_DIR / "runtimes")

    # Create runtime from config
    cfg = runtime_configs.load(cfg_name)
    runtime = make_runtime(cfg)

    # Load the policy
    policy = PolicyLoader.load(policy)

    # Create RNG key
    rng = runtime.backend.rng_seed(42)

    # Create a runner to record a video
    runner = VideoRunner(
        duration=duration,
        fps=fps,
        frame_width=frame_width,
        frame_height=frame_height,
        camera="track",
    )

    # Record to local folder
    video_file = Path(__file__).parent / "recording.mp4"
    runner.record(video_file, runtime, policy, rng)


def parse_cli_args() -> Namespace:
    parser = ArgumentParser("Record a video of chosen policy and runtime config.")
    parser.add_argument(
        "config",
        type=str,
        help="Name of the runtime config to record the policy in (in the configs/runtimes directory).",
    )
    parser.add_argument(
        "policy", type=Path, help="Path to a .policy pickle to load the policy from"
    )
    parser.add_argument(
        "duration", type=float, help="Duration (in seconds) of the video recording"
    )
    parser.add_argument(
        "--fps", type=float, help="FPS of the video recording", default=24
    )
    parser.add_argument(
        "-fw", "--frame-width", type=int, help="Frame width", default=680
    )
    parser.add_argument(
        "-fh", "--frame-height", type=int, help="Frame height", default=480
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_cli_args()
    record_video(
        args.config,
        args.policy,
        args.duration,
        args.fps,
        args.frame_width,
        args.frame_height,
    )
