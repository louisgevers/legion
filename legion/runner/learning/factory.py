from copy import deepcopy
from .base import LearningRunner


def make_learning_runner(cfg: dict) -> LearningRunner:
    # Edit a copy
    cfg = deepcopy(cfg)

    # Extract runner type
    runner_type = cfg["runner"].pop("type")

    if runner_type == "brax":
        from .brax import BraxLearningRunner

        return BraxLearningRunner(**cfg["runner"])

    raise ValueError(f"Unknown learning runner type: {runner_type}")
