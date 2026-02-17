from .runtime import Runtime

from legion.registry import (
    plugin,
    build,
    EMBODIMENTS,
    PHYSICS,
    ACTUATORS,
    SIGNALS,
    REWARDS,
    OBSERVATIONS,
    TERMINATIONS,
)
from legion.task import Task


def make_runtime(cfg: dict) -> Runtime:
    # Load the registries
    # FIXME: find a solution to lazyload these, especially physics engines
    plugin.load_all()

    # Build embodiment
    # - no dependencies
    embodiment = build(cfg["embodiment"], EMBODIMENTS)

    # Build physics
    # - depends on embodiment
    # - provides backend
    physics = build(cfg["physics"], PHYSICS, embodiment=embodiment)
    backend = physics.backend

    # Build actuator
    # - depends on backend, embodiment
    actuator = build(
        cfg["actuator"], ACTUATORS, backend=physics.backend, embodiment=embodiment
    )

    # Build signals, observations, rewards, and terminations
    # - depends on backend, embodiment, actuator
    signals = _build_terms(cfg["signals"], SIGNALS, backend, embodiment, actuator)
    obs = _build_terms(cfg["observations"], OBSERVATIONS, backend, embodiment, actuator)
    rew = _build_terms(cfg["rewards"], REWARDS, backend, embodiment, actuator)
    ter = _build_terms(cfg["terminations"], TERMINATIONS, backend, embodiment, actuator)
    metrics = []
    if "metrics" in cfg:
        # Reward terms with weight 1 as metrics
        metrics = _build_terms(
            cfg["metrics"], REWARDS, backend, embodiment, actuator, weight=1.0
        )

    # Build task
    # - depends on backend, signals, observations, rewards, and terminations
    task = Task(backend, signals, obs, rew, ter, metrics)

    # Build runtime
    # - depends on embodiment, physics, actuator, and task
    return Runtime(embodiment, physics, actuator, task, **cfg["runtime"])


def _build_terms(
    terms: list[dict], registry: dict, backend, embodiment, actuator, **extra_kwargs
):
    return [
        build(
            term,
            registry,
            backend=backend,
            embodiment=embodiment,
            actuator=actuator,
            **extra_kwargs,
        )
        for term in terms
    ]
