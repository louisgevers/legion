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
    DOMAIN_RANDOMIZATIONS,
)
from legion.task import Task
from legion.domain_randomization import DomainRandomization


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
    signals = _build_task_terms(cfg["signals"], SIGNALS, backend, embodiment, actuator)
    obs = _build_task_terms(
        cfg["observations"], OBSERVATIONS, backend, embodiment, actuator
    )
    rew = _build_task_terms(cfg["rewards"], REWARDS, backend, embodiment, actuator)
    ter = _build_task_terms(
        cfg["terminations"], TERMINATIONS, backend, embodiment, actuator
    )
    metrics = []
    if "metrics" in cfg:
        # Reward terms with weight 1 as metrics
        metrics = _build_task_terms(
            cfg["metrics"], REWARDS, backend, embodiment, actuator, weight=1.0
        )

    # Build task
    # - depends on backend, signals, observations, rewards, and terminations
    task = Task(backend, signals, obs, rew, ter, metrics)

    # Build domain randomization
    # - depends on backend, physics
    domain_randomizations = []
    if "domain_randomization" in cfg:
        domain_randomizations = _build_domain_randomization_terms(
            cfg["domain_randomization"], backend, embodiment, physics
        )
    domain_randomization = DomainRandomization(backend, domain_randomizations)

    # Build runtime
    # - depends on embodiment, physics, actuator, and task
    return Runtime(
        embodiment, physics, actuator, task, domain_randomization, **cfg["runtime"]
    )


def _build_task_terms(
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


def _build_domain_randomization_terms(terms: list[dict], backend, embodiment, physics):
    return [
        build(
            term,
            DOMAIN_RANDOMIZATIONS,
            backend=backend,
            embodiment=embodiment,
            physics=physics,
        )
        for term in terms
    ]
