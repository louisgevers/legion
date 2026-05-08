# SPDX-FileCopyrightText: Copyright (c) 2026 Louis Gevers
# SPDX-License-Identifier: BSD-3-Clause
#
# See LICENSE file for full license information.
#
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
    METRICS,
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
    # Each of these are optional (e.g., just want the physics engine)
    # - depends on backend, embodiment, actuator
    signals = []
    if "signals" in cfg:
        signals = _build_task_terms(
            cfg["signals"], SIGNALS, backend, embodiment, actuator
        )

    obs = []
    if "observations" in cfg:
        obs = _build_task_terms(
            cfg["observations"], OBSERVATIONS, backend, embodiment, actuator
        )

    rew = []
    if "rewards" in cfg:
        rew = _build_task_terms(cfg["rewards"], REWARDS, backend, embodiment, actuator)

    ter = []
    if "terminations" in cfg:
        ter = _build_task_terms(
            cfg["terminations"], TERMINATIONS, backend, embodiment, actuator
        )

    metrics = []
    if "metrics" in cfg:
        metric_terms = [term for term in cfg["metrics"] if term["type"] in METRICS]
        reward_terms = [term for term in cfg["metrics"] if term["type"] in REWARDS]
        # Reward terms with weight 1 as metrics
        metrics = _build_task_terms(
            reward_terms, REWARDS, backend, embodiment, actuator, weight=1.0
        ) + _build_task_terms(metric_terms, METRICS, backend, embodiment, actuator)

        # Check if there are any unknown metrics
        for term in cfg["metrics"]:
            term_type = term["type"]
            if term_type not in METRICS and term_type not in REWARDS:
                raise ValueError(f"Unknown metric or reward term: '{term_type}'")

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
    runtime_kwargs = cfg.get("runtime", {})
    return Runtime(
        embodiment, physics, actuator, task, domain_randomization, **runtime_kwargs
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
