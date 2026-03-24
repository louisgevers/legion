# SPDX-FileCopyrightText: Copyright (c) 2026 Louis Gevers
# SPDX-License-Identifier: BSD-3-Clause
#
# See LICENSE file for full license information.
#
from legion.backend import Backend, RNGKey
from legion.physics import PhysicsState, SensorData

from .terms import DomainRandomizationTerm

DomainRandomizationState = tuple[dict, ...]


class DomainRandomization:
    def __init__(self, backend: Backend, terms: list[DomainRandomizationTerm]):
        self.backend = backend
        self.terms = tuple(terms)

    def reset(self, rng: RNGKey) -> DomainRandomizationState:
        rngs = self.backend.rng_split(rng, len(self.terms))
        return tuple(t.reset(rng) for t, rng in zip(self.terms, rngs))

    def step(
        self,
        domain_randomization_state: DomainRandomizationState,
        sensor_data: SensorData,
        rng: RNGKey,
    ) -> DomainRandomizationState:
        rngs = self.backend.rng_split(rng, len(self.terms))
        return tuple(
            t.step(state, sensor_data, rng)
            for t, state, rng in zip(self.terms, domain_randomization_state, rngs)
        )

    def apply_reset(
        self,
        domain_randomization_state: DomainRandomizationState,
        physics_state: PhysicsState,
        rng: RNGKey,
    ) -> PhysicsState:
        rngs = self.backend.rng_split(rng, len(self.terms))

        for term, state, rng in zip(self.terms, domain_randomization_state, rngs):
            physics_state = term.apply_reset(state, physics_state, rng)

        return physics_state

    def apply_step(
        self,
        domain_randomization_state: DomainRandomizationState,
        physics_state: PhysicsState,
        sensor_data: SensorData,
        rng: RNGKey,
    ) -> PhysicsState:
        rngs = self.backend.rng_split(rng, len(self.terms))

        for term, state, rng in zip(self.terms, domain_randomization_state, rngs):
            physics_state = term.apply_step(state, physics_state, sensor_data, rng)

        return physics_state
