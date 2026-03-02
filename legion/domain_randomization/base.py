from legion.backend import Backend, RNGKey
from legion.physics import PhysicsState

from .terms import DomainRandomizationTerm


class DomainRandomization:
    def __init__(self, backend: Backend, terms: list[DomainRandomizationTerm]):
        self.backend = backend
        self.terms = tuple(terms)

    def apply_reset(self, physics_state: PhysicsState, rng: RNGKey):
        rngs = self.backend.rng_split(rng, len(self.terms))

        for term, rng in zip(self.terms, rngs):
            physics_state = term.apply(physics_state, rng)

        return physics_state
