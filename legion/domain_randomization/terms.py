from abc import ABC, abstractmethod

from legion.backend import Backend, RNGKey
from legion.embodiment import Embodiment
from legion.physics import PhysicsEngine, PhysicsState
from legion.registry import DOMAIN_RANDOMIZATIONS, register


class DomainRandomizationTerm(ABC):
    name: str

    def __init__(
        self, backend: Backend, embodiment: Embodiment, physics: PhysicsEngine
    ):
        self.backend = backend
        self.embodiment = embodiment
        self.physics = physics

    @abstractmethod
    def apply(self, physics_state: PhysicsState, rng: RNGKey) -> PhysicsState: ...


@register(DOMAIN_RANDOMIZATIONS, "ground_friction")
class GroundFrictionRandomization(DomainRandomizationTerm):
    name = "ground_friction"

    def __init__(
        self,
        backend: Backend,
        embodiment: Embodiment,
        physics: PhysicsEngine,
        min: float,
        max: float,
    ):
        super().__init__(backend, embodiment, physics)
        self.min = min
        self.max = max

    def apply(self, physics_state: PhysicsState, rng: RNGKey):
        friction = self.backend.rng_uniform(rng, (), minval=self.min, maxval=self.max)
        return self.physics.set_ground_friction(physics_state, friction)


@register(DOMAIN_RANDOMIZATIONS, "scale_masses")
class ScaleMassesRandomization(DomainRandomizationTerm):
    name = "scale_masses"

    def __init__(
        self,
        backend: Backend,
        embodiment: Embodiment,
        physics: PhysicsEngine,
        min: float,
        max: float,
    ):
        super().__init__(backend, embodiment, physics)
        self.min = min
        self.max = max
        self.n_links = embodiment.n_joints + 1  # legs + base

    def apply(self, physics_state: PhysicsState, rng: RNGKey):
        scales = self.backend.rng_uniform(
            rng, (self.n_links,), minval=self.min, maxval=self.max
        )
        return self.physics.scale_masses(physics_state, scales)


@register(DOMAIN_RANDOMIZATIONS, "add_base_mass")
class AddBaseMassRandomization(DomainRandomizationTerm):
    name = "add_base_mass"

    def __init__(
        self,
        backend: Backend,
        embodiment: Embodiment,
        physics: PhysicsEngine,
        min: float,
        max: float,
    ):
        super().__init__(backend, embodiment, physics)
        self.min = min
        self.max = max

    def apply(self, physics_state: PhysicsState, rng: RNGKey):
        added_mass = self.backend.rng_uniform(rng, (), minval=self.min, maxval=self.max)
        return self.physics.add_base_mass(physics_state, added_mass)


@register(DOMAIN_RANDOMIZATIONS, "initial_joint_offsets")
class InitialJointOffsetsRandomization(DomainRandomizationTerm):
    name = "initial_joint_offsets"

    def __init__(
        self,
        backend: Backend,
        embodiment: Embodiment,
        physics: PhysicsEngine,
        min: float,
        max: float,
    ):
        super().__init__(backend, embodiment, physics)
        self.min = min
        self.max = max

    def apply(self, physics_state: PhysicsState, rng: RNGKey):
        offsets = self.backend.rng_uniform(
            rng, (self.embodiment.n_joints), minval=self.min, maxval=self.max
        )
        return self.physics.offset_joints(physics_state, offsets)
