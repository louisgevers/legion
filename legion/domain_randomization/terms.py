from abc import ABC, abstractmethod
from numpy.typing import ArrayLike

from legion.backend import Backend, RNGKey
from legion.embodiment import Embodiment
from legion.physics import PhysicsEngine, PhysicsState, SensorData
from legion.registry import DOMAIN_RANDOMIZATIONS, register


class DomainRandomizationTerm(ABC):
    name: str

    def __init__(
        self, backend: Backend, embodiment: Embodiment, physics: PhysicsEngine
    ):
        self.backend = backend
        self.embodiment = embodiment
        self.physics = physics

    def reset(self, rng: RNGKey) -> dict:
        # Optional internal state
        return {}

    def step(self, state: dict, sensor_data: SensorData, rng: RNGKey) -> dict:
        # Optional internal state
        return state

    # Change physics state at reset time
    def apply_reset(
        self, state: dict, physics_state: PhysicsState, rng: RNGKey
    ) -> PhysicsState:
        return physics_state

    # Change physics at step time
    def apply_step(
        self,
        state: dict,
        physics_state: PhysicsState,
        sensor_data: SensorData,
        rng: RNGKey,
    ) -> PhysicsState:
        return physics_state


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

    def apply_reset(self, state: dict, physics_state: PhysicsState, rng: RNGKey):
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
        self.n_links = embodiment.n_links

    def apply_reset(self, state: dict, physics_state: PhysicsState, rng: RNGKey):
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

    def apply_reset(self, state: dict, physics_state: PhysicsState, rng: RNGKey):
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

    def apply_reset(self, state: dict, physics_state: PhysicsState, rng: RNGKey):
        offsets = self.backend.rng_uniform(
            rng, (self.embodiment.n_joints), minval=self.min, maxval=self.max
        )
        return self.physics.offset_joints(physics_state, offsets)


@register(DOMAIN_RANDOMIZATIONS, "push_robot")
class PushRobotRandomization(DomainRandomizationTerm):
    name = "push_robot"

    def __init__(
        self,
        backend: Backend,
        embodiment: Embodiment,
        physics: PhysicsEngine,
        interval: float,
        duration: float,
        max_velocity: float,
    ):
        super().__init__(backend, embodiment, physics)
        self.interval = interval
        self.duration = duration
        self.max_velocity = max_velocity

    def reset(self, rng: RNGKey) -> dict:
        interval_rng, direction_rng, velocity_rng = self.backend.rng_split(rng, 3)
        return {
            "next_push_s": self._sample_interval(interval_rng),
            "direction": self._sample_direction(direction_rng),
            "velocity": self._sample_velocity(velocity_rng),
        }

    def step(self, state: dict, sensor_data: SensorData, rng: RNGKey):
        interval_rng, direction_rng, velocity_rng = self.backend.rng_split(rng, 3)
        should_resample = sensor_data.t > state["next_push_s"] + self.duration
        return {
            "next_push_s": self.backend.where(
                should_resample,
                sensor_data.t + self._sample_interval(interval_rng),
                state["next_push_s"],
            ),
            "direction": self.backend.where(
                should_resample,
                self._sample_direction(direction_rng),
                state["direction"],
            ),
            "velocity": self.backend.where(
                should_resample,
                self._sample_velocity(velocity_rng),
                state["velocity"],
            ),
        }

    def apply_step(
        self,
        state: dict,
        physics_state: PhysicsState,
        sensor_data: SensorData,
        rng: RNGKey,
    ):
        # Compute force perturbation
        force = self.backend.where(
            sensor_data.t < state["next_push_s"],
            0.0,  # No perturbation
            (  # Compute force (N = kg * m / s ^ 2)
                self.embodiment.total_mass  # kg
                * state["velocity"]  # m/s
                / self.duration  # 1 / s
            ),
        )
        direction = state["direction"]

        return self.physics.apply_base_perturbation(physics_state, force * direction)

    def _sample_interval(self, rng: RNGKey) -> ArrayLike:
        return self.backend.rng_exponential(rng, ()) * self.interval

    def _sample_direction(self, rng: RNGKey) -> ArrayLike:
        angle = self.backend.rng_uniform(rng, (), minval=0, maxval=2 * self.backend.pi)
        return self.backend.array(
            [self.backend.cos(angle), self.backend.sin(angle), 0.0]
        )

    def _sample_velocity(self, rng: RNGKey) -> float:
        return self.backend.rng_uniform(rng, (), minval=0, maxval=self.max_velocity)
