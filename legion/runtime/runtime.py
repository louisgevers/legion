from typing import NamedTuple
from numpy.typing import ArrayLike

from legion.backend import Backend, RNGKey
from legion.embodiment import Embodiment
from legion.physics import PhysicsState, PhysicsEngine
from legion.actuator import ActuatorState, Actuator
from legion.task import TaskState, Task


class RuntimeState(NamedTuple):
    physics: PhysicsState
    actuator: ActuatorState
    task: TaskState
    rng: RNGKey


class RuntimeTransition(NamedTuple):
    state: RuntimeState
    obs: ArrayLike
    reward: ArrayLike
    done: ArrayLike


class Runtime:

    def __init__(
        self,
        embodiment: Embodiment,
        physics: PhysicsEngine,
        actuator: Actuator,
        task: Task,
        actuator_hz: float,
        policy_hz: float,
    ):
        self.embodiment = embodiment
        self.physics = physics
        self.actuator = actuator
        self.task = task

        # Compute static decimations for different control loop frequencies
        physics_hz = int(round(1.0 / physics.dt))
        assert (
            physics_hz % actuator_hz == 0
        ), f"Physics frequency ({physics_hz} Hz) must be a multiple of actuator frequency ({actuator_hz} Hz)"
        assert (
            physics_hz % policy_hz == 0
        ), f"Physics frequency ({physics_hz} Hz) must be a multiple of policy frequency ({policy_hz} Hz)"
        assert (
            actuator_hz % policy_hz == 0
        ), f"Actuator frequency ({actuator_hz} Hz) must be a multiple of policy frequency ({policy_hz} Hz)"
        self._actuator_decimation = physics_hz // actuator_hz
        self._policy_decimation = physics_hz // policy_hz

    # Useful hook for users
    @property
    def backend(self) -> Backend:
        return self.physics.backend

    def reset(self, rng: RNGKey) -> RuntimeState:
        rng, task_rng = self.backend.rng_split(rng, num=2)

        # Initial robot state
        q_init = self.backend.array(self.embodiment.q_nominal)
        base_xyz_init = self.backend.array(self.embodiment.base_xyz_init)

        return RuntimeState(
            physics=self.physics.reset(q=q_init, base_xyz=base_xyz_init),
            actuator=self.actuator.reset(),
            task=self.task.reset(task_rng),
            rng=rng,
        )

    def step(self, state: RuntimeState, action: ArrayLike) -> RuntimeTransition:
        # Generate new keys
        rng, task_rng = self.backend.rng_split(state.rng, num=2)

        # Apply action (steps actuator and physics at different frequencies, collects final states)
        physics_state, actuator_state = self._step_action(
            state.physics,
            state.actuator,
            action,
            n=self._policy_decimation,  # Runs physics for policy_decimation steps
        )

        # Read sensors
        sensor_data = self.physics.get_sensor_data(physics_state)

        # Compute transition components
        obs = self.task.observe(state.task, sensor_data)
        reward = self.task.reward(state.task, sensor_data, action)
        done = self.task.terminate(state.task, sensor_data)

        # If terminated early, set reward to 0
        reward = self.backend.where(done, 0, reward)

        # Clip negative rewards
        reward = self.backend.clip(reward, min=0, max=None)

        # Update signals for next step (AFTER the transition)
        task_state = self.task.step(state.task, sensor_data, action, task_rng)

        return RuntimeTransition(
            state=RuntimeState(
                physics=physics_state,
                actuator=actuator_state,
                task=task_state,
                rng=rng,
            ),
            obs=obs,
            reward=reward,
            done=done,
        )

    def observe(self, state: RuntimeState) -> ArrayLike:
        """Convenience function for current observations (e.g., after a reset)"""
        sensor_data = self.physics.get_sensor_data(state.physics)
        obs = self.task.observe(state.task, sensor_data)
        return obs

    # --- scannable functions --
    def _physics_step_fn(
        self, carry: tuple[PhysicsState, ArrayLike], _
    ) -> tuple[tuple[PhysicsState, ArrayLike], None]:
        physics_state, tau = carry

        # Apply torques and step the simulation one step
        physics_state = self.physics.apply_torques(physics_state, tau)
        physics_state = self.physics.step(physics_state)

        return (physics_state, tau), None

    def _actuator_step_fn(
        self, carry: tuple[PhysicsState, ActuatorState, ArrayLike], _
    ) -> tuple[tuple[PhysicsState, ActuatorState, ArrayLike], None]:
        physics_state, actuator_state, action = carry

        # Collect sensor data and compute torques
        sensor_data = self.physics.get_sensor_data(physics_state)
        tau = self.actuator.tau(action, sensor_data, actuator_state)

        # Advance simulation at actuator frequency
        physics_state = self._step_physics(
            physics_state, tau, n=self._actuator_decimation
        )

        # Step the actuator
        actuator_state = self.actuator.step(actuator_state)

        return (physics_state, actuator_state, action), None

    def _step_physics(
        self, physics_state: PhysicsState, tau: ArrayLike, n: int
    ) -> PhysicsState:
        (physics_state, _), _ = self.backend.scan(
            self._physics_step_fn,
            (physics_state, tau),
            None,
            length=n,
        )
        return physics_state

    def _step_action(
        self,
        physics_state: PhysicsState,
        actuator_state: ActuatorState,
        action: ArrayLike,
        n: int,
    ) -> tuple[PhysicsState, ActuatorState]:
        (physics_state, actuator_state, _), _ = self.backend.scan(
            self._actuator_step_fn,
            (physics_state, actuator_state, action),
            None,
            length=n
            // self._actuator_decimation,  # Internally runs actuator_decimation times
        )
        return physics_state, actuator_state
