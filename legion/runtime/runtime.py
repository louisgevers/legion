from typing import NamedTuple
from numpy.typing import ArrayLike

from legion.backend import Backend
from legion.physics import PhysicsState, PhysicsEngine
from legion.actuator import ActuatorState, Actuator
from legion.task import TaskState, Task


class RuntimeState(NamedTuple):
    physics: PhysicsState
    actuator: ActuatorState
    task: TaskState


class RuntimeTransition(NamedTuple):
    state: RuntimeState
    obs: ArrayLike
    reward: ArrayLike
    done: ArrayLike


class Runtime:

    def __init__(self, physics: PhysicsEngine, actuator: Actuator, task: Task):
        self.physics = physics
        self.actuator = actuator
        self.task = task

    # Useful hook for users
    @property
    def backend(self) -> Backend:
        return self.physics.backend

    def reset(self) -> RuntimeState:
        return RuntimeState(
            physics=self.physics.reset(),
            actuator=self.actuator.reset(),
            task=self.task.reset(),
        )

    def step(self, state: RuntimeState, action: ArrayLike) -> RuntimeTransition:
        # Read sensors (pre-action)
        sensor_data = self.physics.get_sensor_data(state.physics)

        # Apply action
        tau = self.actuator.tau(action, sensor_data, state.actuator)
        physics_state = self.physics.apply_torques(state.physics, tau)
        physics_state = self.physics.step(physics_state)

        # Read sensors (post-action)
        sensor_data = self.physics.get_sensor_data(physics_state)

        # Compute transition components
        obs = self.task.observe(state.task, sensor_data)
        reward = self.task.reward(state.task, sensor_data, action)
        done = self.task.terminate(state.task, sensor_data)

        # Update actuator states and signals for next step (AFTER the transition)
        actuator_state = self.actuator.step(state.actuator)
        task_state = self.task.step(state.task, sensor_data, action)

        return RuntimeTransition(
            state=RuntimeState(
                physics=physics_state,
                actuator=actuator_state,
                task=task_state,
            ),
            obs=obs,
            reward=reward,
            done=done,
        )
