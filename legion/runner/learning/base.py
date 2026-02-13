from abc import ABC, abstractmethod
import datetime

from legion.policy import Policy
from legion.runtime import Runtime


class LearningRunner(ABC):
    def __init__(self, learning_iterations: int, n_envs: int, rollout_length: int):
        self.learning_iterations = learning_iterations
        self.n_envs = n_envs
        self.rollout_length = rollout_length
        self.batch_size = n_envs * rollout_length
        self.n_iterations = self.batch_size * learning_iterations

    def learn(self, runtime: Runtime, algo_cfg: dict) -> Policy:
        # Print some stats
        sim_steps = runtime._policy_decimation * self.n_iterations
        sim_duration = sim_steps * runtime.physics.dt
        rollout_duration = self.rollout_length * runtime._policy_dt
        print(f"[{self.__class__.__name__}] initialized with:")
        print(
            f"- batch size:\t\t\t{self.batch_size:_} ({self.n_envs:_} envs x {self.rollout_length:_} steps)"
        )
        print(
            f"- learning iterations:\t\t{self.learning_iterations:_} ({self.n_iterations:_} policy steps)"
        )
        print(
            f"- rollout length:\t\t{self.rollout_length} (real time: {rollout_duration:.2f}s)"
        )
        print(
            f"- simulation iterations:\t{sim_steps:_} (real time: {datetime.timedelta(seconds=int(sim_duration))})"
        )

        return self.learn_impl(runtime, algo_cfg)

    @abstractmethod
    def learn_impl(self, runtime: Runtime, algo_cfg: dict) -> Policy: ...
