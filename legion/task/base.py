from typing import NamedTuple
from numpy.typing import ArrayLike

from legion.backend import Backend, RNGKey
from legion.physics import SensorData

from .signals import Signal
from .obs import ObsTerm
from .reward import RewardTerm
from .termination import TerminationTerm


class TaskState(NamedTuple):
    signals: tuple[ArrayLike, ...]


class Task:

    def __init__(
        self,
        backend: Backend,
        signals: list[Signal],
        observations: list[ObsTerm],
        rewards: list[RewardTerm],
        terminations: list[TerminationTerm],
    ):
        self.backend = backend

        # Store tuples
        self.signal_defs = tuple(signals)
        self.observations = tuple(observations)
        self.rewards = tuple(rewards)
        self.terminations = tuple(terminations)

        # Build static indices for runtime
        signal_index = {s.name: i for i, s in enumerate(self.signal_defs)}
        self._sig_indices_obs = self._build_sig_indices(
            signal_index, self.observations, "Observation"
        )
        self._sig_indices_rew = self._build_sig_indices(
            signal_index, self.rewards, "Reward"
        )
        self._sig_indices_ter = self._build_sig_indices(
            signal_index, self.terminations, "Termination"
        )

        # Build metric keys
        self._rew_metric_names = tuple(f"reward/{rew.name}" for rew in self.rewards)

    @property
    def metric_names_rewards(self) -> tuple[str, ...]:
        return self._rew_metric_names

    def reset(self, rng: RNGKey) -> TaskState:
        rng_signals = self.backend.rng_split(rng, len(self.signal_defs))
        signals = tuple(s.reset(r) for s, r in zip(self.signal_defs, rng_signals))
        return TaskState(signals=signals)

    def step(
        self,
        task_state: TaskState,
        sensor_data: SensorData,
        action: ArrayLike,
        dt: float,
        rng: RNGKey,
    ) -> TaskState:
        rng_signals = self.backend.rng_split(rng, len(self.signal_defs))
        signals = tuple(
            s.step(prev_signal, sensor_data, action, dt, r)
            for s, prev_signal, r in zip(
                self.signal_defs, task_state.signals, rng_signals
            )
        )
        return TaskState(signals=signals)

    def observe(self, task_state: TaskState, sensor_data: SensorData) -> ArrayLike:
        signals = tuple(
            # For each observation, collect the list of signals
            tuple(task_state.signals[i] for i in idx)
            # For each observation, collect the list of signal indices
            for idx in self._sig_indices_obs
        )
        observations = tuple(
            obs(signal, sensor_data) for obs, signal in zip(self.observations, signals)
        )
        return self.backend.concatenate(observations)

    def reward(
        self, task_state: TaskState, sensor_data: SensorData, action: ArrayLike
    ) -> tuple[ArrayLike, dict[str, ArrayLike]]:
        signals = tuple(
            # For each reward, collect the list of signals
            tuple(task_state.signals[i] for i in idx)
            # For each reward, collect the list of signal indices
            for idx in self._sig_indices_rew
        )
        # Compute individual reward terms
        rewards = self.backend.array(
            [
                rew(signal, sensor_data, action)
                for rew, signal in zip(self.rewards, signals)
            ]
        )
        # Compute weighted individual reward terms
        weighted_rewards = self.backend.array(
            [rew.weight * value for rew, value in zip(self.rewards, rewards)]
        )
        # Sum weighted rewards together
        total = self.backend.sum(weighted_rewards)
        # Put individual reward terms in metrics
        metrics_reward = dict(zip(self._rew_metric_names, weighted_rewards))

        return total, metrics_reward

    def terminate(self, task_state: TaskState, sensor_data: SensorData) -> ArrayLike:
        signals = tuple(
            # For each termination, collect the list of signals
            tuple(task_state.signals[i] for i in idx)
            # For each termination, collect the list of signal indices
            for idx in self._sig_indices_ter
        )
        # Compute individual termination terms
        terminations = self.backend.array(
            [
                ter(signal, sensor_data)
                for ter, signal in zip(self.terminations, signals)
            ]
        )
        # Return if any of the termination functions return true
        return self.backend.any(terminations)

    def _build_sig_indices(
        self, signal_index: dict[str, int], terms: tuple, term_name: str
    ) -> tuple[ArrayLike, ...]:
        # Build a tuple of indices that matches the signal indices per term
        all_indices = []
        for term in terms:
            # Sanity check: are all required signals there?
            for signal_name in term.required_signals:
                if signal_name not in signal_index:
                    raise ValueError(
                        f"Cannot construct task: {term_name} term '{term.name}' requires signal named '{signal_name}' (current signals: {[s.name for s in self.signal_defs]})"
                    )

            # Build indices
            term_indices = tuple(signal_index[s] for s in term.required_signals)

            # Append to all indices
            all_indices.append(term_indices)
        return tuple(all_indices)
