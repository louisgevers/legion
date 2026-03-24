# SPDX-FileCopyrightText: Copyright (c) 2026 Louis Gevers
# SPDX-License-Identifier: BSD-3-Clause
#
# See LICENSE file for full license information.
#
from abc import ABC, abstractmethod
from numpy.typing import ArrayLike

from legion.backend import RNGKey

from .registry import POLICIES


class Policy(ABC):

    @abstractmethod
    def action(self, obs: ArrayLike, rng: RNGKey) -> ArrayLike: ...

    @abstractmethod
    def save_dict(self) -> dict: ...

    @classmethod
    @abstractmethod
    def load_from_dict(cls, data: dict) -> "Policy": ...

    def save(self, path: str):
        import pickle

        with open(path, "wb") as f:
            base_dict = self.save_dict()
            full_dict = {"type": self._policy_name} | base_dict
            pickle.dump(full_dict, f)

    @classmethod
    def load(cls, path: str) -> "Policy":
        import pickle

        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls.load_from_dict(data)


class PolicyLoader:

    @staticmethod
    def _load_default_policies():
        import legion.policy.random
        import legion.policy.brax

    @staticmethod
    def load_from_dict(data: dict):
        # Populate the registry with default ones if needed
        PolicyLoader._load_default_policies()

        # Check policy type
        policy_type = data["type"]
        if policy_type not in POLICIES:
            raise ValueError(f"Unknown policy type '{policy_type}'")

        cls = POLICIES[policy_type]
        return cls.load_from_dict(data)

    @staticmethod
    def load(path: str):
        import pickle

        with open(path, "rb") as f:
            data = pickle.load(f)
        return PolicyLoader.load_from_dict(data)
