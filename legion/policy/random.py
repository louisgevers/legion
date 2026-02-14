from numpy.typing import ArrayLike

from legion.backend import RNGKey, Backend, get_backend

from .base import Policy
from .registry import register_policy


@register_policy("random_uniform")
class RandomUniformPolicy(Policy):
    def __init__(self, backend: Backend, n_u: int, action_scaling: float | ArrayLike):
        self.backend = backend
        self.n_u = n_u
        self.action_scaling = action_scaling

    def action(self, obs: ArrayLike, rng: RNGKey) -> ArrayLike:
        return (
            self.backend.rng_uniform(rng, self.n_u, minval=-1.0, maxval=1.0)
            * self.action_scaling
        )

    def save_dict(self) -> dict:
        return {
            "backend": self.backend.name,
            "n_u": self.n_u,
            "action_scaling": self.action_scaling,
        }

    @classmethod
    def load_from_dict(cls, data: dict) -> "RandomUniformPolicy":
        backend = get_backend(data["backend"])
        return cls(backend, data["n_u"], data["action_scaling"])
