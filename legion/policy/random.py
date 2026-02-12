from numpy.typing import ArrayLike

from legion.backend import RNGKey, Backend


class RandomUniformPolicy:
    def __init__(self, backend: Backend, n_u: int, action_scaling: float | ArrayLike):
        self.backend = backend
        self.n_u = n_u
        self.action_scaling = action_scaling

    def action(self, obs: ArrayLike, rng: RNGKey) -> ArrayLike:
        return (
            self.backend.rng_uniform(rng, self.n_u, minval=-1.0, maxval=1.0)
            * self.action_scaling
        )
