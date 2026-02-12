from typing import Protocol
from numpy.typing import ArrayLike

from legion.backend import RNGKey


class Policy(Protocol):
    def action(self, obs: ArrayLike, rng: RNGKey) -> ArrayLike: ...
