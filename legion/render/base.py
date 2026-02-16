import numpy as np
from typing import Protocol


from legion.physics import PhysicsState


class Renderer(Protocol):
    def render(self, state: PhysicsState) -> np.ndarray: ...
    def close(self): ...
