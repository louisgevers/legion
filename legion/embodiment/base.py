from typing import Protocol


class Embodiment(Protocol):
    name: str
    n_joints: int
    n_actuators: int
    joint_names: tuple[str, ...]
    actuator_names: tuple[str, ...]
    q_nominal: tuple[float, ...]
