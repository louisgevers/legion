from typing import Protocol


class Embodiment(Protocol):
    name: str
    n_joints: int
    n_actuators: int
    n_feet: int
    joint_names: tuple[str, ...]
    actuator_names: tuple[str, ...]
    feet_names: tuple[str, ...]
    q_nominal: tuple[float, ...]
    q_directions: tuple[float, ...]
    base_xyz_init: tuple[float, float, float]
    total_mass: float
