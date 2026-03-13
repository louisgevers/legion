from typing import NamedTuple

from legion.registry import EMBODIMENTS, register


@register(EMBODIMENTS, "go2")
class Go2(NamedTuple):
    name: str = "go2"
    n_joints: int = 12
    n_actuators: int = 12
    n_feet: int = 4
    joint_names: tuple[str, ...] = tuple(
        [
            f"{leg}_{joint}_joint"
            for leg in ["FL", "FR", "RL", "RR"]
            for joint in ["hip", "thigh", "calf"]
        ]
    )
    actuator_names: tuple[str, ...] = tuple(
        [
            f"{leg}_{joint}"
            for leg in ["FL", "FR", "RL", "RR"]
            for joint in ["hip", "thigh", "calf"]
        ]
    )
    feet_names: tuple[str, ...] = ("FL", "FR", "RL", "RR")
    q_nominal: tuple[float, ...] = (0.0, 0.8, -1.5) * 2 + (0.0, 1.0, -1.5) * 2
    q_directions: tuple[float, ...] = (
        1.0,
        1.0,
        1.0,
        -1.0,
        1.0,
        1.0,
    ) * 2  # Make hips symmetric
    base_xyz_init: tuple[float, float, float] = (0.0, 0.0, 0.38)
    total_mass: float = 15.0
