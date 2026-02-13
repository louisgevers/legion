from typing import NamedTuple

from legion.registry import EMBODIMENTS, register


@register(EMBODIMENTS, "go2")
class Go2(NamedTuple):
    name = "go2"
    n_joints = 12
    n_actuators = 12
    n_feet = 4
    joint_names = tuple(
        [
            f"{leg}_{joint}_joint"
            for leg in ["FL", "FR", "RL", "RR"]
            for joint in ["hip", "thigh", "calf"]
        ]
    )
    actuator_names = tuple(
        [
            f"{leg}_{joint}"
            for leg in ["FL", "FR", "RL", "RR"]
            for joint in ["hip", "thigh", "calf"]
        ]
    )
    feet_names = ("FL", "FR", "RL", "RR")
    q_nominal = tuple([0.0, 0.9, -1.8] * 4)
    base_xyz_init = (0.0, 0.0, 0.34)
