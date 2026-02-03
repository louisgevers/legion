from typing import NamedTuple


class Go2(NamedTuple):
    name = "go2"
    n_joints = 12
    n_actuators = 12
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
