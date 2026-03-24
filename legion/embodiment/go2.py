# SPDX-FileCopyrightText: Copyright (c) 2026 Louis Gevers
# SPDX-License-Identifier: BSD-3-Clause
#
# See LICENSE file for full license information.
#
from typing import NamedTuple

from legion.registry import EMBODIMENTS, register


@register(EMBODIMENTS, "go2")
class Go2(NamedTuple):
    name: str = "go2"
    n_joints: int = 12
    n_links: int = 13  # 3 per leg + base
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


@register(EMBODIMENTS, "go2_locked_hips")
class Go2LockedHips(NamedTuple):
    name: str = "go2_locked_hips"
    n_joints: int = 8
    n_links: int = 13  # 3 per leg + base
    n_actuators: int = 8
    n_feet: int = 4
    joint_names: tuple[str, ...] = tuple(
        [
            f"{leg}_{joint}_joint"
            for leg in ["FL", "FR", "RL", "RR"]
            for joint in ["thigh", "calf"]
        ]
    )
    actuator_names: tuple[str, ...] = tuple(
        [
            f"{leg}_{joint}"
            for leg in ["FL", "FR", "RL", "RR"]
            for joint in ["thigh", "calf"]
        ]
    )
    feet_names: tuple[str, ...] = ("FL", "FR", "RL", "RR")
    q_nominal: tuple[float, ...] = (0.8, -1.5) * 4
    q_directions: tuple[float, ...] = (1.0,) * 8
    base_xyz_init: tuple[float, float, float] = (0.0, 0.0, 0.38)
    total_mass: float = 15.0
