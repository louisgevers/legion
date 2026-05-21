# SPDX-FileCopyrightText: Copyright (c) 2026 Louis Gevers
# SPDX-License-Identifier: BSD-3-Clause
#
# See LICENSE file for full license information.
#
from typing import NamedTuple
from numpy.typing import ArrayLike

from legion.backend import Backend
from legion.registry import EMBODIMENTS, register


@register(EMBODIMENTS, "go2")
class Go2(NamedTuple):
    name: str = "go2"
    n_joints: int = 12
    n_links: int = 13  # 3 per leg + base
    n_actuators: int = 12
    n_feet: int = 4
    base_name: str = "base"
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

    def leg_ik(self, backend: Backend, xyz: ArrayLike) -> ArrayLike:
        # Extract desired coordinates
        x, y, z = xyz

        # Go2 link lengths
        l1, l2, l3 = 0.067, 0.213, 0.210

        # Pre-compute common quantities
        sqrt = backend.sqrt(y**2 + z**2 - l1**2)
        D = (x**2 + y**2 + z**2 - l1**2 - l2**2 - l3**2) / (2 * l2 * l3)

        # Analytical inverse kinematics
        theta3 = backend.atan2(-backend.sqrt(1 - D**2), D)
        theta2 = backend.atan2(-x, sqrt) - backend.atan2(
            l3 * backend.sin(theta3), l2 + l3 * backend.cos(theta3)
        )
        theta1 = -backend.atan2(z, -y) - backend.atan2(sqrt, -l1)

        return backend.array([theta1, theta2, theta3])


@register(EMBODIMENTS, "go2_locked_hips")
class Go2LockedHips(NamedTuple):
    name: str = "go2_locked_hips"
    n_joints: int = 8
    n_links: int = 13  # 3 per leg + base
    n_actuators: int = 8
    n_feet: int = 4
    base_name: str = "base"
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

    def leg_ik(self, backend: Backend, xyz: ArrayLike) -> ArrayLike:
        # Extract desired coordinates (ignore y as we cannot set it)
        x, _, z = xyz

        # Go2 link lengths (ignoring hip)
        l1, l2 = 0.213, 0.210

        # Pre-compute common quantities
        D = (x**2 + z**2 - l1**2 - l2**2) / (2 * l1 * l2)

        # Analytical inverse kinematics
        theta2 = backend.atan2(-backend.sqrt(1 - D**2), D)
        theta1 = backend.atan2(-x, -z) - backend.atan2(
            l2 * backend.sin(theta2), l1 + l2 * backend.cos(theta2)
        )

        return backend.array([theta1, theta2])
