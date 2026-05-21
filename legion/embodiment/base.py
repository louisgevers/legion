# SPDX-FileCopyrightText: Copyright (c) 2026 Louis Gevers
# SPDX-License-Identifier: BSD-3-Clause
#
# See LICENSE file for full license information.
#
from typing import Protocol
from numpy.typing import ArrayLike

from legion.backend import Backend


class Embodiment(Protocol):
    name: str
    n_joints: int
    n_links: int
    n_actuators: int
    n_feet: int
    base_name: str
    joint_names: tuple[str, ...]
    actuator_names: tuple[str, ...]
    feet_names: tuple[str, ...]
    q_nominal: tuple[float, ...]
    q_directions: tuple[float, ...]
    base_xyz_init: tuple[float, float, float]
    total_mass: float

    def leg_ik(self, backend: Backend, xyz: ArrayLike) -> ArrayLike: ...
