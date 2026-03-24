# SPDX-FileCopyrightText: Copyright (c) 2026 Louis Gevers
# SPDX-License-Identifier: BSD-3-Clause
#
# See LICENSE file for full license information.
#
def load_all():
    # Embodiments
    import legion.embodiment.go2

    # Physics
    import legion.physics.mujoco
    import legion.physics.mjx

    # Actuators
    import legion.actuator.torque
    import legion.actuator.pd
    import legion.actuator.ekeberg

    # Tasks
    import legion.task.signals
    import legion.task.obs
    import legion.task.reward
    import legion.task.termination

    # Domain randomization
    import legion.domain_randomization.terms
