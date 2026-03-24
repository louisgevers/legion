# SPDX-FileCopyrightText: Copyright (c) 2026 Louis Gevers
# SPDX-License-Identifier: BSD-3-Clause
#
# See LICENSE file for full license information.
#
POLICIES = {}


def register_policy(name):
    def decorator(cls):
        POLICIES[name] = cls
        cls._policy_name = name
        return cls

    return decorator
