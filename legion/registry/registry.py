# SPDX-FileCopyrightText: Copyright (c) 2026 Louis Gevers
# SPDX-License-Identifier: BSD-3-Clause
#
# See LICENSE file for full license information.
#

# Different registries
EMBODIMENTS: dict[str, type] = {}
PHYSICS: dict[str, type] = {}
ACTUATORS: dict[str, type] = {}
SIGNALS: dict[str, type] = {}
REWARDS: dict[str, type] = {}
OBSERVATIONS: dict[str, type] = {}
TERMINATIONS: dict[str, type] = {}
METRICS: dict[str, type] = {}
DOMAIN_RANDOMIZATIONS: dict[str, type] = {}


def register(registry: dict[str, type], name: str):
    def decorator(cls):
        registry[name] = cls
        return cls

    return decorator


def build(cfg: dict, registry: dict[str, type], **extra_kwargs):
    cfg = dict(cfg)  # Copy config
    cls_name = cfg.pop("type")
    cls = registry[cls_name]
    return cls(**cfg, **extra_kwargs)
