# SPDX-FileCopyrightText: Copyright (c) 2026 Louis Gevers
# SPDX-License-Identifier: BSD-3-Clause
#
# See LICENSE file for full license information.
#
def get_backend(name: str):
    # Lazy import backends
    if name == "numpy":
        from .numpy import NumpyBackend

        return NumpyBackend()

    if name == "jax":
        from .jax import JaxBackend

        return JaxBackend()

    raise ValueError(f"Unknown backend: {name}")
