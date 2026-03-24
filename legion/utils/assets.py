# SPDX-FileCopyrightText: Copyright (c) 2026 Louis Gevers
# SPDX-License-Identifier: BSD-3-Clause
#
# See LICENSE file for full license information.
#
from pathlib import Path

ASSETS_DIR = (Path(__file__).parent.parent.parent / "assets").resolve(True)


def get_asset_path(name: str) -> str:
    if name.startswith("legion/"):
        name = name.replace("legion/", "", 1)
        assets_path = ASSETS_DIR / name
        return str(assets_path.resolve(True))
    return name
