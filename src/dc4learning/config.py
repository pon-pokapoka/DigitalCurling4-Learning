"""Local settings from a `.env` file (not committed), so nothing environment-specific is in the code.

The file holds KEY=VALUE lines. It is looked up from the current directory upwards, so
notebooks in `notebooks/` find the one at the repo root. Variables already set in the
environment win over the file.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_FILE = ".env"
_loaded = False


def find_env_file(start: Path | None = None) -> Path | None:
    for d in [(start or Path.cwd()).resolve(), *(start or Path.cwd()).resolve().parents]:
        if (d / ENV_FILE).is_file():
            return d / ENV_FILE
    return None


def load_env() -> Path | None:
    """Load the nearest `.env` into os.environ once; return its path, if any."""
    global _loaded
    path = find_env_file()
    if _loaded or path is None:
        return path
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[len("export ") :]
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))
    _loaded = True
    return path
