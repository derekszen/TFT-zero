"""Debug asset path helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def asset_path_for(root: Path, kind: str, asset_id: str, suffix: str = ".png") -> Path:
    """Return the conventional local debug asset path for an ID."""

    return root / kind / f"{asset_id}{suffix}"


def load_manifest(path: Path = Path("assets/manifest.json")) -> dict[str, Any]:
    """Load the local asset manifest if present."""

    if not path.exists():
        return {"units": {}, "items": {}}
    return json.loads(path.read_text(encoding="utf-8"))
