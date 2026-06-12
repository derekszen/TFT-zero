"""Download local debug icons for the current unit roster."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, cast
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from mini_tft.core.set_data import load_set

DEFAULT_DDRAGON_VERSION = "9.19.1"
CHAMPION_DATA_URL = "https://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/champion.json"
CHAMPION_IMAGE_URL = "https://ddragon.leagueoflegends.com/cdn/{version}/img/champion/{filename}"


def download_unit_icons(
    output_root: Path = Path("assets"),
    version: str = DEFAULT_DDRAGON_VERSION,
    dataset: str = "set1",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Download unit icons and write `assets/manifest.json`."""

    data = load_set(dataset)
    champion_data = _fetch_json(CHAMPION_DATA_URL.format(version=version))
    champions = cast(dict[str, dict[str, Any]], champion_data["data"])
    champions_by_name = {
        _normalize_name(champion["name"]): champion
        for champion in champions.values()
    }

    units_dir = output_root / "units"
    units_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "source": {
            "name": "Riot Data Dragon",
            "version": version,
            "champion_data_url": CHAMPION_DATA_URL.format(version=version),
        },
        "units": {},
        "items": {},
    }
    unit_manifest: dict[str, dict[str, object]] = {}
    missing: list[str] = []

    for unit in data.units.values():
        champion = champions_by_name.get(_normalize_name(unit.name))
        if champion is None:
            missing.append(unit.name)
            continue

        filename = champion["image"]["full"]
        slug = _slug(unit.name)
        relative_path = Path("assets") / "units" / f"{slug}.png"
        output_path = output_root / "units" / f"{slug}.png"
        source_url = CHAMPION_IMAGE_URL.format(version=version, filename=filename)
        if overwrite or not output_path.exists():
            _download(source_url, output_path)
        unit_manifest[str(unit.id)] = {
            "name": unit.name,
            "path": relative_path.as_posix(),
            "source_url": source_url,
            "source_file": filename,
        }

    if missing:
        raise RuntimeError(f"missing champion assets for: {', '.join(missing)}")

    manifest["units"] = unit_manifest
    manifest_path = output_root / "manifest.json"
    manifest_json = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    manifest_path.write_text(manifest_json, encoding="utf-8")
    return manifest


def _fetch_json(url: str) -> dict[str, Any]:
    with urlopen(url, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _download(url: str, output_path: Path) -> None:
    try:
        with urlopen(url, timeout=30) as response:
            output_path.write_bytes(response.read())
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"failed to download {url}") from exc


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _slug(name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return normalized or "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default=DEFAULT_DDRAGON_VERSION)
    parser.add_argument("--dataset", default="set1")
    parser.add_argument("--out", type=Path, default=Path("assets"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest = download_unit_icons(
        output_root=args.out,
        version=args.version,
        dataset=args.dataset,
        overwrite=args.overwrite,
    )
    print(f"downloaded {len(manifest['units'])} unit icons to {args.out / 'units'}")
    print(f"wrote {args.out / 'manifest.json'}")


if __name__ == "__main__":
    main()
