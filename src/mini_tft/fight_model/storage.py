"""Zarr storage for fight labels."""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import zarr
from numpy.typing import NDArray

from mini_tft.fight_model.schema import FIGHT_ARRAY_DTYPES, SCHEMA_VERSION


@dataclass(frozen=True)
class ShardInfo:
    name: str
    path: str
    count: int
    created_at: float
    teacher: str
    sampling: str
    elapsed_sec: float
    failures: int


def write_shard_atomic(
    root: Path,
    shard_name: str,
    arrays: dict[str, NDArray[Any]],
    metadata: dict[str, Any],
) -> ShardInfo:
    root.mkdir(parents=True, exist_ok=True)
    shard_path = root / "shards" / f"{shard_name}.zarr"
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    _validate_arrays(arrays)
    count = int(next(iter(arrays.values())).shape[0])
    if shard_path.exists():
        raise FileExistsError(f"shard already exists: {shard_path}")

    with tempfile.TemporaryDirectory(dir=shard_path.parent) as temp_dir:
        temp_path = Path(temp_dir) / f"{shard_name}.zarr.tmp"
        group = zarr.open_group(temp_path, mode="w", zarr_format=3)
        for name, array in arrays.items():
            chunks = (min(1024, max(1, count)), *array.shape[1:])
            group.create_array(name, data=array, chunks=chunks)
        group.attrs.update({"schema_version": SCHEMA_VERSION, **metadata, "count": count})
        os.replace(temp_path, shard_path)

    info = ShardInfo(
        name=shard_name,
        path=str(shard_path.relative_to(root)),
        count=count,
        created_at=time.time(),
        teacher=str(metadata.get("teacher", "")),
        sampling=str(metadata.get("sampling", "")),
        elapsed_sec=float(metadata.get("elapsed_sec", 0.0)),
        failures=int(metadata.get("failures", 0)),
    )
    append_manifest(root, info, metadata)
    return info


def append_manifest(root: Path, shard: ShardInfo, metadata: dict[str, Any]) -> None:
    manifest_path = root / "manifest.json"
    lock_path = root / "manifest.lock"
    root.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        manifest = read_manifest(root) if manifest_path.exists() else _new_manifest(metadata)
        existing_names = {row["name"] for row in manifest["shards"]}
        if shard.name not in existing_names:
            manifest["shards"].append(shard.__dict__)
        manifest["total_fights"] = int(sum(row["count"] for row in manifest["shards"]))
        manifest["updated_at"] = time.time()
        with tempfile.NamedTemporaryFile(
            "w",
            dir=root,
            delete=False,
            encoding="utf-8",
            prefix="manifest.",
            suffix=".tmp",
        ) as temp_file:
            temp_file.write(json.dumps(manifest, indent=2, sort_keys=True))
            temp_name = temp_file.name
        os.replace(temp_name, manifest_path)


def read_manifest(root: Path) -> dict[str, Any]:
    return json.loads((root / "manifest.json").read_text(encoding="utf-8"))


def load_arrays(root: Path) -> dict[str, NDArray[Any]]:
    manifest = read_manifest(root)
    shards = manifest.get("shards", [])
    if not shards:
        raise ValueError(f"no shards found in {root}")
    loaded: dict[str, list[NDArray[Any]]] = {name: [] for name in FIGHT_ARRAY_DTYPES}
    for shard in shards:
        group = zarr.open_group(root / shard["path"], mode="r")
        for name in FIGHT_ARRAY_DTYPES:
            loaded[name].append(np.asarray(group[name]))
    return {name: np.concatenate(parts, axis=0) for name, parts in loaded.items()}


def validate_dataset(root: Path) -> dict[str, Any]:
    manifest = read_manifest(root)
    total = 0
    for shard in manifest.get("shards", []):
        group = zarr.open_group(root / shard["path"], mode="r")
        count = int(shard["count"])
        total += count
        for name, dtype in FIGHT_ARRAY_DTYPES.items():
            if name not in group:
                raise ValueError(f"{shard['name']} missing array {name}")
            array = cast(zarr.Array, group[name])
            if array.shape[0] != count:
                raise ValueError(f"{shard['name']} array {name} has wrong leading length")
            if np.dtype(array.dtype) != dtype:
                raise ValueError(
                    f"{shard['name']} array {name} has dtype {array.dtype}, expected {dtype}"
                )
    if total != manifest.get("total_fights"):
        raise ValueError("manifest total_fights does not match shard counts")
    return {"total_fights": total, "shards": len(manifest.get("shards", []))}


def completed_shard_names(root: Path) -> set[str]:
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return set()
    manifest = read_manifest(root)
    return {row["name"] for row in manifest.get("shards", [])}


def _new_manifest(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": time.time(),
        "updated_at": time.time(),
        "teacher": metadata.get("teacher", ""),
        "sampling": metadata.get("sampling", ""),
        "total_fights": 0,
        "shards": [],
    }


def _validate_arrays(arrays: dict[str, NDArray[Any]]) -> None:
    missing = set(FIGHT_ARRAY_DTYPES) - set(arrays)
    if missing:
        raise ValueError(f"missing arrays: {sorted(missing)}")
    counts = {array.shape[0] for array in arrays.values()}
    if len(counts) != 1:
        raise ValueError("all arrays must have the same leading length")
    for name, dtype in FIGHT_ARRAY_DTYPES.items():
        if arrays[name].dtype != dtype:
            raise ValueError(f"{name} dtype {arrays[name].dtype} != {dtype}")
