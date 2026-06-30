"""Run a reproducible PufferLib 4 GPU trainer smoke for strategic_tft."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import time
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
OCEAN_SRC = REPO_ROOT / "src" / "mini_tft" / "strategic" / "ocean"
DEFAULT_PUFFERLIB_URL = "https://github.com/PufferAI/PufferLib.git"
DEFAULT_TIMESTEPS = 262_144


@dataclass(frozen=True)
class Puffer4GpuTrainerConfig:
    out_dir: Path
    pufferlib_root: Path
    pufferlib_ref: str
    timesteps: int = DEFAULT_TIMESTEPS
    checkpoint_interval: int = 1_000_000_000
    gpu_sample_interval_sec: float = 1.0
    skip_refresh: bool = False
    skip_build: bool = False


def run_puffer4_gpu_trainer(config: Puffer4GpuTrainerConfig) -> dict[str, Any]:
    config = replace(
        config,
        out_dir=config.out_dir.resolve(),
        pufferlib_root=config.pufferlib_root.resolve(),
    )
    if config.timesteps <= 0:
        raise ValueError("timesteps must be positive")
    if config.gpu_sample_interval_sec <= 0:
        raise ValueError("gpu_sample_interval_sec must be positive")

    config.out_dir.mkdir(parents=True, exist_ok=True)
    command_manifest = {
        "schema": "strategic-puffer4-gpu-trainer-command/v1",
        "repo_root": str(REPO_ROOT),
        "git_sha": _git_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT),
        "git_status_short": _git_output(["git", "status", "--short"], cwd=REPO_ROOT),
        "pufferlib_root": str(config.pufferlib_root),
        "pufferlib_ref": config.pufferlib_ref,
        "timesteps": config.timesteps,
    }
    _write_json(config.out_dir / "command.json", command_manifest)
    _write_text(config.out_dir / "nvidia_smi_before.txt", _nvidia_smi_text())
    _write_text(config.out_dir / "environment.txt", _environment_report())
    toolchain_report, toolchain_env = _prepare_local_toolchain(config.out_dir)

    setup_steps = _ensure_pufferlib_checkout(config)
    install_report = _install_strategic_env(config)
    pufferlib_sha = _git_output(["git", "rev-parse", "HEAD"], cwd=config.pufferlib_root)
    pufferlib_status = _git_output(["git", "status", "--short"], cwd=config.pufferlib_root)

    build_report = {"skipped": True}
    if not config.skip_build:
        build_report = _run_captured(
            ["bash", "build.sh", "strategic_tft"],
            cwd=config.pufferlib_root,
            stdout_path=config.out_dir / "puffer_build_stdout.txt",
            stderr_path=config.out_dir / "puffer_build_stderr.txt",
            env_update=toolchain_env,
        )

    checkpoint_dir = config.out_dir / "checkpoints"
    log_dir = config.out_dir / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    trainer_cmd = [
        "env",
        "-u",
        "UV_PYTHON",
        "uv",
        "run",
        "--all-extras",
        "python",
        "-m",
        "pufferlib.pufferl",
        "train",
        "strategic_tft",
        "--train.total-timesteps",
        str(config.timesteps),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--log-dir",
        str(log_dir),
        "--checkpoint-interval",
        str(config.checkpoint_interval),
    ]
    _write_text(config.out_dir / "trainer_command.txt", _shell_join(trainer_cmd) + "\n")
    _write_text(config.out_dir / "nvidia_smi_pre_train.txt", _nvidia_smi_text())
    if int(build_report.get("returncode", 0)) == 0:
        trainer_report = _run_trainer_with_gpu_sampling(
            trainer_cmd,
            cwd=config.pufferlib_root,
            out_dir=config.out_dir,
            sample_interval_sec=config.gpu_sample_interval_sec,
            env_update=toolchain_env,
        )
    else:
        trainer_report = {
            "command": _shell_join(trainer_cmd),
            "returncode": -1,
            "elapsed_sec": 0.0,
            "total_timesteps": _extract_total_timesteps(trainer_cmd),
            "stdout_path": "trainer_stdout.txt",
            "stderr_path": "trainer_stderr.txt",
            "skipped_reason": "PufferLib build failed; trainer was not launched.",
        }
        _write_text(config.out_dir / "trainer_stdout.txt", "")
        _write_text(config.out_dir / "trainer_stderr.txt", trainer_report["skipped_reason"] + "\n")
        _write_text(config.out_dir / "gpu_samples.jsonl", "")
    _write_text(config.out_dir / "nvidia_smi_after.txt", _nvidia_smi_text())

    checkpoints = _relative_paths(config.out_dir, checkpoint_dir.rglob("*"))
    logs = _relative_paths(config.out_dir, log_dir.rglob("*"))
    log_summary = _summarize_trainer_logs(log_dir)
    gpu_summary = _summarize_gpu_samples(config.out_dir / "gpu_samples.jsonl")
    status = _decision_status(
        build_returncode=int(build_report.get("returncode", 0)),
        trainer_returncode=int(trainer_report["returncode"]),
        checkpoints=checkpoints,
        logs=logs,
    )
    report = {
        "schema": "strategic-puffer4-gpu-trainer/v1",
        "status": status,
        "backend": "pufferlib_4_cuda_trainer",
        "claim_limit": "trainer_throughput_smoke",
        "repo_root": str(REPO_ROOT),
        "git_sha": command_manifest["git_sha"],
        "git_status_short": command_manifest["git_status_short"],
        "pufferlib": {
            "root": str(config.pufferlib_root),
            "requested_ref": config.pufferlib_ref,
            "git_sha": pufferlib_sha,
            "git_status_short": pufferlib_status,
            "setup_steps": setup_steps,
        },
        "strategic_env": install_report,
        "toolchain_workarounds": toolchain_report,
        "build": build_report,
        "trainer": trainer_report,
        "trainer_log_summary": log_summary,
        "gpu_summary": gpu_summary,
        "artifacts": {
            "command": "command.json",
            "trainer_command": "trainer_command.txt",
            "stdout": "trainer_stdout.txt",
            "stderr": "trainer_stderr.txt",
            "gpu_samples": "gpu_samples.jsonl",
            "nvidia_smi_before": "nvidia_smi_before.txt",
            "nvidia_smi_during": "gpu_samples.jsonl",
            "nvidia_smi_after": "nvidia_smi_after.txt",
            "checkpoints": checkpoints,
            "logs": logs,
        },
        "known_limits": [
            "strategic_tft is the simplified strategic lane, not legacy full TFT",
            "this is PufferLib 4 trainer throughput evidence, not MuZero policy quality",
            "a single smoke run is not a repeated matched speed promotion gate",
        ],
    }
    _write_json(config.out_dir / "metrics.json", report)
    _write_text(config.out_dir / "decision.md", _format_decision(report))
    _write_text(config.out_dir / "final_report.md", _format_final_report(report))
    return report


def _ensure_pufferlib_checkout(config: Puffer4GpuTrainerConfig) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    root = config.pufferlib_root
    if not root.exists():
        steps.append(
            _run_captured(
                ["git", "clone", DEFAULT_PUFFERLIB_URL, str(root)],
                cwd=REPO_ROOT,
                stdout_path=config.out_dir / "puffer_git_clone_stdout.txt",
                stderr_path=config.out_dir / "puffer_git_clone_stderr.txt",
            )
        )
    if not (root / ".git").exists():
        raise ValueError(f"PufferLib checkout does not look like a git repo: {root}")
    if not config.skip_refresh:
        steps.append(
            _run_captured(
                ["git", "fetch", "origin"],
                cwd=root,
                stdout_path=config.out_dir / "puffer_git_fetch_stdout.txt",
                stderr_path=config.out_dir / "puffer_git_fetch_stderr.txt",
            )
        )
        steps.append(
            _run_captured(
                ["git", "checkout", "--detach", config.pufferlib_ref],
                cwd=root,
                stdout_path=config.out_dir / "puffer_git_checkout_stdout.txt",
                stderr_path=config.out_dir / "puffer_git_checkout_stderr.txt",
            )
        )
    return steps


def _install_strategic_env(config: Puffer4GpuTrainerConfig) -> dict[str, Any]:
    target_dir = config.pufferlib_root / "ocean" / "strategic_tft"
    target_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in ("strategic_tft.h", "strategic_tft.c", "binding.c"):
        src = OCEAN_SRC / name
        dst = target_dir / name
        shutil.copy2(src, dst)
        copied.append({"source": str(src), "destination": str(dst)})
    config_dst = config.pufferlib_root / "config" / "strategic_tft.ini"
    config_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO_ROOT / "config" / "strategic_tft.ini", config_dst)
    copied.append(
        {
            "source": str(REPO_ROOT / "config" / "strategic_tft.ini"),
            "destination": str(config_dst),
        }
    )
    return {
        "env_name": "strategic_tft",
        "copied_files": copied,
        "config_text": (REPO_ROOT / "config" / "strategic_tft.ini").read_text(
            encoding="utf-8"
        ),
    }


def _run_trainer_with_gpu_sampling(
    cmd: Sequence[str],
    *,
    cwd: Path,
    out_dir: Path,
    sample_interval_sec: float,
    env_update: dict[str, str] | None = None,
) -> dict[str, Any]:
    stdout_path = out_dir / "trainer_stdout.txt"
    stderr_path = out_dir / "trainer_stderr.txt"
    samples_path = out_dir / "gpu_samples.jsonl"
    started = time.perf_counter()
    env = _command_env(env_update)
    with stdout_path.open("w", encoding="utf-8") as stdout_file:
        with stderr_path.open("w", encoding="utf-8") as stderr_file:
            process = subprocess.Popen(
                list(cmd),
                cwd=cwd,
                stdout=stdout_file,
                stderr=stderr_file,
                text=True,
                env=env,
            )
            with samples_path.open("w", encoding="utf-8") as samples_file:
                while process.poll() is None:
                    samples_file.write(json.dumps(_gpu_sample(process.pid)) + "\n")
                    samples_file.flush()
                    time.sleep(sample_interval_sec)
                samples_file.write(json.dumps(_gpu_sample(process.pid)) + "\n")
    elapsed = time.perf_counter() - started
    return {
        "command": _shell_join(cmd),
        "returncode": int(process.returncode or 0),
        "elapsed_sec": elapsed,
        "total_timesteps": _extract_total_timesteps(cmd),
        "stdout_path": "trainer_stdout.txt",
        "stderr_path": "trainer_stderr.txt",
    }


def _run_captured(
    cmd: Sequence[str],
    *,
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
    env_update: dict[str, str] | None = None,
) -> dict[str, Any]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    completed = subprocess.run(
        list(cmd),
        cwd=cwd,
        capture_output=True,
        text=True,
        env=_command_env(env_update),
    )
    elapsed = time.perf_counter() - started
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    return {
        "command": _shell_join(cmd),
        "returncode": completed.returncode,
        "elapsed_sec": elapsed,
        "stdout_path": stdout_path.name,
        "stderr_path": stderr_path.name,
    }


def _prepare_local_toolchain(out_dir: Path) -> tuple[dict[str, Any], dict[str, str]]:
    toolchain_dir = out_dir / "toolchain"
    bin_dir = toolchain_dir / "bin"
    lib_dir = toolchain_dir / "lib"
    bin_dir.mkdir(parents=True, exist_ok=True)
    lib_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "schema": "strategic-puffer4-toolchain-workarounds/v1",
        "bin_dir": str(bin_dir),
        "lib_dir": str(lib_dir),
        "created": [],
        "missing": [],
    }

    if shutil.which("ccache") is None:
        shim = bin_dir / "ccache"
        shim.write_text("#!/bin/sh\nexec \"$@\"\n", encoding="utf-8")
        shim.chmod(0o755)
        report["created"].append(
            {
                "path": str(shim),
                "reason": (
                    "PufferLib 4 build.sh invokes ccache before nvcc; "
                    "this shim execs the compiler directly."
                ),
            }
        )

    _link_first_existing(
        link=lib_dir / "libomp5.so",
        candidates=_library_candidates("libomp.so"),
        report=report,
    )
    _link_first_existing(
        link=lib_dir / "libnccl.so",
        candidates=_library_candidates("libnccl.so", "libnccl.so.2", python_package="nvidia.nccl"),
        report=report,
    )
    _link_first_existing(
        link=lib_dir / "libcudnn.so",
        candidates=_library_candidates(
            "libcudnn.so",
            "libcudnn.so.9",
            python_package="nvidia.cudnn",
        ),
        report=report,
    )

    env = {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        "LIBRARY_PATH": f"{lib_dir}{os.pathsep}{os.environ.get('LIBRARY_PATH', '')}",
        "LD_LIBRARY_PATH": f"{lib_dir}{os.pathsep}{os.environ.get('LD_LIBRARY_PATH', '')}",
    }
    _write_json(out_dir / "toolchain_workarounds.json", report)
    return report, env


def _link_first_existing(
    *,
    link: Path,
    candidates: Sequence[Path],
    report: dict[str, Any],
) -> None:
    for candidate in candidates:
        if candidate.exists():
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(candidate)
            report["created"].append(
                {
                    "path": str(link),
                    "target": str(candidate),
                    "reason": "artifact-scoped linker compatibility symlink",
                }
            )
            return
    report["missing"].append({"path": str(link), "candidates": [str(path) for path in candidates]})


def _library_candidates(*names: str, python_package: str | None = None) -> list[Path]:
    candidates: list[Path] = []
    for directory in (Path("/usr/lib"), Path("/usr/local/cuda/lib64")):
        candidates.extend(directory / name for name in names)
    if python_package is not None:
        package_paths = _python_package_paths(python_package)
        for package_path in package_paths:
            for name in names:
                candidates.extend(package_path.rglob(name))
    return candidates


def _python_package_paths(package_name: str) -> list[Path]:
    script = (
        "import importlib.util, json; "
        f"spec = importlib.util.find_spec({package_name!r}); "
        "print(json.dumps(list(spec.submodule_search_locations or []) if spec else []))"
    )
    completed = subprocess.run(
        ["python", "-c", script],
        capture_output=True,
        text=True,
        env=_command_env(),
    )
    if completed.returncode != 0:
        return []
    try:
        return [Path(path) for path in json.loads(completed.stdout)]
    except json.JSONDecodeError:
        return []


def _summarize_trainer_logs(log_dir: Path) -> dict[str, Any]:
    json_logs = sorted(log_dir.rglob("*.json"))
    latest: dict[str, Any] = {}
    for path in json_logs:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        latest = data if isinstance(data, dict) else latest
    return {
        "json_log_count": len(json_logs),
        "json_logs": [str(path.relative_to(log_dir.parent.parent)) for path in json_logs],
        "latest_keys": sorted(latest.keys()),
        "latest": latest,
        "steps_per_second": _find_numeric(latest, ("sps", "SPS", "steps_per_second")),
    }


def _summarize_gpu_samples(samples_path: Path) -> dict[str, Any]:
    if not samples_path.exists():
        return {"sample_count": 0}
    samples = [
        json.loads(line)
        for line in samples_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    max_memory_used = 0.0
    max_gpu_util = 0.0
    seen_compute_process = False
    for sample in samples:
        for gpu in sample.get("gpus", []):
            max_memory_used = max(max_memory_used, float(gpu.get("memory_used_mib", 0.0)))
            max_gpu_util = max(max_gpu_util, float(gpu.get("utilization_gpu_pct", 0.0)))
        if str(sample.get("trainer_pid")) in sample.get("compute_apps_raw", ""):
            seen_compute_process = True
    return {
        "sample_count": len(samples),
        "max_memory_used_mib": max_memory_used,
        "max_gpu_utilization_pct": max_gpu_util,
        "trainer_seen_in_compute_apps": seen_compute_process,
    }


def _gpu_sample(trainer_pid: int) -> dict[str, Any]:
    return {
        "captured_at": datetime.now(UTC).isoformat(),
        "trainer_pid": trainer_pid,
        "gpus": _parse_gpu_query(_nvidia_smi_query()),
        "compute_apps_raw": _nvidia_smi_compute_apps(),
    }


def _nvidia_smi_query() -> str:
    return _run_best_effort(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,name,index,memory.used,memory.total,utilization.gpu,utilization.memory",
            "--format=csv,noheader,nounits",
        ]
    )


def _nvidia_smi_compute_apps() -> str:
    return _run_best_effort(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )


def _nvidia_smi_text() -> str:
    return _run_best_effort(["nvidia-smi"])


def _parse_gpu_query(raw: str) -> list[dict[str, Any]]:
    rows = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 7:
            rows.append({"raw": line})
            continue
        rows.append(
            {
                "timestamp": parts[0],
                "name": parts[1],
                "index": _as_int(parts[2]),
                "memory_used_mib": _as_float(parts[3]),
                "memory_total_mib": _as_float(parts[4]),
                "utilization_gpu_pct": _as_float(parts[5]),
                "utilization_memory_pct": _as_float(parts[6]),
            }
        )
    return rows


def _decision_status(
    *,
    build_returncode: int,
    trainer_returncode: int,
    checkpoints: Sequence[str],
    logs: Sequence[str],
) -> str:
    if build_returncode != 0 or trainer_returncode != 0:
        return "fail"
    if not checkpoints or not logs:
        return "fail"
    return "smoke_only"


def _format_decision(report: dict[str, Any]) -> str:
    trainer = report["trainer"]
    gpu = report["gpu_summary"]
    return "\n".join(
        [
            "# PufferLib 4 Strategic GPU Trainer Decision",
            "",
            f"Status: `{report['status']}`",
            f"Backend: `{report['backend']}`",
            f"Claim limit: `{report['claim_limit']}`",
            f"Timesteps: `{trainer['total_timesteps']}`",
            f"Trainer return code: `{trainer['returncode']}`",
            f"Elapsed seconds: `{trainer['elapsed_sec']:.3f}`",
            f"GPU samples: `{gpu.get('sample_count', 0)}`",
            f"Max GPU memory MiB: `{gpu.get('max_memory_used_mib', 0.0)}`",
            f"Max GPU utilization pct: `{gpu.get('max_gpu_utilization_pct', 0.0)}`",
            "",
            "## Decision",
            "",
            "This is strategic-lane PufferLib 4 trainer throughput evidence. It is not "
            "legacy full TFT and not full MuZero policy quality.",
            "",
        ]
    )


def _format_final_report(report: dict[str, Any]) -> str:
    artifacts = report["artifacts"]
    checkpoints = artifacts["checkpoints"]
    logs = artifacts["logs"]
    lines = [
        "# PufferLib 4 Strategic GPU Trainer Final Report",
        "",
        f"Status: `{report['status']}`",
        f"Repo SHA: `{report['git_sha']}`",
        f"PufferLib SHA: `{report['pufferlib']['git_sha']}`",
        "",
        "## What This Proves",
        "",
        "- The strategic_tft Ocean C environment can be installed into a PufferLib 4 checkout.",
        "- The PufferLib 4 trainer command can be launched for the strategic lane.",
        "- Checkpoints, logs, and GPU telemetry are captured under one artifact directory.",
        "",
        "## What This Does Not Prove",
        "",
        "- This is not legacy full TFT.",
        "- This is not full MuZero learning quality.",
        "- This is not a repeated matched throughput promotion gate.",
        "",
        "## Key Artifacts",
        "",
        "- Metrics: `metrics.json`",
        "- Decision: `decision.md`",
        f"- Trainer stdout: `{artifacts['stdout']}`",
        f"- Trainer stderr: `{artifacts['stderr']}`",
        f"- GPU samples: `{artifacts['gpu_samples']}`",
        f"- Checkpoints: `{checkpoints}`",
        f"- Logs: `{logs}`",
        "",
    ]
    return "\n".join(lines)


def _relative_paths(root: Path, paths: Any) -> list[str]:
    out = []
    for path in sorted(Path(p) for p in paths):
        if path.is_file():
            out.append(str(path.relative_to(root)))
    return out


def _extract_total_timesteps(cmd: Sequence[str]) -> int | None:
    for idx, value in enumerate(cmd):
        if value == "--train.total-timesteps" and idx + 1 < len(cmd):
            return _as_int(cmd[idx + 1])
    return None


def _find_numeric(data: dict[str, Any], keys: Sequence[str]) -> float | None:
    stack: list[Any] = [data]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            for key, value in current.items():
                if key in keys and isinstance(value, int | float):
                    return float(value)
                if key in keys and isinstance(value, list):
                    numeric = _first_numeric(value)
                    if numeric is not None:
                        return numeric
                stack.append(value)
        elif isinstance(current, list):
            stack.extend(current)
    return None


def _first_numeric(values: Sequence[Any]) -> float | None:
    for value in values:
        if isinstance(value, int | float):
            return float(value)
    return None


def _read_default_puffer_ref() -> str:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    source = data.get("tool", {}).get("uv", {}).get("sources", {}).get("pufferlib", {})
    ref = source.get("rev")
    if not isinstance(ref, str) or not ref:
        raise ValueError("pyproject.toml does not define tool.uv.sources.pufferlib.rev")
    return ref


def _default_out_dir() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return REPO_ROOT / "artifacts" / "strategic_lane" / f"puffer4_gpu_trainer_{stamp}"


def _default_puffer_root() -> Path:
    return REPO_ROOT.parent / "TFT-zero-puffer4"


def _environment_report() -> str:
    lines = [
        "# Environment",
        "",
        "## PATH",
        "",
        os.environ.get("PATH", ""),
        "",
        "## ccache",
        "",
        shutil.which("ccache") or "not found",
        "",
        "## libnccl/libomp",
        "",
        _run_best_effort(["bash", "-lc", "ldconfig -p 2>/dev/null | rg 'libnccl|libomp' || true"]),
        "",
    ]
    return "\n".join(lines)


def _git_output(cmd: Sequence[str], *, cwd: Path) -> str:
    completed = subprocess.run(list(cmd), cwd=cwd, capture_output=True, text=True)
    if completed.returncode != 0:
        return completed.stderr.strip()
    return completed.stdout.strip()


def _run_best_effort(cmd: Sequence[str]) -> str:
    try:
        completed = subprocess.run(list(cmd), capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        return f"{type(exc).__name__}: {exc}"
    return (completed.stdout + completed.stderr).strip()


def _command_env(env_update: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("UV_PYTHON", None)
    if env_update:
        env.update(env_update)
    return env


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _shell_join(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _as_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--pufferlib-root", type=Path, default=None)
    parser.add_argument("--pufferlib-ref", default=None)
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--checkpoint-interval", type=int, default=1_000_000_000)
    parser.add_argument("--gpu-sample-interval-sec", type=float, default=1.0)
    parser.add_argument("--skip-refresh", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = Puffer4GpuTrainerConfig(
        out_dir=args.out_dir or _default_out_dir(),
        pufferlib_root=args.pufferlib_root or _default_puffer_root(),
        pufferlib_ref=args.pufferlib_ref or _read_default_puffer_ref(),
        timesteps=args.timesteps,
        checkpoint_interval=args.checkpoint_interval,
        gpu_sample_interval_sec=args.gpu_sample_interval_sec,
        skip_refresh=args.skip_refresh,
        skip_build=args.skip_build,
    )
    report = run_puffer4_gpu_trainer(config)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] != "fail" else 1


if __name__ == "__main__":
    raise SystemExit(main())
