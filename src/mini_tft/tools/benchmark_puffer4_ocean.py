"""Build and benchmark the PufferLib 4.0 Ocean-style strategic C env."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
OCEAN_SRC = REPO_ROOT / "src" / "mini_tft" / "strategic" / "ocean"


def run_puffer4_ocean_benchmark(
    *,
    out_dir: Path,
    envs: int,
    steps: int,
    cc: str = "cc",
    pufferlib_root: Path | None = None,
    try_puffer_build: bool = False,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    binary = out_dir / "strategic_tft_bench"
    compile_cmd = [
        cc,
        "-O3",
        "-std=c11",
        "-D_POSIX_C_SOURCE=199309L",
        "-I",
        str(OCEAN_SRC),
        str(OCEAN_SRC / "strategic_tft.c"),
        "-lm",
        "-o",
        str(binary),
    ]
    subprocess.run(compile_cmd, cwd=REPO_ROOT, check=True)
    completed = subprocess.run(
        [str(binary), str(envs), str(steps)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    benchmark = json.loads(completed.stdout)

    puffer_build: dict[str, Any] = {"attempted": False}
    if try_puffer_build:
        if pufferlib_root is None:
            raise ValueError("--try-puffer-build requires --pufferlib-root")
        puffer_build = _try_puffer_build(
            pufferlib_root=pufferlib_root,
            out_dir=out_dir,
        )

    report = {
        "schema": "strategic-puffer4-ocean-benchmark/v1",
        "backend": "pufferlib_4_ocean_standalone",
        "source": str(OCEAN_SRC),
        "compile_cmd": compile_cmd,
        "benchmark": benchmark,
        "puffer_build": puffer_build,
        "artifacts": ["metrics.json", "decision.md", "strategic_tft_bench"],
        "known_limits": [
            "standalone benchmark exercises the Ocean C env loop, not the full Puffer trainer",
            "the env mirrors strategic rules for throughput exploration; scalar parity is "
            "covered by Python/native tests",
            "PufferLib 4.0 training requires building the env inside a PufferLib 4.0 "
            "source checkout",
        ],
    }
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "decision.md").write_text(_format_decision(report), encoding="utf-8")
    return report


def _try_puffer_build(*, pufferlib_root: Path, out_dir: Path) -> dict[str, Any]:
    if not (pufferlib_root / "build.sh").exists():
        raise ValueError(f"PufferLib 4.0 checkout not found: {pufferlib_root}")
    target_dir = pufferlib_root / "ocean" / "strategic_tft"
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in ("strategic_tft.h", "strategic_tft.c", "binding.c"):
        shutil.copy2(OCEAN_SRC / name, target_dir / name)
    shutil.copy2(
        REPO_ROOT / "config" / "strategic_tft.ini",
        pufferlib_root / "config" / "strategic_tft.ini",
    )

    completed = subprocess.run(
        ["bash", "build.sh", "strategic_tft", "--local"],
        cwd=pufferlib_root,
        capture_output=True,
        text=True,
    )
    (out_dir / "puffer_build_stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (out_dir / "puffer_build_stderr.txt").write_text(completed.stderr, encoding="utf-8")
    return {
        "attempted": True,
        "returncode": completed.returncode,
        "command": "bash build.sh strategic_tft --local",
        "stdout_path": "puffer_build_stdout.txt",
        "stderr_path": "puffer_build_stderr.txt",
    }


def _format_decision(report: dict[str, Any]) -> str:
    bench = report["benchmark"]
    puffer_build = report["puffer_build"]
    lines = [
        "# Strategic PufferLib 4 Ocean Benchmark",
        "",
        f"Backend: `{report['backend']}`",
        f"Steps/sec: `{bench['steps_per_sec']:.2f}`",
        f"Envs: `{bench['envs']}`",
        f"Steps: `{bench['steps']}`",
        f"Episodes: `{bench['episodes']:.0f}`",
        f"Mean placement: `{bench['mean_placement']:.3f}`",
        "",
        "## Puffer 4 Build",
        "",
    ]
    if puffer_build["attempted"]:
        lines.append(f"- Command: `{puffer_build['command']}`")
        lines.append(f"- Return code: `{puffer_build['returncode']}`")
    else:
        lines.append("- Not attempted.")
    lines.extend(
        [
            "",
            "## Limits",
            "",
            *[f"- {limit}" for limit in report["known_limits"]],
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/strategic_lane/puffer4_ocean"),
    )
    parser.add_argument("--envs", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=10_000_000)
    parser.add_argument("--cc", default="cc")
    parser.add_argument("--pufferlib-root", type=Path)
    parser.add_argument("--try-puffer-build", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = run_puffer4_ocean_benchmark(
        out_dir=args.out_dir,
        envs=args.envs,
        steps=args.steps,
        cc=args.cc,
        pufferlib_root=args.pufferlib_root,
        try_puffer_build=args.try_puffer_build,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
