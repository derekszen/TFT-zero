from __future__ import annotations

import shutil

import pytest

from mini_tft.tools.benchmark_puffer4_ocean import run_puffer4_ocean_benchmark


def test_puffer4_ocean_standalone_benchmark_writes_artifacts(tmp_path) -> None:
    if shutil.which("cc") is None:
        pytest.skip("C compiler is not available")

    report = run_puffer4_ocean_benchmark(out_dir=tmp_path, envs=8, steps=128)

    assert report["backend"] == "pufferlib_4_ocean_standalone"
    assert report["benchmark"]["steps"] == 128
    assert report["benchmark"]["steps_per_sec"] > 0.0
    assert report["puffer_build"]["attempted"] is False
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "decision.md").exists()
    assert (tmp_path / "strategic_tft_bench").exists()
