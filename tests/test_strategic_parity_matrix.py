from __future__ import annotations

import json
import shutil

import pytest

from mini_tft.tools.strategic_parity_matrix import (
    SCENARIOS,
    StrategicParityMatrixConfig,
    run_strategic_parity_matrix,
)


def test_strategic_parity_matrix_accepts_python_native_and_ocean_c(tmp_path) -> None:
    if shutil.which("cc") is None:
        pytest.skip("C compiler is not available")

    report = run_strategic_parity_matrix(
        StrategicParityMatrixConfig(
            out_dir=tmp_path,
            seeds=(0, 7),
            scenarios=SCENARIOS,
        )
    )

    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    matrix_rows = [
        json.loads(line)
        for line in (tmp_path / "matrix.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert report["status"] == "pass"
    assert metrics["status"] == "pass"
    assert metrics["summary"]["failed"] == 0
    assert metrics["summary"]["total_checks"] == len(matrix_rows)
    assert {row["backend"] for row in matrix_rows} == {"native_cpp", "ocean_c"}
    assert {row["scenario"] for row in matrix_rows} == {scenario.name for scenario in SCENARIOS}
    assert all(row["mismatch_count"] == 0 for row in matrix_rows)
    assert (tmp_path / "decision.md").exists()


def test_strategic_parity_matrix_fuzz_cases_are_replayable(tmp_path) -> None:
    if shutil.which("cc") is None:
        pytest.skip("C compiler is not available")

    report = run_strategic_parity_matrix(
        StrategicParityMatrixConfig(
            out_dir=tmp_path,
            seeds=tuple(),
            scenarios=tuple(),
            fuzz_episodes=2,
            fuzz_max_steps=24,
            fuzz_policies=("random_legal", "random_mixed", "heuristic"),
            fuzz_seed=123,
        )
    )

    matrix_rows = [
        json.loads(line)
        for line in (tmp_path / "matrix.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    mixed_rows = [
        row
        for row in matrix_rows
        if row["scenario_kind"] == "fuzz" and row["policy"] == "random_mixed"
    ]

    assert report["status"] == "pass"
    assert report["fuzz"]["cases"] == 6
    assert report["summary"]["total_checks"] == 12
    assert {row["scenario_kind"] for row in matrix_rows} == {"fuzz"}
    assert {row["policy"] for row in matrix_rows} == {"random_legal", "random_mixed", "heuristic"}
    assert all(row["mismatch_count"] == 0 for row in matrix_rows)
    assert any(any(action >= 11 for action in row["actions"]) for row in mixed_rows)
