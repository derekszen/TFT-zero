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
