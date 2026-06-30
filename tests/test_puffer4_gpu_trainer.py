import json

from mini_tft.tools.run_puffer4_gpu_trainer import (
    _decision_status,
    _find_numeric,
    _parse_gpu_query,
    _summarize_gpu_samples,
)


def test_parse_gpu_query_converts_nvidia_smi_csv() -> None:
    rows = _parse_gpu_query(
        "2026/07/01 00:00:00.000, NVIDIA GeForce RTX 5090 D, 0, 3072, 32607, 91, 42\n"
    )

    assert rows == [
        {
            "timestamp": "2026/07/01 00:00:00.000",
            "name": "NVIDIA GeForce RTX 5090 D",
            "index": 0,
            "memory_used_mib": 3072.0,
            "memory_total_mib": 32607.0,
            "utilization_gpu_pct": 91.0,
            "utilization_memory_pct": 42.0,
        }
    ]


def test_summarize_gpu_samples_reports_maxes_and_process_visibility(tmp_path) -> None:
    samples = tmp_path / "gpu_samples.jsonl"
    samples.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "trainer_pid": 123,
                        "gpus": [{"memory_used_mib": 1024, "utilization_gpu_pct": 10}],
                        "compute_apps_raw": "",
                    }
                ),
                json.dumps(
                    {
                        "trainer_pid": 123,
                        "gpus": [{"memory_used_mib": 4096, "utilization_gpu_pct": 93}],
                        "compute_apps_raw": "123, python, 4096",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = _summarize_gpu_samples(samples)

    assert summary["sample_count"] == 2
    assert summary["max_memory_used_mib"] == 4096.0
    assert summary["max_gpu_utilization_pct"] == 93.0
    assert summary["trainer_seen_in_compute_apps"] is True


def test_decision_status_requires_build_trainer_checkpoint_and_log() -> None:
    assert (
        _decision_status(
            build_returncode=0,
            trainer_returncode=0,
            checkpoints=["checkpoints/strategic_tft/model.bin"],
            logs=["logs/strategic_tft/run.json"],
        )
        == "smoke_only"
    )
    assert (
        _decision_status(
            build_returncode=1,
            trainer_returncode=0,
            checkpoints=["checkpoints/strategic_tft/model.bin"],
            logs=["logs/strategic_tft/run.json"],
        )
        == "fail"
    )
    assert (
        _decision_status(
            build_returncode=0,
            trainer_returncode=0,
            checkpoints=[],
            logs=["logs/strategic_tft/run.json"],
        )
        == "fail"
    )


def test_find_numeric_extracts_puffer_list_metric() -> None:
    assert (
        _find_numeric(
            {"metrics": {"SPS": [11_747_044.0, 11_747_044.0, 11_747_044]}},
            ("SPS",),
        )
        == 11_747_044.0
    )
