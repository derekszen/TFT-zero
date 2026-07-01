from __future__ import annotations

import json
from pathlib import Path

from mini_tft.tools.judge_packet import (
    JudgePacketConfig,
    check_judge_verdict,
    generate_judge_packet,
)


def test_judge_packet_writes_reusable_read_only_artifacts(tmp_path: Path) -> None:
    report = generate_judge_packet(
        JudgePacketConfig(
            name="strategic-cache-smoke",
            deliverable="muzero_cache",
            out_root=tmp_path,
            objective="Judge the cache smoke packet.",
            summary="Small docs/tooling packet.",
            changed_files=("docs/ANTIGRAVITY_JUDGE.md",),
            evidence=("artifacts/strategic_lane/muzero_cache/metrics.json",),
            validation_commands=("env -u UV_PYTHON uv run pytest tests/test_judge_packet.py",),
            ai_router_state="available",
            ai_router_dir=tmp_path / "ai-router",
        )
    )

    out_dir = tmp_path / "strategic-cache-smoke"
    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    prompt = (out_dir / "prompt.md").read_text(encoding="utf-8")
    command = (out_dir / "antigravity_ai_router_command.txt").read_text(encoding="utf-8")

    assert report == metrics
    assert metrics["schema"] == "judge-packet/v1"
    assert metrics["status"] == "pending_verdict"
    assert metrics["judge"]["preferred_runner"] == "Antigravity via ai-router"
    assert metrics["judge"]["preferred_model"] == "gemini-3.5-flash-low"
    assert metrics["judge"]["thinking"] == "highest"
    assert metrics["judge"]["reasoning_effort"] == "xhigh"
    assert metrics["judge"]["provider"] == "ai-router CLIProxyAPI Antigravity OAuth"
    assert metrics["judge"]["ai_router"] == "available"
    assert metrics["judge"]["read_only"] is True
    assert metrics["judge"]["fail_closed"] is True
    assert metrics["deliverable"] == "muzero_cache"
    assert (out_dir / "evidence_manifest.json").exists()
    assert (out_dir / "verdict_template.md").exists()
    assert (out_dir / "decision.md").exists()
    assert (out_dir / "antigravity_ai_router_command.txt").exists()
    assert "Verdict: ACCEPT|REJECT" in prompt
    assert "Do not edit files" in prompt
    assert "artifacts/strategic_lane/muzero_cache/metrics.json" in prompt
    assert "bin/ai-env antigravity" in command
    assert "export MODEL=gemini-3.5-flash-low" in command
    assert "export REASONING_EFFORT=xhigh" in command
    assert "evidence_manifest.json" in command
    assert "Local Evidence Contents" in command
    assert "/chat/completions" in command
    assert "verdict.md" in command


def test_judge_verdict_checker_accepts_strict_accept(tmp_path: Path) -> None:
    verdict_path = tmp_path / "verdict.md"
    verdict_path.write_text(
        "\n".join(
            [
                "Verdict: ACCEPT",
                "Evidence checked:",
                "- docs/ANTIGRAVITY_JUDGE.md",
                "Findings:",
                "- None.",
                "Suggested action:",
                "- Complete the gate.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    report = check_judge_verdict(verdict_path)

    assert report["status"] == "accept"
    assert report["accepted"] is True
    assert report["fail_closed"] is False
    assert report["errors"] == []


def test_judge_verdict_checker_fails_closed_on_reject_missing_or_malformed(
    tmp_path: Path,
) -> None:
    reject_path = tmp_path / "reject.md"
    reject_path.write_text(
        "\n".join(
            [
                "Verdict: REJECT",
                "Evidence checked:",
                "- docs/ANTIGRAVITY_JUDGE.md",
                "Findings:",
                "- Missing validation.",
                "Suggested action:",
                "- Run tests.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    malformed_path = tmp_path / "malformed.md"
    malformed_path.write_text("Verdict: ACCEPT\nFindings:\n- None.\n", encoding="utf-8")
    fenced_path = tmp_path / "fenced.md"
    fenced_path.write_text(
        "\n".join(
            [
                "```md",
                "Verdict: ACCEPT",
                "Evidence checked:",
                "- docs/ANTIGRAVITY_JUDGE.md",
                "Findings:",
                "- None.",
                "Suggested action:",
                "- Complete the gate.",
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )

    reject_report = check_judge_verdict(reject_path)
    missing_report = check_judge_verdict(tmp_path / "missing.md")
    malformed_report = check_judge_verdict(malformed_path)
    fenced_report = check_judge_verdict(fenced_path)

    assert reject_report["status"] == "reject"
    assert reject_report["accepted"] is False
    assert reject_report["fail_closed"] is True
    assert missing_report["status"] == "blocked"
    assert missing_report["accepted"] is False
    assert malformed_report["status"] == "blocked"
    assert malformed_report["errors"]
    assert fenced_report["status"] == "blocked"
    assert fenced_report["accepted"] is False
    assert fenced_report["fail_closed"] is True
    assert any("first line" in error for error in fenced_report["errors"])
