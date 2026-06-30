"""Generate reusable read-only judge packets for MiniTFT gates."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DELIVERABLES = ("muzero_cache", "puffer_speed", "playable_demo", "docs", "other")
PREFERRED_JUDGE = "Antigravity"
PREFERRED_MODEL = "Flash 3.5"
PREFERRED_THINKING = "high"
GEMINI_FALLBACK_MODEL = "gemini-3.5-flash"
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SECTION_HEADERS = ("Evidence checked:", "Findings:", "Suggested action:")


@dataclass(frozen=True)
class JudgePacketConfig:
    name: str
    deliverable: str = "other"
    out_root: Path = Path("artifacts/judge")
    objective: str = "Independent read-only judge review."
    summary: str = ""
    changed_files: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    validation_commands: tuple[str, ...] = ()
    antigravity_cli_state: str = "auto"
    gemini_cli_state: str = "auto"


def generate_judge_packet(config: JudgePacketConfig) -> dict[str, Any]:
    """Write a reusable judge packet and return its metrics payload."""

    _validate_config(config)
    out_dir = config.out_root / config.name
    out_dir.mkdir(parents=True, exist_ok=True)

    antigravity_cli_state = _resolve_cli_state("antigravity", config.antigravity_cli_state)
    gemini_cli_state = _resolve_cli_state("gemini", config.gemini_cli_state)
    created_at = datetime.now(UTC).isoformat()

    manifest: dict[str, Any] = {
        "schema": "judge-evidence-manifest/v1",
        "name": config.name,
        "deliverable": config.deliverable,
        "objective": config.objective,
        "summary": config.summary,
        "changed_files": list(config.changed_files),
        "evidence": list(config.evidence),
        "validation_commands": list(config.validation_commands),
        "created_at": created_at,
        "judge_contract": _judge_contract(),
    }

    artifacts = [
        "prompt.md",
        "evidence_manifest.json",
        "verdict_template.md",
        "gemini_fallback_command.txt",
        "decision.md",
        "metrics.json",
    ]
    metrics: dict[str, Any] = {
        "schema": "judge-packet/v1",
        "name": config.name,
        "deliverable": config.deliverable,
        "status": "pending_verdict",
        "out_dir": str(out_dir),
        "created_at": created_at,
        "objective": config.objective,
        "judge": {
            "preferred_runner": PREFERRED_JUDGE,
            "preferred_model": PREFERRED_MODEL,
            "thinking": PREFERRED_THINKING,
            "read_only": True,
            "fail_closed": True,
            "antigravity_cli": antigravity_cli_state,
            "gemini_cli": gemini_cli_state,
        },
        "changed_files": list(config.changed_files),
        "evidence": list(config.evidence),
        "validation_commands": list(config.validation_commands),
        "artifacts": artifacts,
        "strict_output_schema": _judge_contract(),
        "known_limits": [
            "Antigravity is treated as an external IDE/manual runner unless a local CLI exists.",
            (
                "The packet only prepares evidence; the gate remains blocked until a strict "
                "ACCEPT verdict."
            ),
            "Gemini CLI fallback command shape is documented separately and must stay read-only.",
        ],
    }

    _write_json(out_dir / "evidence_manifest.json", manifest)
    _write_json(out_dir / "metrics.json", metrics)
    (out_dir / "prompt.md").write_text(_format_prompt(config, manifest), encoding="utf-8")
    (out_dir / "verdict_template.md").write_text(_format_verdict_template(), encoding="utf-8")
    (out_dir / "gemini_fallback_command.txt").write_text(
        _format_gemini_fallback_command(out_dir),
        encoding="utf-8",
    )
    (out_dir / "decision.md").write_text(_format_decision(metrics), encoding="utf-8")
    return metrics


def check_judge_verdict(path: Path) -> dict[str, Any]:
    """Validate a judge verdict. Non-ACCEPT or malformed verdicts fail closed."""

    if not path.exists():
        return _verdict_report(
            path=path,
            status="blocked",
            verdict="MISSING",
            errors=[f"verdict file does not exist: {path}"],
        )

    text = path.read_text(encoding="utf-8")
    errors = _schema_errors(text)
    verdict = _extract_verdict(text)
    if errors or verdict != "ACCEPT":
        return _verdict_report(
            path=path,
            status="blocked" if errors else "reject",
            verdict=verdict or "INVALID",
            errors=errors,
        )
    return _verdict_report(path=path, status="accept", verdict="ACCEPT", errors=[])


def _validate_config(config: JudgePacketConfig) -> None:
    if not _SAFE_NAME.fullmatch(config.name):
        raise ValueError(
            "name must start with an alphanumeric character and contain only "
            "letters, numbers, '_', '.', or '-'"
        )
    if config.deliverable not in DELIVERABLES:
        raise ValueError(f"deliverable must be one of {', '.join(DELIVERABLES)}")
    for state in (config.antigravity_cli_state, config.gemini_cli_state):
        if state not in {"auto", "available", "unavailable"}:
            raise ValueError("CLI state must be auto, available, or unavailable")


def _resolve_cli_state(command: str, requested: str) -> str:
    if requested != "auto":
        return requested
    return "available" if shutil.which(command) else "unavailable"


def _judge_contract() -> dict[str, Any]:
    return {
        "verdict": "ACCEPT|REJECT",
        "required_sections": ["Evidence checked", "Findings", "Suggested action"],
        "rules": [
            "Return exactly one Verdict line.",
            "Reject if the packet lacks enough evidence to verify the requested claim.",
            "Reject if any requested validation is missing, stale, or inconsistent.",
            "Reject if the review would require modifying files or using unlisted evidence.",
            "Findings must be concise, evidence-backed, and path-specific when possible.",
        ],
    }


def _format_prompt(config: JudgePacketConfig, manifest: dict[str, Any]) -> str:
    changed_files = _format_bullets(config.changed_files, empty="None listed.")
    evidence = _format_bullets(config.evidence, empty="None listed.")
    commands = _format_bullets(config.validation_commands, empty="None listed.")
    summary = config.summary or "No extra summary supplied."
    manifest_text = json.dumps(manifest, indent=2)
    return "\n".join(
        [
            "# MiniTFT Read-Only Judge Packet",
            "",
            f"Preferred runner: {PREFERRED_JUDGE}",
            f"Preferred model: {PREFERRED_MODEL}",
            f"Thinking: {PREFERRED_THINKING}",
            "",
            "You are an independent judge. Inspect only the provided repo files and evidence.",
            (
                "Do not edit files, run write-capable commands, create tickets, or mutate "
                "external systems."
            ),
            "Fail closed: return `Verdict: REJECT` when evidence is missing or inconsistent.",
            "",
            "## Objective",
            "",
            config.objective,
            "",
            "## Deliverable",
            "",
            config.deliverable,
            "",
            "## Summary",
            "",
            summary,
            "",
            "## Changed Files To Inspect",
            "",
            changed_files,
            "",
            "## Evidence To Check",
            "",
            evidence,
            "",
            "## Validation Commands Reported By Implementer",
            "",
            commands,
            "",
            "## Strict Output Schema",
            "",
            "Return only this Markdown schema:",
            "",
            "```md",
            "Verdict: ACCEPT|REJECT",
            "Evidence checked:",
            "- <file, command, or artifact actually inspected>",
            "Findings:",
            "- <issue, risk, or None.>",
            "Suggested action:",
            "- <next action>",
            "```",
            "",
            "Reject unless the requested behavior is implemented, the evidence supports it, "
            "and no blocking issue remains.",
            "",
            "## Evidence Manifest",
            "",
            "```json",
            manifest_text,
            "```",
            "",
        ]
    )


def _format_verdict_template() -> str:
    return "\n".join(
        [
            "Verdict: REJECT",
            "Evidence checked:",
            "- ",
            "Findings:",
            "- ",
            "Suggested action:",
            "- ",
            "",
        ]
    )


def _format_gemini_fallback_command(out_dir: Path) -> str:
    prompt_path = out_dir / "prompt.md"
    return "\n".join(
        [
            "# Optional read-only fallback when Antigravity is unavailable.",
            (
                "# The Gemini CLI option shape was verified with `gemini --help`; "
                "model access is local."
            ),
            "gemini --approval-mode plan "
            f"--model {GEMINI_FALLBACK_MODEL} "
            "--output-format text "
            '--prompt "Run this read-only MiniTFT judge packet. Return only the strict schema." '
            f"< {prompt_path.as_posix()}",
            "",
        ]
    )


def _format_decision(metrics: dict[str, Any]) -> str:
    out_dir = metrics["out_dir"]
    verdict_path = f"{out_dir}/verdict.md"
    return "\n".join(
        [
            "# Judge Gate Decision",
            "",
            "Status: blocked",
            "",
            "Evidence:",
            f"- Packet: {out_dir}/",
            f"- Deliverable: {metrics['deliverable']}",
            (
                f"- Preferred judge: {PREFERRED_JUDGE} / {PREFERRED_MODEL} / "
                f"thinking {PREFERRED_THINKING}"
            ),
            "- Verdict required: `Verdict: ACCEPT` in the strict schema.",
            "",
            "Limits:",
            "- This packet is read-only evidence for an external/manual judge.",
            "- Missing, malformed, or REJECT verdicts fail closed.",
            "",
            "Next:",
            f"- Save judge output to `{verdict_path}`.",
            (
                "- Run `env -u UV_PYTHON uv run python -m "
                f"mini_tft.tools.judge_packet --check-verdict {verdict_path}`."
            ),
            "",
        ]
    )


def _format_bullets(values: tuple[str, ...], *, empty: str) -> str:
    if not values:
        return f"- {empty}"
    return "\n".join(f"- `{value}`" for value in values)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _schema_errors(text: str) -> list[str]:
    errors: list[str] = []
    lines = [line.rstrip() for line in text.splitlines()]
    nonempty = [line for line in lines if line.strip()]
    verdict_lines = [line for line in nonempty if line.startswith("Verdict:")]
    if len(verdict_lines) != 1:
        errors.append("expected exactly one `Verdict:` line")
    elif verdict_lines[0] not in {"Verdict: ACCEPT", "Verdict: REJECT"}:
        errors.append("verdict must be exactly `Verdict: ACCEPT` or `Verdict: REJECT`")

    for header in _SECTION_HEADERS:
        if header not in nonempty:
            errors.append(f"missing `{header}` section")
        elif not _section_has_body(nonempty, header):
            errors.append(f"`{header}` section must include at least one item")
    return errors


def _section_has_body(lines: list[str], header: str) -> bool:
    start = lines.index(header) + 1
    for line in lines[start:]:
        if line in _SECTION_HEADERS or line.startswith("Verdict:"):
            return False
        if line.startswith("- ") and line.strip() != "-":
            return True
    return False


def _extract_verdict(text: str) -> str | None:
    for line in text.splitlines():
        clean = line.strip()
        if clean == "Verdict: ACCEPT":
            return "ACCEPT"
        if clean == "Verdict: REJECT":
            return "REJECT"
        if clean.startswith("Verdict:"):
            return None
    return None


def _verdict_report(
    *,
    path: Path,
    status: str,
    verdict: str,
    errors: list[str],
) -> dict[str, Any]:
    return {
        "schema": "judge-verdict-check/v1",
        "path": str(path),
        "status": status,
        "verdict": verdict,
        "accepted": status == "accept",
        "fail_closed": status != "accept",
        "errors": errors,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="manual")
    parser.add_argument("--deliverable", choices=DELIVERABLES, default="other")
    parser.add_argument("--out-root", type=Path, default=Path("artifacts/judge"))
    parser.add_argument("--objective", default="Independent read-only judge review.")
    parser.add_argument("--summary", default="")
    parser.add_argument("--changed-file", action="append", default=[])
    parser.add_argument("--evidence", action="append", default=[])
    parser.add_argument("--command", action="append", default=[])
    parser.add_argument(
        "--antigravity-cli-state",
        choices=("auto", "available", "unavailable"),
        default="auto",
    )
    parser.add_argument(
        "--gemini-cli-state",
        choices=("auto", "available", "unavailable"),
        default="auto",
    )
    parser.add_argument("--check-verdict", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.check_verdict is not None:
        report = check_judge_verdict(args.check_verdict)
        print(json.dumps(report, indent=2))
        if not report["accepted"]:
            raise SystemExit(1)
        return

    report = generate_judge_packet(
        JudgePacketConfig(
            name=args.name,
            deliverable=args.deliverable,
            out_root=args.out_root,
            objective=args.objective,
            summary=args.summary,
            changed_files=tuple(args.changed_file),
            evidence=tuple(args.evidence),
            validation_commands=tuple(args.command),
            antigravity_cli_state=args.antigravity_cli_state,
            gemini_cli_state=args.gemini_cli_state,
        )
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    try:
        main()
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
