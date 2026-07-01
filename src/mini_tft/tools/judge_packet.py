"""Generate reusable read-only judge packets for MiniTFT gates."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DELIVERABLES = ("muzero_cache", "puffer_speed", "playable_demo", "docs", "other")
PREFERRED_JUDGE = "Antigravity via ai-router"
PREFERRED_MODEL = "gemini-3.5-flash-low"
PREFERRED_THINKING = "highest"
PREFERRED_REASONING_EFFORT = "xhigh"
AI_ROUTER_DIR = Path("/mnt/ssd2/Projects/ai-router")
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
    ai_router_state: str = "auto"
    ai_router_dir: Path = AI_ROUTER_DIR


def generate_judge_packet(config: JudgePacketConfig) -> dict[str, Any]:
    """Write a reusable judge packet and return its metrics payload."""

    _validate_config(config)
    out_dir = config.out_root / config.name
    out_dir.mkdir(parents=True, exist_ok=True)

    ai_router_state = _resolve_ai_router_state(config.ai_router_dir, config.ai_router_state)
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
        "antigravity_ai_router_command.txt",
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
            "reasoning_effort": PREFERRED_REASONING_EFFORT,
            "read_only": True,
            "fail_closed": True,
            "provider": "ai-router CLIProxyAPI Antigravity OAuth",
            "ai_router": ai_router_state,
            "ai_router_dir": str(config.ai_router_dir),
            "openai_base_url": "http://127.0.0.1:8317/v1",
        },
        "changed_files": list(config.changed_files),
        "evidence": list(config.evidence),
        "validation_commands": list(config.validation_commands),
        "artifacts": artifacts,
        "strict_output_schema": _judge_contract(),
        "known_limits": [
            "Antigravity is reached through local ai-router/CLIProxyAPI OAuth tokens.",
            (
                "The packet only prepares evidence; the gate remains blocked until a strict "
                "ACCEPT verdict."
            ),
            "The generated ai-router command is read-only and writes only verdict.md.",
        ],
    }

    _write_json(out_dir / "evidence_manifest.json", manifest)
    _write_json(out_dir / "metrics.json", metrics)
    (out_dir / "prompt.md").write_text(_format_prompt(config, manifest), encoding="utf-8")
    (out_dir / "verdict_template.md").write_text(_format_verdict_template(), encoding="utf-8")
    (out_dir / "antigravity_ai_router_command.txt").write_text(
        _format_antigravity_ai_router_command(out_dir, config.ai_router_dir),
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
    if config.ai_router_state not in {"auto", "available", "unavailable"}:
        raise ValueError("ai-router state must be auto, available, or unavailable")


def _resolve_ai_router_state(ai_router_dir: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    if (ai_router_dir / "bin" / "ai-env").exists():
        return "available"
    return "unavailable"


def _judge_contract() -> dict[str, Any]:
    return {
        "verdict": "ACCEPT|REJECT",
        "required_sections": ["Evidence checked", "Findings", "Suggested action"],
        "rules": [
            "The first line must be exactly `Verdict: ACCEPT` or `Verdict: REJECT`.",
            "Do not wrap the verdict in a Markdown code fence.",
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
            f"Reasoning effort: {PREFERRED_REASONING_EFFORT}",
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
            "Return only these plain Markdown lines.",
            "Do not wrap the response in a code fence.",
            "Do not add prose before or after the schema.",
            "The first line must be exactly `Verdict: ACCEPT` or `Verdict: REJECT`.",
            "",
            "Verdict: ACCEPT|REJECT",
            "Evidence checked:",
            "- <file, command, or artifact actually inspected>",
            "Findings:",
            "- <issue, risk, or None.>",
            "Suggested action:",
            "- <next action>",
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


def _format_antigravity_ai_router_command(out_dir: Path, ai_router_dir: Path) -> str:
    prompt_path = (out_dir / "prompt.md").resolve()
    verdict_path = (out_dir / "verdict.md").resolve()
    manifest_path = (out_dir / "evidence_manifest.json").resolve()
    repo_dir = Path.cwd().resolve()
    return "\n".join(
        [
            "# Read-only Antigravity judge through local ai-router / CLIProxyAPI.",
            "# Requires ai-router running on http://127.0.0.1:8317/v1 and Antigravity OAuth login.",
            "set -euo pipefail",
            f"cd {shlex.quote(ai_router_dir.as_posix())}",
            'eval "$(bin/ai-env antigravity)"',
            f"export MODEL={PREFERRED_MODEL}",
            f"export REASONING_EFFORT={PREFERRED_REASONING_EFFORT}",
            f"prompt_path={shlex.quote(prompt_path.as_posix())}",
            f"manifest_path={shlex.quote(manifest_path.as_posix())}",
            f"verdict_path={shlex.quote(verdict_path.as_posix())}",
            f"repo_dir={shlex.quote(repo_dir.as_posix())}",
            'python - "$prompt_path" "$manifest_path" "$verdict_path" "$repo_dir" <<\'PY\'',
            "import json",
            "import os",
            "import sys",
            "import urllib.error",
            "import urllib.request",
            "from pathlib import Path",
            "",
            "prompt_path, manifest_path, verdict_path, repo_dir = sys.argv[1:5]",
            "repo_dir = Path(repo_dir)",
            "with open(prompt_path, encoding='utf-8') as file:",
            "    prompt = file.read()",
            "with open(manifest_path, encoding='utf-8') as file:",
            "    manifest = json.load(file)",
            "",
            "def file_section(path_text):",
            "    path = Path(path_text)",
            "    if not path.is_absolute():",
            "        path = repo_dir / path",
            "    try:",
            "        text = path.read_text(encoding='utf-8')",
            "    except UnicodeDecodeError:",
            "        return f'## {path_text}\\n<binary or non-UTF-8 file omitted>\\n'",
            "    except FileNotFoundError:",
            "        return f'## {path_text}\\n<MISSING>\\n'",
            "    max_chars = 60000",
            "    if len(text) > max_chars:",
            "        text = text[:max_chars] + '\\n<TRUNCATED>'",
            "    return f'## {path_text}\\n```\\n{text}\\n```\\n'",
            "",
            "sections = [prompt, '\\n# Local Evidence Contents\\n']",
            "manifest_text = json.dumps(manifest, indent=2)",
            "manifest_section = '## evidence_manifest.json\\n```json\\n' + manifest_text",
            "sections.append(manifest_section + '\\n```\\n')",
            "sections.append('\\n# Changed Files\\n')",
            "for path_text in manifest.get('changed_files', []):",
            "    sections.append(file_section(path_text))",
            "sections.append('\\n# Evidence Files\\n')",
            "for path_text in manifest.get('evidence', []):",
            "    sections.append(file_section(path_text))",
            "full_prompt = '\\n'.join(sections)",
            "base_payload = {",
            f"    'model': os.environ.get('MODEL', {PREFERRED_MODEL!r}),",
            "    'messages': [",
            "        {",
            "            'role': 'system',",
            "            'content': (",
            "                'You are an independent read-only MiniTFT judge. '",
            "                'Use the highest available thinking/reasoning setting. '",
            "                'Return only the requested strict Markdown verdict schema. '",
            "                'Do not use a Markdown code fence. '",
            "                'The first response line must be exactly Verdict: ACCEPT '",
            "                'or Verdict: REJECT.'",
            "            ),",
            "        },",
            "        {'role': 'user', 'content': full_prompt},",
            "    ],",
            "}",
            "base_url = os.environ['OPENAI_BASE_URL'].rstrip('/')",
            "",
            "def call_router(payload):",
            "    request = urllib.request.Request(",
            "        f'{base_url}/chat/completions',",
            "        data=json.dumps(payload).encode('utf-8'),",
            "        headers={",
            "            'Authorization': f\"Bearer {os.environ['OPENAI_API_KEY']}\",",
            "            'Content-Type': 'application/json',",
            "        },",
            "        method='POST',",
            "    )",
            "    with urllib.request.urlopen(request, timeout=300) as response:",
            "        return json.load(response)",
            "",
            "payload = dict(base_payload)",
            "payload['reasoning_effort'] = os.environ.get('REASONING_EFFORT', 'xhigh')",
            "try:",
            "    data = call_router(payload)",
            "except urllib.error.HTTPError as error:",
            "    detail = error.read().decode('utf-8', errors='replace')",
            "    if 'reasoning_effort' not in detail:",
            "        raise",
            "    data = call_router(base_payload)",
            "content = data['choices'][0]['message']['content']",
            "with open(verdict_path, 'w', encoding='utf-8') as file:",
            "    file.write(content.rstrip() + '\\n')",
            "PY",
            f"cd {shlex.quote(repo_dir.as_posix())}",
            (
                "env -u UV_PYTHON uv run python -m mini_tft.tools.judge_packet "
                f"--check-verdict {verdict_path.as_posix()}"
            ),
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
    if not lines:
        errors.append("verdict file is empty")
    elif lines[0] not in {"Verdict: ACCEPT", "Verdict: REJECT"}:
        errors.append(
            "first line must be exactly `Verdict: ACCEPT` or `Verdict: REJECT`"
        )
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
        "--ai-router-state",
        choices=("auto", "available", "unavailable"),
        default="auto",
    )
    parser.add_argument("--ai-router-dir", type=Path, default=AI_ROUTER_DIR)
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
            ai_router_state=args.ai_router_state,
            ai_router_dir=args.ai_router_dir,
        )
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    try:
        main()
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
