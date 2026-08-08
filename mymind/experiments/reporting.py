from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def metadata(config: Dict[str, Any], model: str) -> Dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        commit = "unknown"
    try:
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
        ).strip())
    except Exception:
        dirty = None
    source_hasher = hashlib.sha256()
    root = Path.cwd()
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts or "artifacts" in path.parts:
            continue
        source_hasher.update(path.relative_to(root).as_posix().encode("utf-8"))
        source_hasher.update(path.read_bytes())
    encoded = json.dumps(config, ensure_ascii=True, sort_keys=True).encode("utf-8")
    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "git_dirty": dirty,
        "source_hash": source_hasher.hexdigest(),
        "config_hash": hashlib.sha256(encoded).hexdigest(),
        "model": model,
    }


def write_report(report: Dict[str, Any], output_dir: Path, stem: str) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    json_path = output_dir / f"{stem}-{timestamp}.json"
    markdown_path = output_dir / f"{stem}-{timestamp}.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [f"# {report['title']}", "", f"- captured_at: {report['metadata']['captured_at']}"]
    lines.append(f"- git_commit: {report['metadata']['git_commit']}")
    lines.append(f"- git_dirty: {report['metadata']['git_dirty']}")
    lines.append(f"- source_hash: {report['metadata']['source_hash']}")
    lines.append(f"- overall_passed: {report.get('overall_passed', False)}")
    lines.extend(["", "## Variants", ""])
    for name, values in report.get("variants", {}).items():
        lines.append(f"### {name}")
        for key, value in values.items():
            lines.append(f"- {key}: {value}")
        lines.append("")
    failures = report.get("failures", [])
    lines.extend(["## Failures", ""])
    lines.extend([f"- {item}" for item in failures] or ["- none"])
    markdown_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return {"json": str(json_path), "markdown": str(markdown_path)}
