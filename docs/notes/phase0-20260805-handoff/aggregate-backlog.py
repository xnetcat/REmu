#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Aggregate Phase 0 SHADER-OPCODES extracts into a frequency backlog."""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

OPCODE_RE = re.compile(
    r"\[REMu\]\[SHADER-OPCODES\] uses=(\d+) shaders=(\d+) (opcode=(\S+)|raw=(\S+) code=(0x[0-9A-Fa-f]+))"
)
HEADER_RE = re.compile(
    r"\[REMu\]\[SHADER-OPCODES\] shaders=(\d+) opcodes=(\d+) raw=(\d+)"
)
FAIL_RE = re.compile(r"Gen5 .* failed:\s*(.+)$")


def main() -> int:
    handoff = Path(__file__).resolve().parent
    titles = [
        "silent-hill",
        "demons-souls",
        "dreaming-sarah",
        "void-terrarium",
        "oregon-trail",
        "poppy",
    ]
    per_title: dict[str, dict[str, int]] = {}
    headers: dict[str, str] = {}
    failures: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    combined_uses: dict[str, int] = defaultdict(int)
    combined_shaders: dict[str, int] = defaultdict(int)
    title_presence: dict[str, set[str]] = defaultdict(set)

    for title in titles:
        path = handoff / f"{title}-shader-extract.txt"
        uses: dict[str, int] = defaultdict(int)
        if not path.exists() or path.stat().st_size == 0:
            headers[title] = "MISSING_OR_EMPTY"
            per_title[title] = {}
            continue
        text = path.read_text(errors="replace")
        for line in text.splitlines():
            if m := HEADER_RE.search(line):
                headers[title] = (
                    f"shaders={m.group(1)} opcodes={m.group(2)} raw={m.group(3)}"
                )
            if m := OPCODE_RE.search(line):
                key = m.group(4) or f"raw:{m.group(5)}:{m.group(6)}"
                count = int(m.group(1))
                shaders = int(m.group(2))
                # Keep the latest snapshot in the extract (largest header usually last).
                uses[key] = count
                combined_uses[key] = max(combined_uses[key], count)
                combined_shaders[key] = max(combined_shaders[key], shaders)
                title_presence[key].add(title)
            if m := FAIL_RE.search(line):
                failures[title][m.group(1).strip()] += 1
        per_title[title] = dict(uses)

    out = handoff / "SHADER-REFUSAL-BACKLOG.md"
    lines: list[str] = []
    lines.append("# Phase 0 shader opcode / refusal backlog")
    lines.append("")
    lines.append("Generated from `REMU_TRACE_SHADER_OPCODES=1` matrix extracts.")
    lines.append("Decoded-opcode frequency is the Phase 0 ranking input before shader-core edits.")
    lines.append("Metal `Gen5 … failed:` lines are listed separately when present.")
    lines.append("")
    lines.append("## Per-title decode coverage")
    lines.append("")
    lines.append("| Title | Header / status | Extract lines |")
    lines.append("| --- | --- | ---: |")
    for title in titles:
        extract = handoff / f"{title}-shader-extract.txt"
        n = sum(1 for _ in extract.open()) if extract.exists() else 0
        lines.append(f"| {title} | {headers.get(title, 'n/a')} | {n} |")
    lines.append("")
    lines.append("## Combined top opcodes (max uses across title snapshots)")
    lines.append("")
    lines.append("| Rank | Opcode | Max uses | Max shaders | Titles |")
    lines.append("| ---: | --- | ---: | ---: | --- |")
    ranked = sorted(combined_uses.items(), key=lambda kv: (-kv[1], kv[0]))
    for i, (opcode, uses) in enumerate(ranked[:80], 1):
        titles_hit = ",".join(sorted(title_presence[opcode]))
        lines.append(
            f"| {i} | `{opcode}` | {uses} | {combined_shaders[opcode]} | {titles_hit} |"
        )
    lines.append("")
    lines.append("## Gen5 / lowering failure lines")
    lines.append("")
    if not any(failures.values()):
        lines.append(
            "No `Gen5 … failed:` lines captured in the short headless windows "
            "(several titles reported `PERF-PIPELINE failed=0`). "
            "Decoded raw opcodes above are the refusal-surrogate backlog for Phase 1 shader work."
        )
    else:
        for title in titles:
            if not failures[title]:
                continue
            lines.append(f"### {title}")
            lines.append("")
            for reason, count in sorted(
                failures[title].items(), key=lambda kv: (-kv[1], kv[0])
            ):
                lines.append(f"- {count}× `{reason}`")
            lines.append("")
    lines.append("## Loud refusals / caveats")
    lines.append("")
    lines.append(
        "See `README.md` in this handoff for launch failures, dirty-tree gates, "
        "and titles that crashed before decode."
    )
    lines.append("")
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out} ({len(ranked)} distinct opcodes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
