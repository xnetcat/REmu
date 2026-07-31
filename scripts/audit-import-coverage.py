#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Build a deterministic static import-coverage inventory for local game dumps.

The audit deliberately separates four facts which runtime first-hit logs conflate:

* an executable imports a NID;
* REmu knows a human-readable name for that NID;
* a REmu dispatch path exists for it; and
* another emulator registers the same NID.

Reference registrations are evidence to inspect, not proof that an implementation is complete.
Game executables are supplied as repeatable ``--game LABEL=/path/to/eboot.bin`` arguments.
Reference snapshot locations default to ``research/upstreams/`` and can be overridden.

Optional pieces never abort the audit: missing SharpEmu/shadPS4/Kyty trees print a warning and
leave reference columns empty, and a failed or skipped ``remu link`` (typical on Windows, where
guest fixed-base mapping is unavailable) falls back to source-table classification only.
"""

from __future__ import annotations

import argparse
import base64
import collections
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHARPEMU = REPO_ROOT / "research/upstreams/SharpEmu"
DEFAULT_SHADPS4 = REPO_ROOT / "research/upstreams/shadPS4"
DEFAULT_KYTY = REPO_ROOT / "research/upstreams/Kyty"
DEFAULT_KYTY_PS5 = REPO_ROOT / "research/upstreams/KytyPS5"
NID_SUFFIX = bytes.fromhex("518d64a635ded8c1e6b039b1c3e55230")
NID_PATTERN = re.compile(r"^[A-Za-z0-9+-]{11}$")
QUOTED_NID_PATTERN = re.compile(r'"([A-Za-z0-9+-]{11})"')
IMPORT_PATTERN = re.compile(
    r"^import=(?P<symbol>\S+) symbol=(?P<index>\d+) weak=(?P<weak>true|false) "
    r"relocations=(?P<count>\d+) types=(?P<types>[0-9,]*)$"
)
PLT_PATTERN = re.compile(
    r"^plt\[\d+\]=.* nid:(?P<nid>[A-Za-z0-9+-]{11}) "
    r"resolved:(?P<name>\S+) name:"
)
LINK_RESOLUTION_PATTERN = re.compile(
    r"^import_resolution=(?P<kind>[a-z-]+) nid=(?P<nid>[A-Za-z0-9+-]{11}) "
    r"weak=(?P<weak>true|false) symbol=(?P<symbol>\S+)$"
)


@dataclass
class ImportedNid:
    nid: str
    symbols: set[str] = field(default_factory=set)
    relocation_types: set[int] = field(default_factory=set)
    relocation_count: int = 0
    weak: bool = True
    remu_name: str | None = None
    link_resolutions: set[str] = field(default_factory=set)

    @property
    def callable(self) -> bool:
        return 7 in self.relocation_types


@dataclass(frozen=True)
class RuntimeCoverage:
    names: dict[str, str]
    statuses: dict[str, str]


@dataclass(frozen=True)
class ReferenceCoverage:
    names: dict[str, str]
    sharpemu: frozenset[str]
    shadps4: frozenset[str]
    kyty: frozenset[str]
    kyty_ps5: frozenset[str]


def compute_nid(export_name: str) -> str:
    digest = hashlib.sha1(export_name.encode("utf-8") + NID_SUFFIX).digest()
    return base64.b64encode(digest[:8][::-1]).decode("ascii").rstrip("=").replace("/", "-")


def parse_game_argument(value: str) -> tuple[str, Path]:
    try:
        label, raw_path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("games must use LABEL=/path/to/eboot.bin") from error
    if not label or not raw_path:
        raise argparse.ArgumentTypeError("games must use a non-empty label and path")
    return label, Path(raw_path)


def function_slice(source: str, name: str) -> str:
    """Return a Rust function through the next top-level function declaration.

    This is intentionally a small source inventory parser, not a Rust parser. REmu keeps the NID
    registries in named functions, and stopping on a column-zero declaration avoids accidentally
    attributing NIDs from tests or later handlers.
    """

    declaration = re.search(
        rf"(?m)^(?:pub(?:\([^)]*\))?\s+)?(?:const\s+)?(?:unsafe\s+)?"
        rf'(?:extern\s+"C"\s+)?fn\s+{re.escape(name)}\s*\(',
        source,
    )
    if declaration is None:
        raise ValueError(f"Rust function not found: {name}")
    following = re.search(
        r"(?m)^((?:pub(?:\([^)]*\))?\s+)?(?:const\s+)?(?:unsafe\s+)?"
        r'(?:extern\s+"C"\s+)?fn\s+[A-Za-z0-9_]+\s*\()',
        source[declaration.end() :],
    )
    end = len(source) if following is None else declaration.end() + following.start()
    return source[declaration.start() : end]


def constant_slice(source: str, name: str) -> str:
    declaration = re.search(rf"(?m)^const\s+{re.escape(name)}\b", source)
    if declaration is None:
        raise ValueError(f"Rust constant not found: {name}")
    # Registry constants are fixed-size arrays, whose type itself contains a semicolon (`[T; N]`).
    # Stop at the array initializer terminator instead of that type-level semicolon.
    end = source.find("];", declaration.end())
    if end < 0:
        raise ValueError(f"unterminated Rust constant: {name}")
    return source[declaration.start() : end + 2]


def quoted_nids(source: str) -> set[str]:
    return set(QUOTED_NID_PATTERN.findall(source))


def parse_known_names(source: str) -> dict[str, str]:
    body = function_slice(source, "known_nid_name")
    return dict(re.findall(r'"([A-Za-z0-9+-]{11})"\s*=>\s*"([^"]+)"', body))


def sibling_function_slice(runtime_path: Path, name: str) -> str:
    """Find a function by following modules declared from a Rust crate root."""

    pending = [runtime_path]
    visited: set[Path] = set()
    root_error: ValueError | None = None
    while pending:
        source_path = pending.pop()
        if source_path in visited:
            continue
        visited.add(source_path)
        source = source_path.read_text(encoding="utf-8")
        try:
            return function_slice(source, name)
        except ValueError as error:
            if root_error is None:
                root_error = error

        module_base = (
            source_path.parent
            if source_path.stem in {"lib", "main", "mod"}
            else source_path.parent / source_path.stem
        )
        modules = re.findall(
            r"(?m)^(?:pub(?:\([^)]*\))?\s+)?mod\s+([A-Za-z0-9_]+)\s*;",
            source,
        )
        for module in modules:
            candidates = (
                module_base / f"{module}.rs",
                module_base / module / "mod.rs",
            )
            pending.extend(candidate for candidate in candidates if candidate.is_file())

    assert root_error is not None
    raise root_error


def parse_flat_stub_arms(dispatch_source: str, return_expression: str) -> set[str]:
    alternatives = r'((?:"[A-Za-z0-9+-]{11}"\s*(?:\|\s*)?)+)'
    matches = re.finditer(
        alternatives + r"\s*=>\s*" + re.escape(return_expression) + r"\s*,",
        dispatch_source,
        re.MULTILINE,
    )
    result: set[str] = set()
    for match in matches:
        result.update(quoted_nids(match.group(1)))
    return result


def area_module_nids(runtime_path: Path) -> set[str]:
    """Returns the NIDs served by the per-area dispatch modules beside the main dispatcher.

    Each area module owns one service family and exposes a `dispatch` the main dispatcher consults
    after its own table misses, so a NID handled there is implemented just as fully as one handled
    inline.  Modules are discovered by reading the dispatch chain rather than from a hard-coded
    list, so adding an area to the runtime does not silently understate coverage here.
    """
    source = runtime_path.read_text(encoding="utf-8")
    try:
        chain = function_slice(source, "dispatch_extended_import")
    except ValueError:
        return set()
    served: set[str] = set()
    for module in re.findall(r"(\w+)::dispatch\(", chain):
        module_path = runtime_path.with_name(f"{module}.rs")
        if not module_path.is_file():
            continue
        module_source = module_path.read_text(encoding="utf-8")
        # Everything outside the test module, so a NID named only by a test is not counted.
        tests = re.search(r"(?m)^#\[cfg\(test\)\]", module_source)
        served |= quoted_nids(module_source[: tests.start()] if tests else module_source)
    return served


def inspect_runtime_source(path: Path) -> RuntimeCoverage:
    source = path.read_text(encoding="utf-8")
    names = dict(
        re.findall(
            r'"([A-Za-z0-9+-]{11})"\s*=>\s*"([^"]+)"',
            sibling_function_slice(path, "known_nid_name"),
        )
    )
    agc_command_source_path = path.with_name("agc_command_abi.rs")
    native_agc_nids: set[str] = set()
    if agc_command_source_path.is_file():
        agc_command_source = agc_command_source_path.read_text(encoding="utf-8")
        native_agc_nids = set().union(
            *(
                quoted_nids(function_slice(agc_command_source, function))
                for function in (
                    "dispatch_indirect_register_patch",
                    "dispatch_common_command",
                    "dispatch_async_command",
                    "dispatch_draw_command",
                    "dispatch_draw_sync_command",
                    "dispatch_packet_patch",
                )
            )
        )
    status_sets: list[tuple[str, set[str]]] = [
        (
            "native-host",
            set().union(
                *(quoted_nids(sibling_function_slice(path, function)) for function in (
                    "native_cxx_symbol_name",
                    "native_unwind_symbol_name",
                    "native_libc_symbol_name",
                ))
            ),
        ),
        (
            "synthetic-data",
            quoted_nids(sibling_function_slice(path, "synthetic_data_imports"))
            | quoted_nids(sibling_function_slice(path, "synthetic_data_import_target")),
        ),
        ("hle-native-agc", native_agc_nids),
        ("hle-area-module", area_module_nids(path)),
    ]

    probe_dispatch = function_slice(source, "dispatch_probe_import")
    stubs = parse_flat_stub_arms(probe_dispatch, "0")
    passthrough_stubs = parse_flat_stub_arms(probe_dispatch, "arguments[0]")
    constant_stubs = parse_flat_stub_arms(probe_dispatch, "1")
    status_sets.extend(
        (
            ("stub-success", stubs),
            ("stub-passthrough", passthrough_stubs),
            ("stub-constant", constant_stubs),
            (
                "hle-lock-aware",
                quoted_nids(function_slice(source, "import_dispatch")),
            ),
            (
                "hle",
                quoted_nids(probe_dispatch)
                | quoted_nids(function_slice(source, "dispatch_stateless_import"))
                | quoted_nids(function_slice(source, "dispatch_kernel_file_import"))
                | quoted_nids(sibling_function_slice(path, "is_guest_sync_import"))
                | quoted_nids(sibling_function_slice(path, "dispatch_guest_sync_import")),
            ),
        )
    )

    # The list above is ordered from strongest/specialized binding to broadest dispatch. Preserve
    # the first classification. Bundled guest-module bindings are now resolved from each process's
    # complete Sony symbol namespace at link time, so no obsolete source-level allowlist is used.
    statuses: dict[str, str] = {}
    for status, nids in status_sets:
        for nid in nids:
            statuses.setdefault(nid, status)
    for nid in names:
        statuses.setdefault(nid, "named-only")
    return RuntimeCoverage(names=names, statuses=statuses)


def _run_remu(command: Sequence[str], subcommand: str, path: Path) -> subprocess.CompletedProcess[str]:
    """Run a remu CLI subcommand with UTF-8 output, including on Windows code pages."""

    return subprocess.run(
        [*command, subcommand, str(path)],
        cwd=REPO_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def run_import_inventory(command: Sequence[str], path: Path) -> str:
    try:
        result = _run_remu(command, "imports", path)
    except FileNotFoundError as error:
        raise RuntimeError(
            f"remu CLI not found ({error}). Build remu-cli or pass --remu-binary "
            f"to a remu executable that can run `imports`."
        ) from error
    if result.returncode != 0:
        raise RuntimeError(
            f"import inventory failed for {path} (status {result.returncode}):\n{result.stderr}"
        )
    return result.stdout


def run_link_inventory(command: Sequence[str], path: Path) -> str | None:
    """Return link inventory stdout, or ``None`` when linking is unavailable.

    Linking materializes the guest image at a fixed virtual base. That fails on hosts that cannot
    reserve that range — Windows in particular, and native arm64 macOS builds — even when
    ``imports`` still works. The audit then continues without per-symbol link resolutions and
    classifies NIDs from the runtime source tables alone.
    """

    try:
        result = _run_remu(command, "link", path)
    except FileNotFoundError as error:
        print(
            f"warning: remu CLI not found for link inventory ({error}); "
            "continuing without link resolutions",
            file=sys.stderr,
        )
        return None
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        print(
            f"warning: link inventory unavailable for {path} "
            f"(status {result.returncode}); continuing without link resolutions"
            + (f":\n{detail}" if detail else ""),
            file=sys.stderr,
        )
        return None
    return result.stdout


def parse_link_resolutions(output: str) -> dict[str, set[str]]:
    resolutions: dict[str, set[str]] = collections.defaultdict(set)
    for line in output.splitlines():
        match = LINK_RESOLUTION_PATTERN.match(line)
        if match is not None:
            resolutions[match.group("symbol")].add(match.group("kind"))
    return dict(resolutions)


def parse_import_inventory(output: str) -> tuple[list[str], dict[str, ImportedNid]]:
    dependencies: list[str] = []
    names: dict[str, str] = {}
    inventory: dict[str, ImportedNid] = {}
    for line in output.splitlines():
        if line.startswith("dependency="):
            dependencies.append(line.removeprefix("dependency="))
            continue
        if match := PLT_PATTERN.match(line):
            if match.group("name") != "unknown":
                names[match.group("nid")] = match.group("name")
            continue
        match = IMPORT_PATTERN.match(line)
        if match is None:
            continue
        symbol = match.group("symbol")
        nid = symbol.partition("#")[0]
        if not NID_PATTERN.fullmatch(nid):
            continue
        imported = inventory.setdefault(nid, ImportedNid(nid=nid))
        imported.symbols.add(symbol)
        imported.relocation_types.update(
            int(value) for value in match.group("types").split(",") if value
        )
        imported.relocation_count += int(match.group("count"))
        imported.weak &= match.group("weak") == "true"
    for nid, name in names.items():
        if nid in inventory:
            inventory[nid].remu_name = name
    return dependencies, inventory


def iter_source_files(root: Path, suffixes: tuple[str, ...]) -> Iterable[Path]:
    if not root.is_dir():
        return ()
    return (
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes and ".git" not in path.parts
    )


def scan_pattern(root: Path, suffixes: tuple[str, ...], pattern: re.Pattern[str]) -> set[str]:
    result: set[str] = set()
    for path in iter_source_files(root, suffixes):
        try:
            source = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        result.update(pattern.findall(source))
    return result


def read_reference_names(sharpemu: Path) -> dict[str, str]:
    names_file = sharpemu / "scripts/ps5_names.txt"
    if not names_file.is_file():
        return {}
    try:
        lines = names_file.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        print(f"warning: could not read reference names at {names_file}: {error}", file=sys.stderr)
        return {}
    result: dict[str, str] = {}
    for raw_name in lines:
        name = raw_name.strip()
        if name:
            result.setdefault(compute_nid(name), name)
    return result


def warn_missing_upstream(label: str, root: Path, required: Path) -> bool:
    """Return True when ``required`` is usable; otherwise warn and return False."""

    if required.is_dir():
        return True
    if root.is_dir():
        try:
            missing = required.relative_to(root)
        except ValueError:
            missing = required
        print(
            f"warning: {label} checkout at {root} is missing {missing}; "
            "reference coverage for it will be empty",
            file=sys.stderr,
        )
    else:
        print(
            f"warning: {label} upstream not found at {root}; "
            "reference coverage for it will be empty",
            file=sys.stderr,
        )
    return False


def inspect_references(
    sharpemu: Path, shadps4: Path, kyty: Path, kyty_ps5: Path
) -> ReferenceCoverage:
    """Scan optional upstream checkouts. Missing trees yield empty sets, not an error."""

    sharp_pattern = re.compile(r'Nid\s*=\s*"([A-Za-z0-9+-]{11})"')
    cpp_pattern = re.compile(
        r'LIB_(?:FUNCTION|FUNC|OBJECT)\s*\(\s*"([A-Za-z0-9+-]{11})"'
    )
    sharpemu_libs = sharpemu / "src/SharpEmu.Libs"
    shadps4_src = shadps4 / "src"
    kyty_src = kyty / "source/emulator"
    kyty_ps5_src = kyty_ps5 / "src"

    warn_missing_upstream("SharpEmu", sharpemu, sharpemu_libs)
    warn_missing_upstream("shadPS4", shadps4, shadps4_src)
    warn_missing_upstream("Kyty", kyty, kyty_src)
    warn_missing_upstream("KytyPS5", kyty_ps5, kyty_ps5_src)

    try:
        return ReferenceCoverage(
            names=read_reference_names(sharpemu),
            sharpemu=frozenset(scan_pattern(sharpemu_libs, (".cs",), sharp_pattern)),
            shadps4=frozenset(
                scan_pattern(shadps4_src, (".c", ".cc", ".cpp", ".h"), cpp_pattern)
            ),
            kyty=frozenset(scan_pattern(kyty_src, (".c", ".cc", ".cpp", ".h"), cpp_pattern)),
            kyty_ps5=frozenset(
                scan_pattern(kyty_ps5_src, (".c", ".cc", ".cpp", ".h"), cpp_pattern)
            ),
        )
    except OSError as error:
        print(
            f"warning: reference scan failed ({error}); continuing without upstream coverage",
            file=sys.stderr,
        )
        return ReferenceCoverage(
            names={},
            sharpemu=frozenset(),
            shadps4=frozenset(),
            kyty=frozenset(),
            kyty_ps5=frozenset(),
        )


def remu_executable_names() -> tuple[str, ...]:
    """Return remu binary basenames for the current host (``.exe`` on Windows)."""

    if os.name == "nt":
        return ("remu.exe", "remu")
    return ("remu",)


def remu_candidate_binaries() -> tuple[Path, ...]:
    """Ordered remu binary search paths for the current host."""

    names = remu_executable_names()
    host_dirs = (
        REPO_ROOT / "target" / "release",
        REPO_ROOT / "target" / "debug",
    )
    # Apple Silicon developers often keep a Rosetta remu that can map the guest base; keep it
    # after the native host dirs so a Windows/Linux checkout never prefers macOS artifacts.
    apple_dirs = (
        REPO_ROOT / "target" / "x86_64-apple-darwin" / "release",
        REPO_ROOT / "target" / "x86_64-apple-darwin" / "debug",
    )
    ordered_dirs = host_dirs if os.name == "nt" else (*host_dirs, *apple_dirs)
    return tuple(directory / name for directory in ordered_dirs for name in names)


def resolve_command(explicit: Path | None) -> list[str]:
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(f"remu binary not found: {explicit}")
        return [str(explicit)]
    for candidate in remu_candidate_binaries():
        if candidate.is_file():
            return [str(candidate)]
    return ["cargo", "run", "--quiet", "-p", "remu-cli", "--"]


def inventory_row(
    title: str,
    imported: ImportedNid,
    runtime: RuntimeCoverage,
    references: ReferenceCoverage,
) -> dict[str, object]:
    nid = imported.nid
    link_kinds = imported.link_resolutions
    if "hle-dispatch" in link_kinds or not link_kinds:
        status = runtime.statuses.get(nid, "unknown")
    elif link_kinds == {"bundled-guest"}:
        status = "guest-module"
    elif link_kinds == {"host-direct"}:
        status = "native-host"
    elif link_kinds == {"unresolved-weak"}:
        status = "unresolved-weak"
    else:
        status = "linked-mixed"
    return {
        "title": title,
        "nid": nid,
        "name": runtime.names.get(nid) or imported.remu_name or references.names.get(nid),
        "status": status,
        "link_resolutions": sorted(link_kinds),
        "callable": imported.callable,
        "weak": imported.weak,
        "symbols": sorted(imported.symbols),
        "relocation_types": sorted(imported.relocation_types),
        "relocation_count": imported.relocation_count,
        "references": [
            name
            for name, values in (
                ("sharpemu", references.sharpemu),
                ("shadps4", references.shadps4),
                ("kyty", references.kyty),
                ("kyty-ps5", references.kyty_ps5),
            )
            if nid in values
        ],
    }


def build_report(
    games: Sequence[tuple[str, Path]],
    command: Sequence[str],
    runtime: RuntimeCoverage,
    references: ReferenceCoverage,
    *,
    skip_link: bool = False,
) -> dict[str, object]:
    titles = []
    all_rows: list[dict[str, object]] = []
    for label, path in games:
        if not path.is_file():
            raise FileNotFoundError(f"game executable not found: {path}")
        dependencies, inventory = parse_import_inventory(run_import_inventory(command, path))
        if skip_link:
            print(
                f"warning: skipping link inventory for {path}; "
                "classifying from runtime source tables only",
                file=sys.stderr,
            )
            link_output = None
        else:
            link_output = run_link_inventory(command, path)
        link_available = link_output is not None
        link_resolutions = parse_link_resolutions(link_output or "")
        for imported in inventory.values():
            for symbol in imported.symbols:
                imported.link_resolutions.update(link_resolutions.get(symbol, ()))
        rows = [
            inventory_row(label, imported, runtime, references)
            for imported in sorted(inventory.values(), key=lambda value: value.nid)
        ]
        all_rows.extend(rows)
        counts = collections.Counter(str(row["status"]) for row in rows)
        callable_rows = [row for row in rows if row["callable"]]
        callable_counts = collections.Counter(str(row["status"]) for row in callable_rows)
        unknown_rows = [row for row in callable_rows if row["status"] == "unknown"]
        titles.append(
            {
                "title": label,
                "dependencies": sorted(dependencies),
                "unique_nids": len(rows),
                "callable_nids": len(callable_rows),
                "link_inventory": link_available,
                "status_counts": dict(sorted(counts.items())),
                "callable_status_counts": dict(sorted(callable_counts.items())),
                "callable_unknown_named": sum(bool(row["name"]) for row in unknown_rows),
                "callable_unknown_in_reference": sum(
                    bool(row["references"]) for row in unknown_rows
                ),
            }
        )
    return {
        "schema_version": 1,
        "titles": titles,
        "imports": all_rows,
    }


def print_text_report(report: dict[str, object], details: bool) -> None:
    titles = report["titles"]
    assert isinstance(titles, list)
    for title in titles:
        assert isinstance(title, dict)
        link_note = (
            "link_inventory=yes" if title.get("link_inventory", True) else "link_inventory=no"
        )
        print(
            f"title={title['title']} dependencies={len(title['dependencies'])} "
            f"unique_nids={title['unique_nids']} callable_nids={title['callable_nids']} "
            f"{link_note}"
        )
        counts = title["callable_status_counts"]
        assert isinstance(counts, dict)
        print("callable_status=" + ",".join(f"{key}:{counts[key]}" for key in sorted(counts)))
        print(
            f"callable_unknown_named={title['callable_unknown_named']} "
            f"callable_unknown_in_reference={title['callable_unknown_in_reference']}"
        )

    imports = report["imports"]
    assert isinstance(imports, list)
    gaps = [
        row
        for row in imports
        if isinstance(row, dict)
        and row["callable"]
        and row["status"] in {"unknown", "named-only", "stub-success", "stub-passthrough", "stub-constant"}
    ]
    gaps.sort(
        key=lambda row: (
            {"stub-success": 0, "stub-passthrough": 0, "stub-constant": 0, "named-only": 1, "unknown": 2}.get(str(row["status"]), 3),
            str(row["title"]),
            str(row["name"] or "~"),
            str(row["nid"]),
        )
    )
    print(f"priority_callable_gaps={len(gaps)}")
    if details:
        for row in gaps:
            references = ",".join(row["references"]) if row["references"] else "-"
            print(
                f"gap title={row['title']} status={row['status']} nid={row['nid']} "
                f"name={row['name'] or 'unknown'} references={references}"
            )


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--game",
        action="append",
        type=parse_game_argument,
        help="title and executable as LABEL=/path/to/eboot.bin; repeatable and required",
    )
    parser.add_argument("--remu-binary", type=Path, help="existing remu CLI binary")
    parser.add_argument(
        "--runtime-source",
        type=Path,
        default=REPO_ROOT / "crates/remu-runtime/src/lib.rs",
    )
    parser.add_argument("--sharpemu", type=Path, default=DEFAULT_SHARPEMU)
    parser.add_argument("--shadps4", type=Path, default=DEFAULT_SHADPS4)
    parser.add_argument("--kyty", type=Path, default=DEFAULT_KYTY)
    parser.add_argument("--kyty-ps5", type=Path, default=DEFAULT_KYTY_PS5)
    link_group = parser.add_mutually_exclusive_group()
    link_group.add_argument(
        "--skip-link",
        action="store_true",
        default=None,
        help="do not run `remu link`; classify from runtime source tables only",
    )
    link_group.add_argument(
        "--link",
        action="store_true",
        default=None,
        help="attempt `remu link` even on hosts where guest mapping usually fails",
    )
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--details", action="store_true", help="print every callable gap")
    parser.add_argument("--output", type=Path, help="write the report instead of stdout")
    return parser


def should_skip_link(skip_link: bool | None, force_link: bool | None) -> bool:
    """Windows defaults to skipping link; elsewhere link is attempted unless opted out."""

    if force_link:
        return False
    if skip_link:
        return True
    return os.name == "nt"


def main(argv: Sequence[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    games = args.game
    if not games:
        print(
            "error: at least one --game LABEL=/path/to/eboot.bin argument is required",
            file=sys.stderr,
        )
        return 1
    try:
        runtime = inspect_runtime_source(args.runtime_source)
        references = inspect_references(args.sharpemu, args.shadps4, args.kyty, args.kyty_ps5)
        report = build_report(
            games,
            resolve_command(args.remu_binary),
            runtime,
            references,
            skip_link=should_skip_link(args.skip_link, args.link),
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    if args.format == "json":
        rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    else:
        from io import StringIO
        from contextlib import redirect_stdout

        output = StringIO()
        with redirect_stdout(output):
            print_text_report(report, args.details)
        rendered = output.getvalue()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
