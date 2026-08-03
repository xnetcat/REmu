# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import importlib.util
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "audit-import-coverage.py"
SPEC = importlib.util.spec_from_file_location("audit_import_coverage", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


class ImportCoverageAuditTests(unittest.TestCase):
    def test_nid_hash_matches_known_ps5_exports(self) -> None:
        self.assertEqual(AUDIT.compute_nid("scePthreadYield"), "T72hz6ffq08")
        self.assertEqual(AUDIT.compute_nid("sceKernelMprotect"), "vSMAm3cxYTY")

    def test_import_inventory_aggregates_modules_and_relocation_kinds(self) -> None:
        output = """\
dependency=libkernel.prx
plt[0]=offset:0x8 symbol:1 nid:T72hz6ffq08 resolved:scePthreadYield name:T72hz6ffq08#A#B
import=T72hz6ffq08#A#B symbol=1 weak=false relocations=2 types=1,7
import=T72hz6ffq08#C#D symbol=2 weak=true relocations=1 types=7
import=P330P3dFF68#A#B symbol=3 weak=false relocations=1 types=1
"""
        dependencies, imports = AUDIT.parse_import_inventory(output)
        self.assertEqual(dependencies, ["libkernel.prx"])
        self.assertEqual(imports["T72hz6ffq08"].relocation_count, 3)
        self.assertEqual(imports["T72hz6ffq08"].relocation_types, {1, 7})
        self.assertFalse(imports["T72hz6ffq08"].weak)
        self.assertTrue(imports["T72hz6ffq08"].callable)
        self.assertFalse(imports["P330P3dFF68"].callable)

    def test_link_inventory_preserves_exact_scoped_symbol_resolution(self) -> None:
        output = """\
import_resolution=bundled-guest nid=T72hz6ffq08 weak=false symbol=T72hz6ffq08#A#B
import_resolution=hle-dispatch nid=T72hz6ffq08 weak=false symbol=T72hz6ffq08#C#D
import_resolution=host-direct nid=P330P3dFF68 weak=false symbol=P330P3dFF68#A#B
"""
        resolutions = AUDIT.parse_link_resolutions(output)
        self.assertEqual(resolutions["T72hz6ffq08#A#B"], {"bundled-guest"})
        self.assertEqual(resolutions["T72hz6ffq08#C#D"], {"hle-dispatch"})
        self.assertEqual(resolutions["P330P3dFF68#A#B"], {"host-direct"})

    def test_runtime_source_distinguishes_handlers_from_names_and_stubs(self) -> None:
        runtime = AUDIT.inspect_runtime_source(
            Path(__file__).resolve().parents[2] / "crates/remu-runtime/src/lib.rs"
        )
        self.assertEqual(runtime.statuses["T72hz6ffq08"], "hle-lock-aware")
        self.assertEqual(runtime.statuses["kALvdgEv5ME"], "stub-success")
        self.assertEqual(runtime.statuses["hqi8yMOCmG0"], "stub-passthrough")
        self.assertEqual(runtime.statuses["qj7QZpgr9Uw"], "hle-native-agc")
        self.assertEqual(runtime.statuses["vcmNN+AAXnY"], "hle-native-agc")
        self.assertNotEqual(runtime.statuses["8zTFvBIAIN8"], "named-only")

    def test_stateless_dispatch_chain_modules_are_counted_as_served(self) -> None:
        # The fiber family demonstrably runs (docs/status.md) but is dispatched from
        # dispatch_stateless_import, a chain the auditor previously did not walk — its NIDs
        # showed as unserved, making the audit an upper bound instead of a work list.
        runtime = AUDIT.inspect_runtime_source(
            Path(__file__).resolve().parents[2] / "crates/remu-runtime/src/lib.rs"
        )
        self.assertEqual(runtime.statuses.get("a0LLrZWac0M"), "hle-area-module")
        self.assertEqual(runtime.statuses.get("BohYr-F7-is"), "hle-area-module")

    def test_parse_known_names_follows_nid_table_module(self) -> None:
        names = AUDIT.parse_known_names(
            Path(__file__).resolve().parents[2] / "crates/remu-runtime/src/lib.rs"
        )
        self.assertEqual(names["T72hz6ffq08"], "scePthreadYield")
        self.assertEqual(names["vSMAm3cxYTY"], "sceKernelMprotect")

    def test_missing_upstream_trees_yield_empty_reference_coverage(self) -> None:
        missing = Path("/definitely/missing/remu-upstream")
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            references = AUDIT.inspect_references(missing, missing, missing, missing)
        self.assertEqual(references.names, {})
        self.assertEqual(references.sharpemu, frozenset())
        self.assertEqual(references.shadps4, frozenset())
        self.assertEqual(references.kyty, frozenset())
        self.assertEqual(references.kyty_ps5, frozenset())
        self.assertIn("SharpEmu upstream not found", stderr.getvalue())
        self.assertIn("shadPS4 upstream not found", stderr.getvalue())

    def test_failed_link_inventory_is_soft_and_optional(self) -> None:
        inventory = """\
dependency=libkernel.prx
plt[0]=offset:0x8 symbol:1 nid:T72hz6ffq08 resolved:scePthreadYield name:T72hz6ffq08#A#B
import=T72hz6ffq08#A#B symbol=1 weak=false relocations=1 types=7
"""
        runtime = AUDIT.RuntimeCoverage(
            names={"T72hz6ffq08": "scePthreadYield"},
            statuses={"T72hz6ffq08": "hle-lock-aware"},
        )
        references = AUDIT.ReferenceCoverage(
            names={},
            sharpemu=frozenset(),
            shadps4=frozenset(),
            kyty=frozenset(),
            kyty_ps5=frozenset(),
        )
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as handle:
            game = Path(handle.name)
            handle.write(b"EB")
        try:
            with mock.patch.object(AUDIT, "run_import_inventory", return_value=inventory), mock.patch.object(
                AUDIT, "run_link_inventory", return_value=None
            ):
                report = AUDIT.build_report(
                    [("demo", game)],
                    ["remu"],
                    runtime,
                    references,
                    skip_link=False,
                )
            self.assertFalse(report["titles"][0]["link_inventory"])
            self.assertEqual(report["imports"][0]["status"], "hle-lock-aware")

            with mock.patch.object(AUDIT, "run_import_inventory", return_value=inventory) as imports_mock, mock.patch.object(
                AUDIT, "run_link_inventory"
            ) as link_mock:
                report = AUDIT.build_report(
                    [("demo", game)],
                    ["remu"],
                    runtime,
                    references,
                    skip_link=True,
                )
            imports_mock.assert_called_once()
            link_mock.assert_not_called()
            self.assertFalse(report["titles"][0]["link_inventory"])
        finally:
            game.unlink(missing_ok=True)

    def test_windows_defaults_to_skipping_link(self) -> None:
        with mock.patch.object(AUDIT.os, "name", "nt"):
            self.assertTrue(AUDIT.should_skip_link(None, None))
            self.assertFalse(AUDIT.should_skip_link(None, True))
            self.assertTrue(AUDIT.should_skip_link(True, None))
        with mock.patch.object(AUDIT.os, "name", "posix"):
            self.assertFalse(AUDIT.should_skip_link(None, None))
            self.assertTrue(AUDIT.should_skip_link(True, None))
            self.assertFalse(AUDIT.should_skip_link(None, True))

    def test_windows_remu_candidates_prefer_exe_over_apple_triples(self) -> None:
        with mock.patch.object(AUDIT.os, "name", "nt"):
            names = AUDIT.remu_executable_names()
            candidates = AUDIT.remu_candidate_binaries()
        self.assertEqual(names[0], "remu.exe")
        self.assertTrue(all("apple-darwin" not in str(path) for path in candidates))
        self.assertTrue(any(path.name == "remu.exe" for path in candidates))


if __name__ == "__main__":
    unittest.main()
