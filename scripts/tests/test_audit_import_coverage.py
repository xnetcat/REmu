# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


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
        self.assertNotEqual(runtime.statuses["8zTFvBIAIN8"], "named-only")


if __name__ == "__main__":
    unittest.main()
