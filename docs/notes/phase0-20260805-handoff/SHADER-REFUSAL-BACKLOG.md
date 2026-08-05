# Phase 0 shader opcode / refusal backlog

Generated from `REMU_TRACE_SHADER_OPCODES=1` matrix extracts.
Decoded-opcode frequency is the Phase 0 ranking input before shader-core edits.
Metal `Gen5 … failed:` lines are listed separately when present.

## Per-title decode coverage

| Title | Header / status | Extract lines |
| --- | --- | ---: |
| silent-hill | shaders=65 opcodes=197 raw=0 | 199 |
| demons-souls | MISSING_OR_EMPTY | 0 |
| dreaming-sarah | MISSING_OR_EMPTY | 0 |
| void-terrarium | MISSING_OR_EMPTY | 0 |
| oregon-trail | shaders=6 opcodes=32 raw=0 | 34 |
| poppy | shaders=2 opcodes=16 raw=0 | 18 |

## Combined top opcodes (max uses across title snapshots)

| Rank | Opcode | Max uses | Max shaders | Titles |
| ---: | --- | ---: | ---: | --- |
| 1 | `VMulF32` | 1756 | 43 | silent-hill |
| 2 | `VMacF32` | 1431 | 35 | silent-hill |
| 3 | `VMadF32` | 1330 | 48 | oregon-trail,silent-hill |
| 4 | `SWaitcnt` | 772 | 63 | oregon-trail,poppy,silent-hill |
| 5 | `VSubF32` | 602 | 21 | silent-hill |
| 6 | `VInterpP1F32` | 600 | 25 | silent-hill |
| 7 | `VInterpP2F32` | 600 | 25 | silent-hill |
| 8 | `VAddF32` | 505 | 20 | silent-hill |
| 9 | `VMinF32` | 479 | 23 | silent-hill |
| 10 | `VMaxF32` | 416 | 23 | silent-hill |
| 11 | `ImageSampleLz` | 337 | 27 | silent-hill |
| 12 | `SMovB32` | 332 | 47 | silent-hill |
| 13 | `VCndmaskB32` | 203 | 26 | silent-hill |
| 14 | `VMadMkF32` | 198 | 8 | silent-hill |
| 15 | `VMovB32` | 172 | 36 | oregon-trail,silent-hill |
| 16 | `VRcpF32` | 153 | 12 | silent-hill |
| 17 | `SCbranchExecz` | 149 | 20 | oregon-trail,poppy,silent-hill |
| 18 | `SMovB64` | 144 | 19 | silent-hill |
| 19 | `Exp` | 142 | 46 | oregon-trail,silent-hill |
| 20 | `SBfeU32` | 119 | 15 | oregon-trail,silent-hill |
| 21 | `VExpF32` | 116 | 6 | silent-hill |
| 22 | `VMadAkF32` | 111 | 7 | silent-hill |
| 23 | `VLogF32` | 104 | 7 | silent-hill |
| 24 | `SLoadDwordx4` | 100 | 26 | silent-hill |
| 25 | `SBufferLoadDwordx16` | 94 | 32 | silent-hill |
| 26 | `VAddI32` | 87 | 10 | oregon-trail,poppy,silent-hill |
| 27 | `VLshlrevB32` | 81 | 9 | silent-hill |
| 28 | `SLshlB32` | 75 | 21 | oregon-trail,silent-hill |
| 29 | `TBufferLoadFormatX` | 72 | 1 | silent-hill |
| 30 | `SBufferLoadDwordx4` | 71 | 43 | oregon-trail,poppy,silent-hill |
| 31 | `SAndB32` | 69 | 17 | oregon-trail,silent-hill |
| 32 | `SOrB32` | 67 | 16 | oregon-trail,silent-hill |
| 33 | `VCvtPkrtzF16F32` | 66 | 30 | silent-hill |
| 34 | `SEndpgm` | 64 | 64 | oregon-trail,poppy,silent-hill |
| 35 | `VLshlAddU32` | 64 | 18 | oregon-trail,poppy,silent-hill |
| 36 | `SNop` | 60 | 16 | oregon-trail,silent-hill |
| 37 | `SInstPrefetch` | 59 | 59 | oregon-trail,poppy,silent-hill |
| 38 | `VCmpGtF32` | 57 | 8 | silent-hill |
| 39 | `SAndB64` | 56 | 10 | silent-hill |
| 40 | `SCmpEqU32` | 56 | 16 | silent-hill |
| 41 | `SBufferLoadDword` | 55 | 21 | oregon-trail,poppy,silent-hill |
| 42 | `SCselectB32` | 50 | 15 | silent-hill |
| 43 | `DsWrite2B32` | 49 | 4 | silent-hill |
| 44 | `VCmpLtF32` | 45 | 8 | silent-hill |
| 45 | `SBufferLoadDwordx8` | 41 | 34 | silent-hill |
| 46 | `VFmaMixloF16` | 41 | 2 | silent-hill |
| 47 | `SCselectB64` | 40 | 19 | silent-hill |
| 48 | `VFmaMixF32` | 39 | 2 | silent-hill |
| 49 | `ImageLoad` | 36 | 8 | silent-hill |
| 50 | `SLoadDwordx8` | 35 | 12 | silent-hill |
| 51 | `SMovkI32` | 35 | 7 | silent-hill |
| 52 | `VCvtF32U32` | 35 | 13 | oregon-trail,silent-hill |
| 53 | `SBufferLoadDwordx2` | 34 | 22 | oregon-trail,poppy,silent-hill |
| 54 | `VCmpxLtF32` | 33 | 6 | silent-hill |
| 55 | `VCmpxGtU32` | 32 | 12 | oregon-trail,poppy,silent-hill |
| 56 | `SSubI32` | 31 | 16 | oregon-trail,silent-hill |
| 57 | `SCmpEqI32` | 30 | 15 | silent-hill |
| 58 | `VCvtI32F32` | 28 | 2 | silent-hill |
| 59 | `VFmaMixhiF16` | 28 | 2 | silent-hill |
| 60 | `SBfeU64` | 27 | 13 | silent-hill |
| 61 | `SLshl4AddU32` | 26 | 13 | silent-hill |
| 62 | `SLshrB64` | 26 | 13 | oregon-trail,silent-hill |
| 63 | `VTruncF32` | 26 | 1 | silent-hill |
| 64 | `ImageSample` | 24 | 6 | silent-hill |
| 65 | `SAddU32` | 24 | 1 | silent-hill |
| 66 | `SAddcU32` | 24 | 1 | silent-hill |
| 67 | `SBranch` | 24 | 7 | silent-hill |
| 68 | `SGetpcB64` | 24 | 1 | silent-hill |
| 69 | `VLshrrevB32` | 23 | 6 | oregon-trail,silent-hill |
| 70 | `VMadU32U24` | 23 | 2 | silent-hill |
| 71 | `VMin3F32` | 23 | 6 | silent-hill |
| 72 | `DsRead2B32` | 22 | 5 | silent-hill |
| 73 | `DsReadB128` | 22 | 2 | silent-hill |
| 74 | `DsReadB32` | 21 | 5 | silent-hill |
| 75 | `DsWriteB32` | 21 | 5 | silent-hill |
| 76 | `SAndSaveexecB64` | 21 | 5 | silent-hill |
| 77 | `VCmpLeF32` | 21 | 4 | silent-hill |
| 78 | `SAddI32` | 20 | 12 | silent-hill |
| 79 | `ImageStore` | 19 | 11 | oregon-trail,poppy,silent-hill |
| 80 | `VAndB32` | 19 | 10 | oregon-trail,silent-hill |

## Gen5 / lowering failure lines

No `Gen5 … failed:` lines captured in the short headless windows (several titles reported
`PERF-PIPELINE failed=0`). Decoded-opcode frequency above is the refusal-surrogate backlog for
Phase 1 shader work — not a Metal compile-failure ranking.

## Loud refusals / caveats

- **Demon's Souls / Dreaming Sarah / Void Terrarium:** crash before AGC decode → empty extracts.
  Do not invent opcode frequencies for them.
- **Dreaming Sarah:** missing `/app0/shaders/water.vert` (and related); SIGSEGV ~1s. No FPS claim;
  no `bf50225` control run this session.
- **Oregon Trail:** must launch `…/decrypted/eboot.bin`; root eboot duplicates `libc.prx`.
- **raw=0** across successful titles in this window — backlog is known opcodes by use count.
- Full caveat list: `README.md`.

