# Known test failures at the 2026-08-04 stop

This ledger distinguishes passing compile/focused gates from broader tests that did not pass. The
game implementation was not resumed after the owner asked to stop.

## Passing gates

```text
cargo fmt --all -- --check
passed

cargo check --workspace --all-targets
passed, zero warnings

cargo check --workspace --all-targets --target x86_64-apple-darwin
passed, zero warnings

cargo test -p remu-gpu-metal --lib presenter::tests::shader --no-default-features
34 passed, 0 failed

cargo test -p remu-gpu-metal --lib diagnostics --no-default-features
19 passed, 0 failed

cargo test -p remu-memory --lib -- --test-threads=1
90 passed, 0 failed

python3 -m unittest scripts/tests/test_lldb_taskgraph_stall_snapshot.py
15 passed, 0 failed
```

## `remu-gpu-metal` serial suite

Result: 477 passed, 1 failed.

Failing test:

```text
presenter::tests::texture::a_continuously_written_resource_still_completes_content_validation
```

The assertion requires completion within 250 ms. On this host the bounded four full 16 MiB
fingerprint passes took about 2.46 seconds. This is a real gate failure even though the operation
remained bounded.

An earlier parallel graphics-suite attempt was invalid as a health signal because tests interfere
through known process-global state; it produced ordering failures and a SIGSEGV. The serial results
above supersede it.

## `remu-shader` serial suite

Result: 145 passed, 3 failed.

```text
decoder::tests::decodes_scratch_memory_without_treating_offsets_as_guest_pointers
  actual sources: [v0, s5]
  expected:       [s5]

decoder::tests::unknown_deterministically_framed_opcodes_decode_as_raw_instead_of_killing_the_shader
  word 0x7DD80704 decoded as VCmpNleF16 instead of raw

evaluator::tests::buffer::benign_mode_control_and_hwreg_opcodes_do_not_drop_the_shader
  SGetregB32 remains unsupported
```

`remu-gpu` passed 43/43 in the same serial command. The command stopped after the shader failures,
so `remu-shader-metal` did not run in that invocation.

## Runtime serial suite

Command:

```text
cargo test -p remu-runtime -p remu-memory --lib -- --test-threads=1
```

`remu-memory` completed 90/90. `remu-runtime` reported at least these three failures:

```text
agc_driver::tests::a_multi_buffer_submission_runs_every_listed_command_buffer
agc_driver::tests::a_multi_buffer_submission_runs_nothing_when_one_entry_is_invalid
ajm_engine::tests::mp3_decodes_a_stream_and_matches_the_independent_ffmpeg_golden
```

The runtime test binary then stopped producing output and remained alive for more than three
minutes. It was terminated with Ctrl-C (exit 130) because work had been stopped. Therefore there is
no valid final pass/fail count for the 653-test runtime suite, and it must not be reported as
complete.

## Compatibility gates not run after the final NGG edit

No post-change all-game matrix, long Silent Hill title/gameplay proof, or Poppy Playtime retest was
run. Silent Hill's actual title background remained black in the last proven capture; Poppy's only
permitted observation remained black at 0 FPS.
