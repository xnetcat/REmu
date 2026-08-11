# AGC Context State Operation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve and execute all four SDK 10 `ContextStateOperation` values in command-stream order, including reset-queue recovery of a pushed Cx state.

**Architecture:** The HLE constructor emits a valid private NOP packet carrying a magic word and operation, so packet copying and indirect-buffer execution preserve the operation without guessing Sony's native multipacket encoding. The submission parser recognizes that packet in normal predication/COND_EXEC order and applies it to a one-deep Cx snapshot stored beside the existing graphics register shadow.

**Tech Stack:** Rust, PM4/AGC command parsing, `cargo test -p remu-runtime`.

## Global Constraints

- Use test-driven development: observe each new behavioral test fail before production implementation.
- Do not launch the game.
- Do not modify `pthread.rs`, `platform_independent_tests/imports.rs`, or `platform_independent_tests/mod.rs`.
- Preserve ordinary type-2 and NOP behavior.
- Commit the coherent implementation only after focused and package-wide verification passes.

---

### Task 1: Command ABI encoding and size

**Files:**
- Modify: `crates/remu-runtime/src/agc_command_abi/tests.rs`
- Modify: `crates/remu-runtime/src/agc_command_abi/common.rs`
- Modify: `crates/remu-runtime/src/agc_command_abi.rs`

**Interfaces:**
- Produces: `context_state_op([u64; 6]) -> u64`, a three-DWORD private `IT_NOP` packet with selector `R_CONTEXT_STATE_OP`, magic, and operation.
- Produces: `context_state_op_size([u64; 6]) -> u64`, returning 12 for operations 0 through 3 and 0 otherwise.

- [x] **Step 1: Write failing ABI tests**

Add table-driven literal assertions that operations 0, 1, 2, and 3 produce distinct `[pm4(3, IT_NOP, R_CONTEXT_STATE_OP), CONTEXT_STATE_OP_MAGIC, operation]` packets and advance the cursor by 12 bytes. Add invalid-operation assertions that construction and get-size return zero without moving the cursor.

- [x] **Step 2: Verify the ABI tests fail**

Run: `cargo test -p remu-runtime agc_command_abi::tests::builds_context_state_operation_packets -- --exact`

Expected: failure because the current constructor emits the one-DWORD literal `0x80000000` and get-size is not dispatched.

- [x] **Step 3: Implement the minimal constructor and size dispatch**

Validate `arguments[1] <= 3`, reserve three DWORDs, write the private NOP packet, and route NIDs `qj7QZpgr9Uw` and `H6vHS5cidSA` to constructor and size functions.

- [x] **Step 4: Verify the ABI tests pass**

Run the same focused command and the invalid-operation test by exact name.

### Task 2: Ordered Cx state semantics

**Files:**
- Modify: `crates/remu-runtime/src/platform_independent_tests/agc_packets.rs`
- Modify: `crates/remu-runtime/src/lib.rs`
- Modify: `crates/remu-runtime/src/agc_submission/parse.rs`
- Modify: `crates/remu-runtime/src/agc_submission.rs`
- Modify: `crates/remu-runtime/src/agc_registers.rs`

**Interfaces:**
- Produces: `AgcContextStateSnapshot`, containing the Cx register map and blend provenance.
- Produces: `AgcGraphicsState::apply_context_state_op(op) -> Result<(), AgcSubmissionError>`.

- [x] **Step 1: Write failing ordered parser tests**

Exercise real command streams containing direct Cx writes and private context markers. Assert Clear reseeds SDK defaults, Push/Pop restores a custom register and blend provenance, PushClear exposes defaults before Pop, and packet order changes the visible state.

- [x] **Step 2: Verify the parser tests fail**

Run each new test with `cargo test -p remu-runtime platform_independent_tests::agc_packets::<name> -- --exact`.

Expected: failure because the private marker is currently an inert NOP and no Cx snapshot exists.

- [x] **Step 3: Implement state and ordered parsing**

Recognize the exact NOP selector, length, and magic after COND_EXEC/predication selection; dispatch operations 0 through 3. Snapshot only `context_registers` and `blend_register_provenance`, keep register-write sequencing monotonic, and increment the Cx generation on clear or restore so cached snapshots cannot remain stale.

- [x] **Step 4: Verify ordered parser tests pass**

Run the new tests and the existing CLEAR_STATE tests.

### Task 3: Stack errors and reset bit 10

**Files:**
- Modify: `crates/remu-runtime/src/platform_independent_tests/agc_packets.rs`
- Modify: `crates/remu-runtime/src/lib.rs`
- Modify: `crates/remu-runtime/src/agc_registers.rs`

**Interfaces:**
- Consumes: the one-deep snapshot from Task 2.
- Produces: explicit double-push and Pop-underflow submission errors.
- Produces: `ResetQueueOp::CONTEXT_OP_PUSH` restoration before optional Cx clearing.

- [x] **Step 1: Write failing error/reset tests**

Assert a second Push/PushClear fails, Pop without a snapshot fails, reset bit 10 restores a pushed value, and bit 10 combined with `CX_STATE` finishes at SDK defaults.

- [x] **Step 2: Verify the error/reset tests fail**

Run the four tests by exact name and confirm the missing or wrong behavior causes each failure.

- [x] **Step 3: Implement minimal stack validation and reset ordering**

Add submission error variants. For reset bit 10, restore and consume an occupied snapshot without treating an empty stack as an error, then process `CX_STATE`; this makes `CONTEXT_OP_PUSH | CX_STATE` end at defaults.

- [x] **Step 4: Verify the error/reset tests pass**

Run the four focused tests again.

### Task 4: Regression and final verification

**Files:**
- Modify only files listed above.

**Interfaces:**
- Verifies all interfaces from Tasks 1 through 3.

- [x] **Step 1: Add parser-boundary regressions**

Assert ordinary `0x80000000` remains inert and a context marker in a nested indirect buffer executes in stream order.

- [x] **Step 2: Observe regression tests fail before any required production adjustment**

Run both tests by exact name. If the existing implementation already satisfies one, retain it only when it protects a distinct boundary; otherwise remove the redundant test.

- [x] **Step 3: Make the minimal production adjustment**

Ensure recursive command parsing routes the private marker through the same handler and leaves ordinary type-2 packets untouched.

- [x] **Step 4: Run formatting and full verification**

Run:

```sh
cargo fmt --all -- --check
cargo test -p remu-runtime agc_command_abi::tests
cargo test -p remu-runtime platform_independent_tests::agc_packets
cargo test -p remu-runtime
```

- [x] **Step 5: Review and commit**

Inspect `git diff --check`, `git diff`, and `git status --short`; stage only this plan and the AGC implementation/test files, then commit with `fix: preserve AGC context state operations`.
