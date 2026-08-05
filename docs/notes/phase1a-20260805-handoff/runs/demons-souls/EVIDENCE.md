Demon's Souls evidence (revision d3fc2f9, clean tree).
run.json was not present in the artifact dir after the guest SIGSEGV (exit 139); host wrapper recorded:
  vm.loadavg at launch: { 3.49 3.31 2.69 }
  artifacts: .../20260805T204317Z-PPSA01342-app0
  outcome: guest-failure / exit_status=139 (see summary.txt)
Primary signals: rejected unlock slot=Some(0); snd_osfuncs.cpp:410; sceAcmBatchStartBuffers unimplemented.
