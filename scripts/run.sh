#!/usr/bin/env bash
# Build tc-compile if needed and run a convenient compiler demo.
#   run.sh               walk models/linear.onnx through every stage
#   run.sh model.onnx    JIT-run one model
#   run.sh args...       pass arguments straight to tc-compile
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
"$SCRIPT_DIR/build.sh" tc-compile
TC_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
TC_COMPILE="$TC_ROOT/build/bin/tc-compile"

if [[ "$#" -eq 0 ]]; then
  MODEL="$TC_ROOT/models/linear.onnx"

  printf '\n=== Parsed ONNX graph ===\n'
  "$TC_COMPILE" "$MODEL" -emit=proto

  printf '\n=== Custom tc dialect ===\n'
  "$TC_COMPILE" "$MODEL" -emit=mlir

  printf '\n=== Linalg on tensors ===\n'
  "$TC_COMPILE" "$MODEL" -emit=linalg

  printf '\n=== CPU JIT result ===\n'
  "$TC_COMPILE" "$MODEL" -emit=jit
elif [[ "$#" -eq 1 ]]; then
  "$TC_COMPILE" "$1" -emit=jit
else
  exec "$TC_COMPILE" "$@"
fi
