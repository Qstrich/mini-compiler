#!/usr/bin/env bash
# Build tc-compile if needed and run a convenient compiler demo.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"

if [[ ! -d "$TC_ROOT/build" ]]; then
  cmake -G Ninja -S "$TC_ROOT" -B "$TC_ROOT/build" \
    -DCMAKE_BUILD_TYPE=Debug \
    -DMLIR_DIR="$MLIR_DIR" \
    -DLLVM_DIR="$LLVM_DIR" \
    -DLLVM_EXTERNAL_LIT="$LLVM_EXTERNAL_LIT"
fi

cmake --build "$TC_ROOT/build" --target tc-compile
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
