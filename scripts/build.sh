#!/usr/bin/env bash
# Configure build/ on first use, then build the given targets (default: all).
# Usage: scripts/build.sh [target...]
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh" >/dev/null

if [[ ! -f "$TC_ROOT/build/build.ninja" ]]; then
  cmake -G Ninja -S "$TC_ROOT" -B "$TC_ROOT/build" \
    -DCMAKE_BUILD_TYPE=Debug \
    -DMLIR_DIR="$MLIR_DIR" \
    -DLLVM_DIR="$LLVM_DIR" \
    -DLLVM_EXTERNAL_LIT="$LLVM_EXTERNAL_LIT"
fi

cmake --build "$TC_ROOT/build" ${1:+--target "$@"}
