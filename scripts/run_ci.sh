#!/usr/bin/env bash
# CPU-only CI entry: lit FileCheck + JIT goldens, then pytest if present.
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

cmake --build "$TC_ROOT/build" --target check-tc

export TC_COMPILE="$TC_ROOT/build/bin/tc-compile"
if python3 -c "import pytest" >/dev/null 2>&1; then
  python3 -m pytest "$TC_ROOT/test/e2e" -q
else
  python3 "$TC_ROOT/scripts/check_numeric.py" --tc-compile "$TC_COMPILE" --src "$TC_ROOT"
fi
