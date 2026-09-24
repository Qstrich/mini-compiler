#!/usr/bin/env bash
# Clone and build LLVM/MLIR into ~/src/llvm-project/build (no sudo install).
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"

LLVM_TAG="${LLVM_TAG:-llvmorg-23.1.1}"
JOBS="${JOBS:-$(nproc)}"

mkdir -p "$(dirname "$LLVM_SRC")"

if [[ ! -d "$LLVM_SRC/.git" ]]; then
  echo "Cloning LLVM $LLVM_TAG into $LLVM_SRC ..."
  git clone --depth 1 --branch "$LLVM_TAG" \
    https://github.com/llvm/llvm-project.git "$LLVM_SRC"
else
  echo "Using existing LLVM checkout at $LLVM_SRC"
fi

mkdir -p "$LLVM_BUILD"

# MLIR Python bindings (LLVM 23.1.x) need nanobind 2.9.x in the active venv.
if [[ -x "$TC_ROOT/.venv/bin/pip" ]]; then
  "$TC_ROOT/.venv/bin/pip" install -q "nanobind==2.9.2" pybind11 numpy
fi

cmake -G Ninja -S "$LLVM_SRC/llvm" -B "$LLVM_BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DPython3_EXECUTABLE="$TC_ROOT/.venv/bin/python" \
  -DPython3_FIND_VIRTUALENV=ONLY \
  -DLLVM_USE_LINKER=lld \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_INSTALL_UTILS=ON \
  -DLLVM_CCACHE_BUILD=ON \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DMLIR_ENABLE_CUDA_RUNNER=OFF

echo "Building LLVM/MLIR with -j${JOBS} (this can take 30–90 minutes)..."
echo "If the machine OOMs, re-run with: JOBS=4 ./scripts/build_llvm.sh"
cmake --build "$LLVM_BUILD" -j"$JOBS"

echo
echo "Done. Activate with: source $TC_ROOT/scripts/env.sh"
echo "Verify: which mlir-opt && mlir-opt --version"
