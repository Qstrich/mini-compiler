#!/usr/bin/env bash
# Activate the mini-compiler development environment.
# Usage: source scripts/env.sh

# Resolve repo root even when sourced from elsewhere.
_TC_ENV_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export TC_ROOT="$(cd -- "${_TC_ENV_DIR}/.." && pwd)"
unset _TC_ENV_DIR

export LLVM_SRC="${LLVM_SRC:-$HOME/src/llvm-project}"
export LLVM_BUILD="${LLVM_BUILD:-$LLVM_SRC/build}"
export LLVM_TAG="${LLVM_TAG:-llvmorg-23.1.1}"

export CC="${CC:-clang}"
export CXX="${CXX:-clang++}"

export MLIR_DIR="$LLVM_BUILD/lib/cmake/mlir"
export LLVM_DIR="$LLVM_BUILD/lib/cmake/llvm"
export LLVM_EXTERNAL_LIT="$LLVM_BUILD/bin/llvm-lit"
export CMAKE_PREFIX_PATH="$LLVM_BUILD${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"

# Prefer tools from our LLVM build over anything in /usr.
case ":$PATH:" in
  *":$LLVM_BUILD/bin:"*) ;;
  *) export PATH="$LLVM_BUILD/bin:$PATH" ;;
esac

# MLIR Python packages live next to the build tree when bindings are enabled.
_MLIR_PY="$LLVM_BUILD/tools/mlir/python_packages/mlir_core"
if [[ -d "$_MLIR_PY" ]]; then
  case ":${PYTHONPATH:-}:" in
    *":$_MLIR_PY:"*) ;;
    *) export PYTHONPATH="$_MLIR_PY${PYTHONPATH:+:$PYTHONPATH}" ;;
  esac
fi
unset _MLIR_PY

# Project venv for PyTorch baselines (MLIR Python comes via PYTHONPATH above).
if [[ -f "$TC_ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$TC_ROOT/.venv/bin/activate"
fi

echo "TC_ROOT=$TC_ROOT"
echo "LLVM_SRC=$LLVM_SRC"
echo "LLVM_BUILD=$LLVM_BUILD"
if command -v mlir-opt >/dev/null 2>&1; then
  echo "mlir-opt=$(command -v mlir-opt)"
else
  echo "mlir-opt=not found (run ./scripts/build_llvm.sh)"
fi
