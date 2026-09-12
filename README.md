# Mini Tensor Compiler

End-to-end mini tensor compiler via MLIR. ONNX linear-layer graphs lower through a custom `tc` dialect to Linalg, MemRef, LLVM, CPU JIT, and (later) NVPTX emit.

## Layout (WSL2 Ubuntu 24.04)

```
WSL2 (Ubuntu 24.04 ext4)
├── System packages (apt): clang, lld, cmake, ninja-build, ccache
├── LLVM / MLIR source and build: ~/src/llvm-project/
├── Out-of-tree compiler: ~/workspace/mini-compiler/
└── Python venv: ~/workspace/mini-compiler/.venv
```

Nothing compiler-owned is installed under `/usr` except the apt host tools. LLVM is used from its **build tree** (no `sudo make install`).

Pinned LLVM tag: **`llvmorg-23.1.1`**.

## One-time host setup

```bash
sudo apt update
sudo apt install -y \
  build-essential cmake ninja-build git \
  clang lld ccache \
  python3 python3-venv python3-dev \
  zlib1g-dev libzstd-dev libxml2-dev libedit-dev libncurses-dev \
  pkg-config
```

## Activate environment

```bash
cd ~/workspace/mini-compiler
python3 -m venv .venv
source scripts/env.sh          # sets TC_ROOT, LLVM_*, PATH, activates .venv
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu
pip install "nanobind==2.9.2" pybind11
# optional: pip install onnx   # only if you prefer the official checker
```

`scripts/env.sh` points tools at `~/src/llvm-project/build/bin` and puts MLIR Python packages on `PYTHONPATH`.

## Build LLVM / MLIR (30–90 min)

```bash
source scripts/env.sh
./scripts/build_llvm.sh
# if OOM: JOBS=4 ./scripts/build_llvm.sh

which mlir-opt   # must be ~/src/llvm-project/build/bin/mlir-opt
mlir-opt --version
```

Flags: Release, Clang+LLD, `host;NVPTX`, assertions, utils, ccache, Python bindings on, CUDA runner off.

## Build this project

Run these commands in **WSL Ubuntu bash**, not Windows PowerShell.

```bash
source scripts/env.sh
cmake -G Ninja -S . -B build \
  -DCMAKE_BUILD_TYPE=Debug \
  -DMLIR_DIR="$MLIR_DIR" \
  -DLLVM_DIR="$LLVM_DIR" \
  -DLLVM_EXTERNAL_LIT="$LLVM_EXTERNAL_LIT"
cmake --build build
ninja -C build check-tc
```

Binaries: `build/bin/tc-opt`, `build/bin/tc-compile`.

```bash
./build/bin/tc-opt test/smoke/empty.mlir
./build/bin/tc-opt test/Conversion/tc-to-linalg.mlir --convert-tc-to-linalg
./build/bin/tc-compile models/linear.onnx -emit=proto
./build/bin/tc-compile models/linear.onnx -emit=mlir
./build/bin/tc-compile models/linear.onnx -emit=linalg
./build/bin/tc-compile models/linear.onnx -emit=memref
./build/bin/tc-compile models/linear.onnx -emit=llvm
./build/bin/tc-compile models/linear.onnx -emit=jit
```

Regenerate sample ONNX graphs (already checked in under `models/`):

```bash
python3 scripts/gen_models.py
```

`tc-compile` implements `-emit=proto|mlir|linalg|memref|llvm|jit`. `-emit=nvptx` is not implemented yet.

`-emit=jit` fills missing `-input=` files with `0, 1, 2, …` and prints the result tensor. Compare against NumPy:

```bash
python3 scripts/check_numeric.py --tc-compile ./build/bin/tc-compile --src .
```

## Status

CPU pipeline: `tc` → Linalg → one-shot bufferize → host LLVM → CPU JIT. Next: Linalg tile/fuse, then NVPTX dump.
