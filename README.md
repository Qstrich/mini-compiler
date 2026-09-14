# Mini Tensor Compiler

End-to-end mini tensor compiler via MLIR. ONNX linear-layer graphs lower
through a custom `tc` dialect to Linalg, MemRef, host LLVM, CPU JIT, and a
compile-only NVPTX/PTX dump (no GPU required).

Architecture notes: [docs/DESIGN.md](docs/DESIGN.md).

## Layout (WSL2 Ubuntu 24.04)

```
WSL2 (Ubuntu 24.04 ext4)
├── System packages (apt): clang, lld, cmake, ninja-build, ccache
├── LLVM / MLIR source and build: ~/src/llvm-project/
├── Out-of-tree compiler: ~/workspace/mini-compiler/
└── Python venv: ~/workspace/mini-compiler/.venv
```

Nothing compiler-owned is installed under `/usr` except the apt host tools.
LLVM is used from its **build tree** (no `sudo make install`).

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
# optional goldens: pip install onnx onnxruntime
```

`scripts/env.sh` points tools at `~/src/llvm-project/build/bin` and puts MLIR
Python packages on `PYTHONPATH`.

## Build LLVM / MLIR (30–90 min)

```bash
source scripts/env.sh
./scripts/build_llvm.sh
# if OOM: JOBS=4 ./scripts/build_llvm.sh

which mlir-opt   # must be ~/src/llvm-project/build/bin/mlir-opt
mlir-opt --version
```

Flags: Release, Clang+LLD, `host;NVPTX`, assertions, utils, ccache, Python
bindings on, CUDA runner off.

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
# or: ./scripts/run_ci.sh
```

Binaries: `build/bin/tc-opt`, `build/bin/tc-compile`.

## Flags (`tc-compile`)

| Flag | Meaning |
|---|---|
| `-emit=proto` | Structured ONNX `ModelInfo` dump |
| `-emit=mlir` | `tc` dialect |
| `-emit=linalg` | After Linalg lowering + opts |
| `-emit=memref` | After one-shot bufferize |
| `-emit=llvm` | Host LLVM dialect |
| `-emit=jit` | JIT-compile and run on CPU |
| `-emit=nvptx` | PTX text (no launch) |
| `-o file` | Write that dump to `file` (stdout if omitted) |
| `-tile-sizes=M,N,K` | Linalg tile sizes (default `32,32,32`) |
| `-input=file.npy` | C-contiguous `float32` input for each `@main` arg |

`-emit=linalg` (and later stages) run transpose-B, elementwise fuse, then
tile+fuse. `-emit=jit` fills missing `-input=` files with `0, 1, 2, …` and
prints the result tensor. For `-emit=nvptx`, `-tile-sizes=M,N,K` controls the
GPU thread-block shape from `M,N`; `K` remains a sequential reduction loop.

```bash
./build/bin/tc-opt test/smoke/empty.mlir
./build/bin/tc-opt test/Conversion/tc-to-linalg.mlir --convert-tc-to-linalg
./build/bin/tc-compile models/linear.onnx -emit=proto
./build/bin/tc-compile models/linear.onnx -emit=mlir
./build/bin/tc-opt test/Transforms/transpose-matmul-b.mlir --tc-transpose-matmul-b
./build/bin/tc-compile models/linear.onnx -emit=linalg
./build/bin/tc-compile models/linear.onnx -emit=linalg -tile-sizes=2,2,2
./build/bin/tc-compile models/linear.onnx -emit=memref
./build/bin/tc-compile models/linear.onnx -emit=llvm
./build/bin/tc-compile models/linear.onnx -emit=jit
./build/bin/tc-compile models/linear.onnx -emit=nvptx -o linear.ptx
```

Regenerate sample ONNX graphs (already checked in under `models/`):

```bash
python3 scripts/gen_models.py
```

Compare JIT against NumPy (and ONNX Runtime if installed):

```bash
python3 scripts/check_numeric.py --tc-compile ./build/bin/tc-compile --src .
# or: python3 -m pytest test/e2e -q
```

## Tests / CI

`ninja -C build check-tc` is the CPU-only gate: FileCheck on every `-emit=`
stage plus JIT goldens. It does not need a GPU. `scripts/run_ci.sh` runs that
suite and pytest when available.

## Status

Done: ONNX → `tc` → Linalg (transpose-B / tile / fuse) → bufferize → host
LLVM → CPU JIT, plus bufferized Linalg → tiled parallel loops → mapped GPU
block/thread launches → NVVM → compile-only PTX dump. The outline pass remains
available as a standalone fallback for un-mapped GPU IR; no CUDA runtime is
required.
