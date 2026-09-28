# Mini Tensor Compiler

End-to-end mini tensor compiler via MLIR. ONNX linear-layer graphs lower
through a custom `tc` dialect to Linalg, MemRef, host LLVM, and CPU JIT.

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
pip install -r requirements.txt   # numpy, onnx
# optional second golden: pip install onnxruntime
```

`scripts/env.sh` points tools at `~/src/llvm-project/build/bin`.

## Build LLVM / MLIR (30–90 min)

```bash
source scripts/env.sh
./scripts/build_llvm.sh
# if OOM: JOBS=4 ./scripts/build_llvm.sh

which mlir-opt   # must be ~/src/llvm-project/build/bin/mlir-opt
mlir-opt --version
```

Flags: Release, Clang+LLD, `host`, assertions, utils, ccache, Python bindings
off, CUDA runner off.

## Build this project

Run these commands in **WSL Ubuntu bash**, not Windows PowerShell.

```bash
scripts/build.sh            # configures build/ on first use, then builds
scripts/build.sh check-tc   # build + run the test suite (same as scripts/run_ci.sh)
```

Binaries: `build/bin/tc-opt`, `build/bin/tc-compile`.

## How it fits together

`tc-compile` = ONNX import → three named pass pipelines → JIT. The stage
table, where each rule lives, and how to add an op are in
[docs/DESIGN.md](docs/DESIGN.md#pipeline).

## Flags (`tc-compile`)

| Flag | Meaning |
|---|---|
| `-emit=proto` | Structured ONNX `ModelInfo` dump |
| `-emit=mlir` | `tc` dialect |
| `-emit=linalg` | After lowering to Linalg-on-tensors |
| `-emit=memref` | After one-shot bufferize |
| `-emit=llvm` | Host LLVM dialect |
| `-emit=jit` | JIT-compile and run on CPU |
| `-o file` | Write that dump to `file` (stdout if omitted) |
| `-input=file.npy` | C-contiguous `float32` input for each `@main` arg |

`-emit=jit` fills missing `-input=` files with `0, 1, 2, …` and prints the
result tensor.

```bash
./build/bin/tc-opt test/smoke/empty.mlir
./build/bin/tc-opt test/lowering/tc-to-linalg.mlir --tc-lower-to-linalg
./build/bin/tc-opt test/lowering/tc-to-linalg.mlir \
    --tc-lower-to-linalg --tc-bufferize --tc-lower-to-llvm
./build/bin/tc-compile models/linear.onnx -emit=proto
./build/bin/tc-compile models/linear.onnx -emit=mlir
./build/bin/tc-compile models/linear.onnx -emit=linalg
./build/bin/tc-compile models/linear.onnx -emit=memref
./build/bin/tc-compile models/linear.onnx -emit=llvm
./build/bin/tc-compile models/linear.onnx -emit=jit
```

Sample graphs are defined in `scripts/models.py`. Regenerate the checked-in
`models/*.onnx` after editing it:

```bash
python3 scripts/gen_models.py
```

Compare JIT against NumPy (and ONNX Runtime if installed):

```bash
python3 scripts/check_numeric.py --tc-compile ./build/bin/tc-compile --src .
```

## Tests / CI

`ninja -C build check-tc` (or `scripts/run_ci.sh`) is the CPU test gate:
FileCheck on every `-emit=` stage, clean errors for invalid input, and the
JIT numeric goldens above.

## Status

Done: ONNX → `tc` → Linalg → bufferize → host LLVM → CPU JIT.
