# Design

Out-of-tree MLIR compiler for static `fp32` ONNX graphs that are a linear
layer or simpler: `Constant`, `Add`, `Relu`, `MatMul`. Input is ONNX.
Output is either CPU machine code (`-emit=jit`) or PTX text (`-emit=nvptx`).
PTX is compile-only: no CUDA runtime, no device launch.

Pinned LLVM: **llvmorg-23.1.1**, built with `host;NVPTX` and
`MLIR_ENABLE_CUDA_RUNNER=OFF`.

## Pipeline

```
ONNX protobuf
    → ModelInfo (names, shapes, initializers, nodes)
    → tc dialect  (-emit=mlir)
    → Linalg-on-tensors + opts  (-emit=linalg)
    → one-shot bufferize  (-emit=memref)
         ├─ scf/memref → host LLVM → ExecutionEngine  (-emit=llvm|jit)
         └─ scf.parallel → gpu.launch → NVVM → PTX  (-emit=nvptx)
```

The GPU path is a second lowering, not a retarget of host LLVM IR.

## Tools

| Tool | Role |
|---|---|
| `tc-opt` | Pass driver (`mlir-opt` for this project). Registers `tc` plus upstream dialects/passes. |
| `tc-compile` | ONNX → selected emit stage. |

`func.func @main` is the entry. JIT sets `llvm.emit_c_interface` and calls
`_mlir_ciface_main` with packed ranked-memref descriptors.

## Dialect `tc`

ODS in `include/tc/Dialect/TCOps.td`. Ranked `tensor<...xf32>` only.

| Op | Meaning | Verifier |
|---|---|---|
| `tc.constant` | Dense elements payload | Result type matches the attribute |
| `tc.add` | Elementwise add | Same shape, or one operand is `tensor<f32>` |
| `tc.relu` | `max(x, 0)` | Same type in/out |
| `tc.matmul` | `A[M,K] @ B[K,N] → C[M,N]` | Rank-2, contracting `K` matches |

The SSA graph lives in `func.func`. Shape errors fail in the dialect / frontend,
not in LLVM.

Lowering (`--convert-tc-to-linalg`):

- `tc.constant` → `arith.constant`
- `tc.add` → `linalg.generic` + `arith.addf`
- `tc.relu` → `linalg.generic` + `arith.maximumf`
- `tc.matmul` → `linalg.matmul`

## Frontend

`third_party/onnx/onnx.proto` is a field-number subset. A small wire decoder
(`ProtobufReader`) skips unknown fields. No system `libprotobuf`.

`parseONNXFile` → `ModelInfo` → `MLIRGen`. Graph inputs that are not
initializers become `@main` arguments. Initializers become `tc.constant`.
Supported node types: `Add`, `Relu`, `MatMul`, `Constant`.

Sample graphs in `models/` are written by `scripts/gen_models.py`.

## Optimizations

All on Linalg-on-tensors, before bufferization. Canonicalize + CSE after each.

1. **Transpose-B** (`--tc-transpose-matmul-b`): rewrite `linalg.matmul` so the
   contracting dimension of `B` is contiguous (`linalg.matmul_transpose_b`).
2. **Elementwise fusion**: upstream `linalg-fuse-elementwise-ops` (Add+Relu).
3. **Tile and fuse** (`--tc-tile-and-fuse`): tile matmul on `M,N` (not `K`)
   with `-tile-sizes=M,N,K` (default `32,32,32`), then fuse tensor elementwise
   consumers into the tile loops. Reduction dests are zero-filled before tiling.

## CPU backend

After opts:

1. One-shot bufferize with identity layouts and function-boundary bufferization
2. Results → out-params, then buffer deallocation
3. `linalg` → `scf.for`
4. SCF / affine / memref / arith / func / math / ub → LLVM dialect
5. `mlir::ExecutionEngine` JIT

Memref ABI: `{allocated*, aligned*, offset, sizes[rank], strides[rank]}`.
Missing `-input=` files are filled with `0, 1, 2, …`. Outputs print as a shape
header plus row-major floats. Goldens are NumPy (`1e-4` rel / `1e-5` abs);
ONNX Runtime is used too when it is installed.

## NVPTX emit

After the same bufferize:

1. `linalg` → `scf.parallel` (reduction dims stay `scf.for`)
2. `gpu-map-parallel-loops` + `convert-parallel-loops-to-gpu`
3. GPU launches use the parallel `M,N` dimensions as their block grid;
   reductions remain sequential loops inside each block
4. Upstream `gpu-lower-to-nvvm-pipeline` with `cubin-format=isa`, `sm_75`
5. Extract PTX from `gpu.binary` assembly objects

`--tc-outline-gpu-kernel` is a standalone fallback pass for IR that has no
mapped launch; it is not part of the normal `tc-compile -emit=nvptx` path.
No `ptxas`, no cubin, no runtime launch. FileCheck verifies the PTX entry and
the mapped block IDs.

## Tests

`ninja -C build check-tc` is the CPU-only gate (no GPU).

| Area | What it freezes |
|---|---|
| `test/Dialect` | Op assembly + verifier errors |
| `test/Frontend` | `-emit=proto` / `-emit=mlir` |
| `test/Conversion` | `tc` → Linalg |
| `test/Transforms` | Transpose-B, tile+fuse, GPU outline |
| `test/Pipeline` | Every later `-emit=`, plus JIT vs NumPy |
| `test/e2e` | Pytest: JIT vs NumPy/ORT, PTX dump |

`scripts/run_ci.sh` builds `check-tc` and, if pytest is present, `test/e2e`.
