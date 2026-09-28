# Design

Out-of-tree MLIR compiler for static `fp32` ONNX graphs that are a linear
layer or simpler: `Constant`, `Add`, `Relu`, `MatMul`. Input is ONNX.
Output is CPU machine code (`-emit=jit`).

Pinned LLVM: **llvmorg-23.1.1**, built with `host` and
`MLIR_ENABLE_CUDA_RUNNER=OFF`.

## Layout

The source tree is the pipeline. Each stage is one folder under `src/` and one
library, and `test/` uses the same names:

```
src/import/    ONNX file -> ModelInfo -> tc IR          (TCImport)
src/dialect/   the tc IR: ops and their shape rules     (TCDialect)
src/lowering/  tc IR -> Linalg -> loops -> LLVM dialect (TCLowering)
src/runtime/   JIT-compile and run on the host CPU      (TCRuntime)
src/support/   shared helpers
tools/         tc-compile (the driver), tc-opt (run passes on .mlir)
test/          import/ dialect/ lowering/ e2e/
```

Headers sit next to their `.cpp` and `.td` files (as in IREE), not in a
separate `include/` tree: nothing is installed, so a public-header split would
only spread each stage over two places. Includes are stage-relative, e.g.
`#include "dialect/TCOps.h"`, and TableGen output lands at the same relative
path under `build/src/`.

## Pipeline

Each stage has one owner. `tc-compile -emit=X` stops after stage X.

| Stage | Produces | `-emit=` | Code |
|---|---|---|---|
| import: decode | `onnx::GraphProto` structs | | `src/import/OnnxProto.cpp` |
| import: rules | `ModelInfo` (names, shapes, weights, nodes) | `proto` | `src/import/ONNXImporter.cpp` |
| import: generate | `tc` dialect in `func.func @main` | `mlir` | `src/import/MLIRGen.cpp` |
| lowering: `tc-lower-to-linalg` | Linalg on tensors | `linalg` | `src/lowering/Pipeline.cpp` |
| lowering: `tc-bufferize` | memrefs + `scf.for` loops | `memref` | `src/lowering/Pipeline.cpp` |
| lowering: `tc-lower-to-llvm` | LLVM dialect | `llvm` | `src/lowering/Pipeline.cpp` |
| runtime | results on the host CPU | `jit` | `src/runtime/JIT.cpp` |

The three pass pipelines are also registered with `tc-opt` under those names,
so any stage can be run on a `.mlir` file:

```bash
tc-opt in.mlir --tc-lower-to-linalg --tc-bufferize --tc-lower-to-llvm
```

`tools/tc-compile/tc-compile.cpp` is just that table in order (`compile()`).

## Where each rule lives

| Rule | Only place |
|---|---|
| ONNX wire format, field numbers | `OnnxProto.cpp` |
| ONNX restrictions: static shapes, fp32, `Constant` → initializer | `ONNXImporter.cpp` |
| Which ONNX op becomes which `tc` op | `kSupportedOps` in `MLIRGen.cpp` |
| Shape rules (`Add` broadcast, `MatMul` `K` match) | `inferReturnTypes` in `src/dialect/TCOps.cpp` |
| `tc` → Linalg | `src/lowering/TCToLinalg.cpp` |
| Memref ABI for calling `@main` | `src/runtime/JIT.cpp` |

Shape errors are therefore reported once, by the dialect, at the ONNX value
they would define (MLIRGen gives each op a `NameLoc`).

### Adding an op

1. `src/dialect/TCOps.td`: the op, plus `inferReturnTypes` in
   `TCOps.cpp` if the result type is not just the operand type.
2. `MLIRGen.cpp`: one row in `kSupportedOps`.
3. `TCToLinalg.cpp`: one conversion pattern (elementwise ops use
   `buildElementwise`).
4. `scripts/models.py`: a sample model with a NumPy reference, then
   `python3 scripts/gen_models.py`.

## Dialect `tc`

ODS in `src/dialect/TCOps.td`. Ranked `tensor<...xf32>` only.

| Op | Meaning | Result type |
|---|---|---|
| `tc.constant` | Dense elements payload | The attribute's type |
| `tc.add` | Elementwise add | Same shape, or one operand is `tensor<f32>` (broadcast) |
| `tc.relu` | `max(x, 0)` | Operand type |
| `tc.matmul` | `A[M,K] @ B[K,N]` | `tensor<MxNxf32>` |

`tc.add`, `tc.relu` and `tc.matmul` implement `InferTypeOpInterface`; its
verifier rejects any op whose written result type differs from the inferred
one. There are no hand-written verifiers.

Lowering (`--convert-tc-to-linalg`, the core of `tc-lower-to-linalg`):

- `tc.constant` → `arith.constant`
- `tc.add` → `linalg.generic` + `arith.addf`
- `tc.relu` → `linalg.generic` + `arith.maximumf`
- `tc.matmul` → `linalg.fill` with 0, then `linalg.matmul` (it accumulates
  into its init, `C += A·B`, so the init must start at zero)

## Import

`third_party/onnx/onnx.proto` is the reference for field numbers. The decoder
reads only the messages in `src/import/OnnxProto.h`, skips unknown fields,
and does not link `libprotobuf`.

Graph inputs that are not initializers become `@main` arguments;
initializers and `Constant` nodes become `tc.constant`. `@main` carries
`llvm.emit_c_interface` so the JIT can call `_mlir_ciface_main`.

## CPU backend

1. `tc-bufferize`: one-shot bufferize with identity layouts at function
   boundaries; results → caller-allocated out-params; buffer deallocation;
   `linalg` → `scf.for`.
2. `tc-lower-to-llvm`: SCF / affine / memref / arith / func / math / ub →
   LLVM dialect.
3. `runJIT`: `mlir::ExecutionEngine`, calling
   `_mlir_ciface_main(inputs..., outputs...)`.

Memref descriptor ABI: `{allocated*, aligned*, offset, sizes[rank],
strides[rank]}`, built directly around the host buffers. Missing `-input=`
files are filled with `0, 1, 2, …`. Outputs print as a `shape:` header plus
row-major floats.

## Tests

`ninja -C build check-tc` (or `scripts/run_ci.sh`) is the only test gate.

| Area | What it freezes |
|---|---|
| `test/dialect` | Op assembly + inference/verifier errors |
| `test/import` | `-emit=proto` / `-emit=mlir`, and clean import errors |
| `test/lowering` | `tc` → Linalg, and every later `-emit=` |
| `test/e2e` | JIT vs NumPy (and ONNX Runtime if installed) |

Sample graphs and their NumPy references are defined once in
`scripts/models.py`: `gen_models.py` writes `models/*.onnx` from them and
`check_numeric.py` (run by `test/e2e/numeric.mlir`) checks the JIT
against them (`1e-4` rel / `1e-5` abs).
