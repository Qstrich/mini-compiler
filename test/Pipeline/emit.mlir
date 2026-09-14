// RUN: tc-compile %tc_src/models/add.onnx -emit=linalg -tile-sizes=1,1 | FileCheck %s --check-prefix=ADD-LINALG
// RUN: tc-compile %tc_src/models/relu.onnx -emit=linalg -tile-sizes=1,1 | FileCheck %s --check-prefix=RELU-LINALG
// RUN: tc-compile %tc_src/models/matmul.onnx -emit=linalg | FileCheck %s --check-prefix=MATMUL-LINALG
// RUN: tc-compile %tc_src/models/linear.onnx -emit=linalg | FileCheck %s --check-prefix=LINEAR-LINALG
// RUN: tc-compile %tc_src/models/linear.onnx -emit=memref | FileCheck %s --check-prefix=LINEAR-MEMREF
// RUN: tc-compile %tc_src/models/linear.onnx -emit=llvm | FileCheck %s --check-prefix=LINEAR-LLVM
// RUN: tc-compile %tc_src/models/linear.onnx -emit=linalg -tile-sizes=2,2,2 | FileCheck %s --check-prefix=LINEAR-TILED

// ADD-LINALG: func.func @main
// ADD-LINALG: scf.for
// ADD-LINALG: linalg.generic
// ADD-LINALG: arith.addf

// RELU-LINALG: func.func @main
// RELU-LINALG: scf.for
// RELU-LINALG: linalg.generic
// RELU-LINALG: arith.maximumf

// MATMUL-LINALG: func.func @main
// MATMUL-LINALG: linalg.transpose
// MATMUL-LINALG: linalg.matmul

// Default 32x32x32 tiles are larger than this 2x3 output, so one-trip
// `scf.for`s are canonicalized away. The constant weight is folded through
// transpose-B into a 3x4 tensor, and add/relu fuse into one generic.
// LINEAR-LINALG: func.func @main
// LINEAR-LINALG: tensor<3x4xf32>
// LINEAR-LINALG: linalg.matmul
// LINEAR-LINALG: arith.addf
// LINEAR-LINALG: arith.maximumf
// LINEAR-LINALG-NOT: tc.

// LINEAR-TILED: func.func @main
// LINEAR-TILED: scf.for
// LINEAR-TILED: linalg.matmul
// LINEAR-TILED: arith.addf
// LINEAR-TILED: arith.maximumf

// LINEAR-MEMREF: memref.global
// LINEAR-MEMREF: func.func @main
// LINEAR-MEMREF: scf.for
// LINEAR-MEMREF-NOT: linalg.matmul
// LINEAR-MEMREF-NOT: tensor.empty

// LINEAR-LLVM: llvm.func @main
// LINEAR-LLVM-NOT: func.func @main
