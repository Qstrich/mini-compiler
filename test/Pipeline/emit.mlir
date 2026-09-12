// RUN: tc-compile %tc_src/models/add.onnx -emit=linalg | FileCheck %s --check-prefix=ADD-LINALG
// RUN: tc-compile %tc_src/models/relu.onnx -emit=linalg | FileCheck %s --check-prefix=RELU-LINALG
// RUN: tc-compile %tc_src/models/matmul.onnx -emit=linalg | FileCheck %s --check-prefix=MATMUL-LINALG
// RUN: tc-compile %tc_src/models/linear.onnx -emit=linalg | FileCheck %s --check-prefix=LINEAR-LINALG
// RUN: tc-compile %tc_src/models/linear.onnx -emit=memref | FileCheck %s --check-prefix=LINEAR-MEMREF
// RUN: tc-compile %tc_src/models/linear.onnx -emit=llvm | FileCheck %s --check-prefix=LINEAR-LLVM

// ADD-LINALG: func.func @main
// ADD-LINALG: linalg.generic
// ADD-LINALG: arith.addf

// RELU-LINALG: func.func @main
// RELU-LINALG: linalg.generic
// RELU-LINALG: arith.maximumf

// MATMUL-LINALG: func.func @main
// MATMUL-LINALG: linalg.matmul

// LINEAR-LINALG: func.func @main
// LINEAR-LINALG: arith.constant
// LINEAR-LINALG: arith.constant
// LINEAR-LINALG: linalg.matmul
// LINEAR-LINALG: linalg.generic
// LINEAR-LINALG: arith.addf
// LINEAR-LINALG: linalg.generic
// LINEAR-LINALG: arith.maximumf
// LINEAR-LINALG-NOT: tc.

// LINEAR-MEMREF: memref.global
// LINEAR-MEMREF: func.func @main
// LINEAR-MEMREF: scf.for
// LINEAR-MEMREF-NOT: linalg.matmul
// LINEAR-MEMREF-NOT: tensor.empty

// LINEAR-LLVM: llvm.func @main
// LINEAR-LLVM-NOT: func.func @main
