// RUN: tc-opt %s --convert-tc-to-linalg --tc-tile-and-fuse=tile-sizes=2,2,2 | FileCheck %s

func.func @linear(%x: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %w = tc.constant dense<1.0> : tensor<4x4xf32>
  %b = tc.constant dense<0.5> : tensor<4x4xf32>
  %m = tc.matmul %x, %w : tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>
  %a = tc.add %m, %b : tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>
  %r = tc.relu %a : tensor<4x4xf32>
  return %r : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @linear
// CHECK: scf.for
// CHECK: linalg.matmul
// CHECK: arith.addf
// CHECK: arith.maximumf
// CHECK-NOT: tc.
